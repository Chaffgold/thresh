//! Coordinated Turn motion model.
//!
//! State: [x, vx, y, vy, omega] (5D).
//! Quasi-linear given omega; EKF treatment standard.

use nalgebra::{DMatrix, DVector};

use crate::traits::MotionModel;

/// Coordinated Turn model in Cartesian coordinates.
///
/// State: [x, vx, y, vy, omega].
/// The transition is quasi-linear given omega, but since omega
/// evolves, EKF/UKF treatment is standard.
pub struct CoordinatedTurn {
    /// Process noise for velocity components (m/s²).
    pub sigma_v: f64,
    /// Process noise for turn rate (rad/s²).
    pub sigma_omega: f64,
    /// Threshold for omega degeneracy.
    pub omega_epsilon: f64,
}

impl CoordinatedTurn {
    pub fn new(sigma_v: f64, sigma_omega: f64) -> Self {
        Self {
            sigma_v,
            sigma_omega,
            omega_epsilon: 1e-6,
        }
    }
}

impl MotionModel for CoordinatedTurn {
    fn state_dim(&self) -> usize {
        5
    }

    fn predict(&self, state: &DVector<f64>, dt: f64) -> DVector<f64> {
        let x = state[0];
        let vx = state[1];
        let y = state[2];
        let vy = state[3];
        let omega = state[4];

        let mut next = DVector::zeros(5);

        if omega.abs() < self.omega_epsilon {
            // Degenerate: constant velocity
            next[0] = x + vx * dt;
            next[1] = vx;
            next[2] = y + vy * dt;
            next[3] = vy;
            next[4] = omega;
        } else {
            let sin_wt = (omega * dt).sin();
            let cos_wt = (omega * dt).cos();

            next[0] = x + sin_wt / omega * vx - (1.0 - cos_wt) / omega * vy;
            next[1] = cos_wt * vx - sin_wt * vy;
            next[2] = y + (1.0 - cos_wt) / omega * vx + sin_wt / omega * vy;
            next[3] = sin_wt * vx + cos_wt * vy;
            next[4] = omega;
        }
        next
    }

    fn jacobian(&self, state: &DVector<f64>, dt: f64) -> DMatrix<f64> {
        let vx = state[1];
        let vy = state[3];
        let omega = state[4];

        let mut f = DMatrix::identity(5, 5);

        if omega.abs() < self.omega_epsilon {
            f[(0, 1)] = dt;
            f[(2, 3)] = dt;
        } else {
            let sin_wt = (omega * dt).sin();
            let cos_wt = (omega * dt).cos();
            let w2 = omega * omega;

            // Row 0: x
            f[(0, 1)] = sin_wt / omega;
            f[(0, 3)] = -(1.0 - cos_wt) / omega;
            f[(0, 4)] = dt * cos_wt / omega * vx - sin_wt / w2 * vx + dt * sin_wt / omega * vy
                - (1.0 - cos_wt) / w2 * vy;

            // Row 1: vx
            f[(1, 1)] = cos_wt;
            f[(1, 3)] = -sin_wt;
            f[(1, 4)] = -dt * sin_wt * vx - dt * cos_wt * vy;

            // Row 2: y
            f[(2, 1)] = (1.0 - cos_wt) / omega;
            f[(2, 3)] = sin_wt / omega;
            f[(2, 4)] = dt * sin_wt / omega * vx - (1.0 - cos_wt) / w2 * vx
                + dt * cos_wt / omega * vy
                - sin_wt / w2 * vy;

            // Row 3: vy
            f[(3, 1)] = sin_wt;
            f[(3, 3)] = cos_wt;
            f[(3, 4)] = dt * cos_wt * vx - dt * sin_wt * vy;

            // Row 4: omega is identity (already set)
        }

        f
    }

    fn process_noise(&self, dt: f64) -> DMatrix<f64> {
        let sv2 = self.sigma_v * self.sigma_v;
        let so2 = self.sigma_omega * self.sigma_omega;
        let dt2 = dt * dt;

        let mut q = DMatrix::zeros(5, 5);
        // Discrete white noise acceleration model per axis
        q[(0, 0)] = dt2 * dt2 / 4.0 * sv2;
        q[(0, 1)] = dt2 * dt / 2.0 * sv2;
        q[(1, 0)] = dt2 * dt / 2.0 * sv2;
        q[(1, 1)] = dt2 * sv2;

        q[(2, 2)] = dt2 * dt2 / 4.0 * sv2;
        q[(2, 3)] = dt2 * dt / 2.0 * sv2;
        q[(3, 2)] = dt2 * dt / 2.0 * sv2;
        q[(3, 3)] = dt2 * sv2;

        q[(4, 4)] = dt2 * so2;
        q
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    #[test]
    fn ct_straight_line() {
        let ct = CoordinatedTurn::new(1.0, 0.1);
        let state = DVector::from_column_slice(&[0.0, 100.0, 0.0, 0.0, 0.0]);
        let next = ct.predict(&state, 1.0);
        assert!((next[0] - 100.0).abs() < 1e-10);
        assert!(next[2].abs() < 1e-10);
    }

    #[test]
    fn ct_full_circle() {
        let ct = CoordinatedTurn::new(1.0, 0.1);
        // omega = 2*PI rad/s, speed 100 m/s -> full circle in 1 second
        let state = DVector::from_column_slice(&[0.0, 100.0, 0.0, 0.0, 2.0 * PI]);
        let next = ct.predict(&state, 1.0);
        // After full circle: back to start, velocities restored
        assert!((next[0]).abs() < 1e-8);
        assert!((next[2]).abs() < 1e-8);
        assert!((next[1] - 100.0).abs() < 1e-8);
    }
}
