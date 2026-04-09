//! Constant Acceleration (CA) motion model.
//!
//! State: [x, vx, ax, y, vy, ay, z, vz, az] (9D).

use nalgebra::{DMatrix, DVector};

use crate::traits::{LinearModel, MotionModel};

/// Constant Acceleration model in 3D.
///
/// State vector: [x, vx, ax, y, vy, ay, z, vz, az].
pub struct ConstantAcceleration {
    /// Process noise spectral density (jerk noise, m/s³).
    pub sigma_j: f64,
}

impl ConstantAcceleration {
    pub fn new(sigma_j: f64) -> Self {
        Self { sigma_j }
    }
}

impl MotionModel for ConstantAcceleration {
    fn state_dim(&self) -> usize {
        9
    }

    fn predict(&self, state: &DVector<f64>, dt: f64) -> DVector<f64> {
        let f = self.transition_matrix(dt);
        &f * state
    }

    fn jacobian(&self, _state: &DVector<f64>, dt: f64) -> DMatrix<f64> {
        self.transition_matrix(dt)
    }

    fn process_noise(&self, dt: f64) -> DMatrix<f64> {
        let q = self.sigma_j * self.sigma_j;
        let dt2 = dt * dt;
        let dt3 = dt2 * dt;
        let dt4 = dt3 * dt;
        let dt5 = dt4 * dt;

        // Per-axis 3x3 block
        let mut noise = DMatrix::zeros(9, 9);
        for i in 0..3 {
            let p = i * 3;
            noise[(p, p)] = dt5 / 20.0 * q;
            noise[(p, p + 1)] = dt4 / 8.0 * q;
            noise[(p, p + 2)] = dt3 / 6.0 * q;
            noise[(p + 1, p)] = dt4 / 8.0 * q;
            noise[(p + 1, p + 1)] = dt3 / 3.0 * q;
            noise[(p + 1, p + 2)] = dt2 / 2.0 * q;
            noise[(p + 2, p)] = dt3 / 6.0 * q;
            noise[(p + 2, p + 1)] = dt2 / 2.0 * q;
            noise[(p + 2, p + 2)] = dt * q;
        }
        noise
    }
}

impl LinearModel for ConstantAcceleration {
    fn transition_matrix(&self, dt: f64) -> DMatrix<f64> {
        let dt2 = dt * dt;
        let mut f = DMatrix::identity(9, 9);
        for i in 0..3 {
            let p = i * 3;
            f[(p, p + 1)] = dt;
            f[(p, p + 2)] = dt2 / 2.0;
            f[(p + 1, p + 2)] = dt;
        }
        f
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ca_uniform_acceleration() {
        let ca = ConstantAcceleration::new(1.0);
        // ax=10 m/s², starting from rest
        let state = DVector::from_column_slice(&[0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        let next = ca.predict(&state, 1.0);
        // x = 0.5 * a * t^2 = 5
        assert!((next[0] - 5.0).abs() < 1e-10);
        // vx = a * t = 10
        assert!((next[1] - 10.0).abs() < 1e-10);
        // ax unchanged
        assert!((next[2] - 10.0).abs() < 1e-10);
    }
}
