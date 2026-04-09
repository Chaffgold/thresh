//! Constant Turn Rate and Velocity (CTRV) motion model.
//!
//! State: [x, y, theta, v, omega] (5D).
//! Nonlinear — requires EKF or UKF.

use nalgebra::{DMatrix, DVector};

use crate::traits::MotionModel;

/// CTRV model for 2D planar tracking with heading.
///
/// State: [x, y, theta, v, omega] where theta is heading, v is speed,
/// omega is yaw rate.
pub struct Ctrv {
    /// Process noise standard deviation for velocity (m/s²).
    pub sigma_v: f64,
    /// Process noise standard deviation for yaw rate (rad/s²).
    pub sigma_omega: f64,
    /// Threshold below which omega is considered zero (degenerates to CV).
    pub omega_epsilon: f64,
}

impl Ctrv {
    pub fn new(sigma_v: f64, sigma_omega: f64) -> Self {
        Self {
            sigma_v,
            sigma_omega,
            omega_epsilon: 1e-6,
        }
    }
}

impl MotionModel for Ctrv {
    fn state_dim(&self) -> usize {
        5
    }

    fn predict(&self, state: &DVector<f64>, dt: f64) -> DVector<f64> {
        let x = state[0];
        let y = state[1];
        let theta = state[2];
        let v = state[3];
        let omega = state[4];

        let mut next = DVector::zeros(5);

        if omega.abs() < self.omega_epsilon {
            // Degenerate case: straight-line motion
            next[0] = x + v * theta.cos() * dt;
            next[1] = y + v * theta.sin() * dt;
            next[2] = theta;
            next[3] = v;
            next[4] = omega;
        } else {
            let v_over_w = v / omega;
            let new_theta = theta + omega * dt;
            next[0] = x + v_over_w * (new_theta.sin() - theta.sin());
            next[1] = y + v_over_w * (-(new_theta.cos()) + theta.cos());
            next[2] = new_theta;
            next[3] = v;
            next[4] = omega;
        }
        next
    }

    fn jacobian(&self, state: &DVector<f64>, dt: f64) -> DMatrix<f64> {
        let theta = state[2];
        let v = state[3];
        let omega = state[4];

        let mut f = DMatrix::identity(5, 5);

        if omega.abs() < self.omega_epsilon {
            // Jacobian of straight-line model
            f[(0, 2)] = -v * theta.sin() * dt;
            f[(0, 3)] = theta.cos() * dt;
            f[(1, 2)] = v * theta.cos() * dt;
            f[(1, 3)] = theta.sin() * dt;
        } else {
            let v_over_w = v / omega;
            let new_theta = theta + omega * dt;

            // dx/dtheta
            f[(0, 2)] = v_over_w * (new_theta.cos() - theta.cos());
            // dx/dv
            f[(0, 3)] = (new_theta.sin() - theta.sin()) / omega;
            // dx/domega
            f[(0, 4)] = v * dt * new_theta.cos() / omega
                - v * (new_theta.sin() - theta.sin()) / (omega * omega);

            // dy/dtheta
            f[(1, 2)] = v_over_w * (new_theta.sin() - theta.sin());
            // dy/dv
            f[(1, 3)] = (-new_theta.cos() + theta.cos()) / omega;
            // dy/domega
            f[(1, 4)] = v * dt * new_theta.sin() / omega
                - v * (-new_theta.cos() + theta.cos()) / (omega * omega);

            // dtheta/domega
            f[(2, 4)] = dt;
        }

        f
    }

    fn process_noise(&self, dt: f64) -> DMatrix<f64> {
        let dt2 = dt * dt;
        let mut q = DMatrix::zeros(5, 5);
        // Acceleration noise affects v, which propagates to x, y
        let sv2 = self.sigma_v * self.sigma_v;
        let so2 = self.sigma_omega * self.sigma_omega;

        q[(0, 0)] = dt2 * dt2 / 4.0 * sv2;
        q[(1, 1)] = dt2 * dt2 / 4.0 * sv2;
        q[(2, 2)] = dt2 * so2;
        q[(3, 3)] = dt2 * sv2;
        q[(3, 0)] = dt2 * dt / 2.0 * sv2;
        q[(0, 3)] = dt2 * dt / 2.0 * sv2;
        q[(4, 4)] = dt2 * so2;
        q[(4, 2)] = dt * so2;
        q[(2, 4)] = dt * so2;

        q
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    #[test]
    fn ctrv_straight_line_when_omega_zero() {
        let ctrv = Ctrv::new(1.0, 0.1);
        // Heading 0 (along +x), speed 100 m/s, omega = 0
        let state = DVector::from_column_slice(&[0.0, 0.0, 0.0, 100.0, 0.0]);
        let next = ctrv.predict(&state, 1.0);
        assert!((next[0] - 100.0).abs() < 1e-10);
        assert!(next[1].abs() < 1e-10);
    }

    #[test]
    fn ctrv_circular_motion() {
        let ctrv = Ctrv::new(1.0, 0.1);
        // Full circle: omega = 2*PI, dt = 1 -> should return to start
        let state = DVector::from_column_slice(&[0.0, 0.0, 0.0, 100.0, 2.0 * PI]);
        let next = ctrv.predict(&state, 1.0);
        assert!((next[0] - 0.0).abs() < 1e-8);
        assert!((next[1] - 0.0).abs() < 1e-8);
    }

    #[test]
    fn ctrv_degenerate_omega_small() {
        let ctrv = Ctrv::new(1.0, 0.1);
        // Very small omega should behave like straight line
        let state = DVector::from_column_slice(&[0.0, 0.0, 0.0, 100.0, 1e-10]);
        let next = ctrv.predict(&state, 1.0);
        assert!((next[0] - 100.0).abs() < 1e-3);
        assert!(next[1].abs() < 1e-3);
    }

    #[test]
    fn ctrv_quarter_turn() {
        let ctrv = Ctrv::new(1.0, 0.1);
        // omega = PI/2, dt = 1 -> 90 degree turn
        let omega = PI / 2.0;
        let v = 100.0;
        let state = DVector::from_column_slice(&[0.0, 0.0, 0.0, v, omega]);
        let next = ctrv.predict(&state, 1.0);
        // radius = v/omega
        let r = v / omega;
        // After 90 deg: x = r*sin(pi/2) = r, y = r*(1-cos(pi/2)) = r
        assert!((next[0] - r).abs() < 1e-8);
        assert!((next[1] - r).abs() < 1e-8);
    }
}
