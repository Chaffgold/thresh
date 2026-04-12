//! Constant Velocity (CV) motion model.
//!
//! State: [x, vx, y, vy, z, vz] (6D).

use nalgebra::{DMatrix, DVector};

use crate::traits::{LinearModel, MotionModel};

/// Constant Velocity model in 3D.
///
/// State vector: [x, vx, y, vy, z, vz].
/// Assumes constant velocity with white noise acceleration.
pub struct ConstantVelocity {
    /// Process noise spectral density (acceleration noise, m/s²).
    pub sigma_a: f64,
}

impl ConstantVelocity {
    pub fn new(sigma_a: f64) -> Self {
        Self { sigma_a }
    }
}

impl MotionModel for ConstantVelocity {
    fn state_dim(&self) -> usize {
        6
    }

    fn predict(&self, state: &DVector<f64>, dt: f64) -> DVector<f64> {
        let f = self.transition_matrix(dt);
        &f * state
    }

    fn jacobian(&self, _state: &DVector<f64>, dt: f64) -> DMatrix<f64> {
        self.transition_matrix(dt)
    }

    fn process_noise(&self, dt: f64) -> DMatrix<f64> {
        let q = self.sigma_a * self.sigma_a;
        let dt2 = dt * dt;
        let dt3 = dt2 * dt;
        let dt4 = dt3 * dt;

        // Per-axis block: [[dt^4/4, dt^3/2], [dt^3/2, dt^2]]
        let mut noise = DMatrix::zeros(6, 6);
        for i in 0..3 {
            let p = i * 2;
            noise[(p, p)] = dt4 / 4.0 * q;
            noise[(p, p + 1)] = dt3 / 2.0 * q;
            noise[(p + 1, p)] = dt3 / 2.0 * q;
            noise[(p + 1, p + 1)] = dt2 * q;
        }
        noise
    }
}

impl LinearModel for ConstantVelocity {
    fn transition_matrix(&self, dt: f64) -> DMatrix<f64> {
        let mut f = DMatrix::identity(6, 6);
        // x += vx * dt, y += vy * dt, z += vz * dt
        f[(0, 1)] = dt;
        f[(2, 3)] = dt;
        f[(4, 5)] = dt;
        f
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cv_straight_line() {
        let cv = ConstantVelocity::new(1.0);
        // Moving at 100 m/s along x
        let state = DVector::from_column_slice(&[0.0, 100.0, 0.0, 0.0, 0.0, 0.0]);
        let next = cv.predict(&state, 1.0);
        assert!((next[0] - 100.0).abs() < 1e-10);
        assert!((next[1] - 100.0).abs() < 1e-10); // velocity unchanged
    }

    #[test]
    fn cv_process_noise_symmetric() {
        let cv = ConstantVelocity::new(5.0);
        let q = cv.process_noise(0.1);
        for i in 0..6 {
            for j in 0..6 {
                assert!((q[(i, j)] - q[(j, i)]).abs() < 1e-15);
            }
        }
    }
}
