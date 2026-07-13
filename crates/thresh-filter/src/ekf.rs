//! Extended Kalman Filter with Jacobian-based linearization.

use nalgebra::{DMatrix, DVector};

use crate::UpdateOutcome;
use crate::traits::MotionModel;

/// Extended Kalman Filter state.
pub struct ExtendedKalmanFilter {
    /// Current state estimate.
    pub x: DVector<f64>,
    /// Current covariance estimate.
    pub p: DMatrix<f64>,
}

impl ExtendedKalmanFilter {
    /// Create a new EKF with initial state and covariance.
    pub fn new(x: DVector<f64>, p: DMatrix<f64>) -> Self {
        Self { x, p }
    }

    /// Predict step using a nonlinear motion model.
    pub fn predict(&mut self, model: &dyn MotionModel, dt: f64) {
        let f_jac = model.jacobian(&self.x, dt);
        let q = model.process_noise(dt);

        // Nonlinear state propagation
        self.x = model.predict(&self.x, dt);
        // Linearized covariance propagation
        self.p = &f_jac * &self.p * f_jac.transpose() + q;
    }

    /// Update step given measurement z, nonlinear observation function h(x),
    /// observation Jacobian H, and noise R.
    ///
    /// `h_of_x` is h(x_predicted), the expected measurement.
    ///
    /// Returns the [`UpdateOutcome`] diagnostics of this update: the
    /// innovation `y = z − h(x_pred)` and the linearized covariance
    /// `S = H_k·P_pred·H_kᵀ + R` actually used to form the gain, plus the
    /// NIS `yᵀ S⁻¹ y` computed at the same `S` factorization. Diagnostics
    /// exist **only** as this return value — nothing is stored on the
    /// filter, so there is no stale or pre-first-update query surface. For
    /// NEES, read the predicted state and covariance from the `pub` `x` /
    /// `p` fields after [`Self::predict`] and before this call (no
    /// accessors alias those fields).
    pub fn update(
        &mut self,
        z: &DVector<f64>,
        h_of_x: &DVector<f64>,
        h_jac: &DMatrix<f64>,
        r: &DMatrix<f64>,
    ) -> UpdateOutcome {
        // Innovation
        let y = z - h_of_x;
        // Innovation covariance
        let s = h_jac * &self.p * h_jac.transpose() + r;

        let s_inv = s
            .clone()
            .try_inverse()
            .expect("Innovation covariance S is singular");
        let k = &self.p * h_jac.transpose() * &s_inv;

        // NIS at the factorization that produced the gain: yᵀ S⁻¹ y.
        let nis = (y.transpose() * &s_inv * &y)[(0, 0)];

        // State update
        self.x = &self.x + &k * &y;

        // Joseph-form covariance
        let n = self.x.len();
        let i_kh = DMatrix::identity(n, n) - &k * h_jac;
        self.p = &i_kh * &self.p * i_kh.transpose() + &k * r * k.transpose();

        UpdateOutcome {
            innovation: y,
            innovation_covariance: s,
            nis,
        }
    }

    /// Linear update (for sensors with linear observation models).
    ///
    /// Returns the same [`UpdateOutcome`] diagnostics as [`Self::update`].
    pub fn update_linear(
        &mut self,
        z: &DVector<f64>,
        h: &DMatrix<f64>,
        r: &DMatrix<f64>,
    ) -> UpdateOutcome {
        let h_of_x = h * &self.x;
        self.update(z, &h_of_x, h, r)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::ctrv::Ctrv;

    #[test]
    fn ekf_tracks_turning_target() {
        let model = Ctrv::new(1.0, 0.1);
        let mut ekf = ExtendedKalmanFilter::new(
            DVector::from_column_slice(&[0.0, 0.0, 0.0, 100.0, 0.1]),
            DMatrix::identity(5, 5) * 10.0,
        );

        // H observes x, y only
        let h = DMatrix::from_row_slice(2, 5, &[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]);
        let r = DMatrix::identity(2, 2) * 25.0;

        // True trajectory: CTRV with omega=0.1 rad/s, v=100 m/s
        let true_model = Ctrv::new(0.0, 0.0);
        let mut true_state = DVector::from_column_slice(&[0.0, 0.0, 0.0, 100.0, 0.1]);

        for _ in 0..50 {
            true_state = true_model.predict(&true_state, 0.1);
            let z = DVector::from_column_slice(&[true_state[0], true_state[1]]);

            ekf.predict(&model, 0.1);
            let h_of_x = &h * &ekf.x;
            ekf.update(&z, &h_of_x, &h, &r);
        }

        // Should track reasonably close
        let pos_err =
            ((ekf.x[0] - true_state[0]).powi(2) + (ekf.x[1] - true_state[1]).powi(2)).sqrt();
        assert!(pos_err < 20.0, "Position error too large: {pos_err}");
    }

    // Task 1.3 (eval-consistency-metrics), spec "EKF diagnostics reflect the
    // linearized update": outcome matches z − h(x_pred) and H_k·P_pred·H_kᵀ + R
    // at the predicted-state Jacobian, against hand-computed values.
    #[test]
    fn ekf_update_outcome_matches_linearized_closed_form() {
        // Range-only measurement h(x) = √(x₀² + x₁²) at x_pred = [3, 4]:
        //   h(x_pred) = 5, Jacobian at x_pred: H_k = [3/5, 4/5] = [0.6, 0.8]
        //   P_pred = I₂, R = [1], z = [6]
        //   innovation = z − h(x_pred) = 6 − 5 = 1
        //   S = H_k·P_pred·H_kᵀ + R = (0.36 + 0.64) + 1 = 2
        //   NIS = 1² / 2 = 0.5
        let mut ekf = ExtendedKalmanFilter::new(
            DVector::from_column_slice(&[3.0, 4.0]),
            DMatrix::identity(2, 2),
        );
        let x_pred = ekf.x.clone();
        let h_of_x = DVector::from_element(1, (x_pred[0].powi(2) + x_pred[1].powi(2)).sqrt());
        let h_jac = DMatrix::from_row_slice(1, 2, &[0.6, 0.8]);
        let r = DMatrix::from_element(1, 1, 1.0);
        let z = DVector::from_element(1, 6.0);

        let outcome = ekf.update(&z, &h_of_x, &h_jac, &r);
        assert!((outcome.innovation[0] - 1.0).abs() < 1e-12);
        assert!((outcome.innovation_covariance[(0, 0)] - 2.0).abs() < 1e-12);
        assert!((outcome.nis - 0.5).abs() < 1e-12);
    }
}
