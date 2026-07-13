//! Linear Kalman Filter with Joseph-form covariance update.

use nalgebra::{DMatrix, DVector};

use crate::UpdateOutcome;
use crate::traits::LinearModel;

/// Linear Kalman Filter state.
pub struct KalmanFilter {
    /// Current state estimate.
    pub x: DVector<f64>,
    /// Current covariance estimate.
    pub p: DMatrix<f64>,
}

impl KalmanFilter {
    /// Create a new KF with initial state and covariance.
    pub fn new(x: DVector<f64>, p: DMatrix<f64>) -> Self {
        Self { x, p }
    }

    /// Predict step using a linear motion model.
    pub fn predict(&mut self, model: &dyn LinearModel) -> f64 {
        self.predict_dt(model, 1.0)
    }

    /// Predict step with explicit dt.
    pub fn predict_dt(&mut self, model: &dyn LinearModel, dt: f64) -> f64 {
        let f = model.transition_matrix(dt);
        let q = model.process_noise(dt);

        self.x = &f * &self.x;
        self.p = &f * &self.p * f.transpose() + q;
        dt
    }

    /// Update step given measurement z, observation matrix H, and noise R.
    ///
    /// Uses the Joseph form for numerical stability.
    ///
    /// Returns the [`UpdateOutcome`] diagnostics of this update: the
    /// innovation `y = z − H·x_pred` and covariance `S = H·P_pred·Hᵀ + R`
    /// actually used to form the gain, plus the NIS `yᵀ S⁻¹ y` computed at
    /// the same `S` factorization. Diagnostics exist **only** as this return
    /// value — nothing is stored on the filter, so there is no stale or
    /// pre-first-update query surface. For NEES, read the predicted state
    /// and covariance from the `pub` `x` / `p` fields after `predict` and
    /// before this call (no accessors alias those fields); contrast the
    /// pre-update [`Self::innovation`] / [`Self::innovation_covariance`]
    /// gating helpers, which compute a hypothetical residual for a candidate
    /// `z` without updating.
    pub fn update(
        &mut self,
        z: &DVector<f64>,
        h: &DMatrix<f64>,
        r: &DMatrix<f64>,
    ) -> UpdateOutcome {
        // Innovation
        let y = z - h * &self.x;
        // Innovation covariance
        let s = h * &self.p * h.transpose() + r;

        // Kalman gain
        let s_inv = s
            .clone()
            .try_inverse()
            .expect("Innovation covariance S is singular");
        let k = &self.p * h.transpose() * &s_inv;

        // NIS at the factorization that produced the gain: yᵀ S⁻¹ y.
        let nis = (y.transpose() * &s_inv * &y)[(0, 0)];

        // State update
        self.x = &self.x + &k * &y;

        // Joseph-form covariance update: P = (I - KH)P(I - KH)' + KRK'
        let n = self.x.len();
        let i_kh = DMatrix::identity(n, n) - &k * h;
        self.p = &i_kh * &self.p * i_kh.transpose() + &k * r * k.transpose();

        UpdateOutcome {
            innovation: y,
            innovation_covariance: s,
            nis,
        }
    }

    /// Return the innovation (residual) for a measurement without updating.
    ///
    /// Pre-update gating helper: the **hypothetical residual for a candidate
    /// `z`** against the current state, computed before committing to an
    /// update. For the diagnostics of the update that actually happened, use
    /// the [`UpdateOutcome`] returned by [`Self::update`].
    pub fn innovation(&self, z: &DVector<f64>, h: &DMatrix<f64>) -> DVector<f64> {
        z - h * &self.x
    }

    /// Return the innovation covariance S.
    ///
    /// Pre-update gating helper paired with [`Self::innovation`]: the `S` a
    /// hypothetical update at the current state would use, computed before
    /// committing to it. For the `S` of the update that actually happened,
    /// use the [`UpdateOutcome`] returned by [`Self::update`].
    pub fn innovation_covariance(&self, h: &DMatrix<f64>, r: &DMatrix<f64>) -> DMatrix<f64> {
        h * &self.p * h.transpose() + r
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::cv::ConstantVelocity;

    #[test]
    fn kf_converges_on_linear_system() {
        let model = ConstantVelocity::new(0.1);
        let mut kf = KalmanFilter::new(
            DVector::from_column_slice(&[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            DMatrix::identity(6, 6) * 1000.0,
        );

        // Observe only position (x, y, z)
        let h = DMatrix::from_row_slice(
            3,
            6,
            &[
                1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                1.0, 0.0,
            ],
        );
        let r = DMatrix::identity(3, 3) * 1.0;

        // True target at (100, 200, 50) moving at (10, 5, 0)
        for step in 0..100 {
            kf.predict_dt(&model, 1.0);
            let t = (step + 1) as f64;
            let z = DVector::from_column_slice(&[100.0 + 10.0 * t, 200.0 + 5.0 * t, 50.0]);
            kf.update(&z, &h, &r);
        }

        // Should converge close to true state
        assert!((kf.x[1] - 10.0).abs() < 1.0); // velocity x
        assert!((kf.x[3] - 5.0).abs() < 1.0); // velocity y
    }

    // Task 1.3 (eval-consistency-metrics), spec "Linear KF diagnostics match
    // closed form": outcome against hand-computed values.
    #[test]
    fn kf_update_outcome_matches_closed_form() {
        // Hand-computed: x_pred = [1, 2], P_pred = [[4, 1], [1, 2]],
        // H = [1 0], R = [1], z = [3].
        //   innovation = z − H·x_pred = 3 − 1 = 2
        //   S = H·P_pred·Hᵀ + R = 4 + 1 = 5
        //   NIS = yᵀ S⁻¹ y = 2 · (1/5) · 2 = 0.8
        let mut kf = KalmanFilter::new(
            DVector::from_column_slice(&[1.0, 2.0]),
            DMatrix::from_row_slice(2, 2, &[4.0, 1.0, 1.0, 2.0]),
        );
        let h = DMatrix::from_row_slice(1, 2, &[1.0, 0.0]);
        let r = DMatrix::from_element(1, 1, 1.0);
        let z = DVector::from_element(1, 3.0);

        let outcome = kf.update(&z, &h, &r);
        assert_eq!(outcome.innovation.len(), 1);
        assert!((outcome.innovation[0] - 2.0).abs() < 1e-12);
        assert_eq!(outcome.innovation_covariance.shape(), (1, 1));
        assert!((outcome.innovation_covariance[(0, 0)] - 5.0).abs() < 1e-12);
        assert!((outcome.nis - 0.8).abs() < 1e-12);
    }

    // Task 1.3 / 1.5: the predicted state and covariance are read from the
    // pub x/p fields after predict and before update, and the outcome's
    // innovation/S are exactly the closed forms at that predicted pair
    // (spec "Linear KF diagnostics match closed form", predicted-state
    // clause).
    #[test]
    fn kf_predicted_state_readable_between_predict_and_update() {
        let model = ConstantVelocity::new(0.5);
        let mut kf = KalmanFilter::new(
            DVector::from_column_slice(&[1.0, 0.5, -2.0, 1.0, 3.0, -0.5]),
            DMatrix::identity(6, 6) * 100.0,
        );
        kf.predict_dt(&model, 1.0);

        // NEES/pre-update contract: the pub fields hold (x_pred, P_pred) here.
        let x_pred = kf.x.clone();
        let p_pred = kf.p.clone();

        let h = DMatrix::from_row_slice(
            3,
            6,
            &[
                1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                1.0, 0.0,
            ],
        );
        let r = DMatrix::identity(3, 3) * 2.0;
        let z = DVector::from_column_slice(&[1.5, -2.5, 3.5]);
        let outcome = kf.update(&z, &h, &r);

        let expected_y = &z - &h * &x_pred;
        let expected_s = &h * &p_pred * h.transpose() + &r;
        assert!((&outcome.innovation - expected_y).amax() < 1e-12);
        assert!((&outcome.innovation_covariance - expected_s).amax() < 1e-12);
    }

    // Task 1.3: NIS of a zero innovation is exactly 0 (not merely small).
    #[test]
    fn kf_zero_innovation_nis_is_exactly_zero() {
        let mut kf = KalmanFilter::new(
            DVector::from_column_slice(&[1.0, 2.0]),
            DMatrix::from_row_slice(2, 2, &[4.0, 1.0, 1.0, 2.0]),
        );
        let h = DMatrix::from_row_slice(1, 2, &[1.0, 0.0]);
        let r = DMatrix::from_element(1, 1, 1.0);
        // z = H·x_pred ⇒ innovation = 0 ⇒ NIS must be exactly 0.0.
        let z = &h * &kf.x;

        let outcome = kf.update(&z, &h, &r);
        assert!(outcome.innovation.iter().all(|&v| v == 0.0));
        assert_eq!(outcome.nis, 0.0);
    }

    #[test]
    fn kf_covariance_stays_psd() {
        let model = ConstantVelocity::new(1.0);
        let mut kf = KalmanFilter::new(DVector::zeros(6), DMatrix::identity(6, 6) * 100.0);

        let h = DMatrix::from_row_slice(
            3,
            6,
            &[
                1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                1.0, 0.0,
            ],
        );
        let r = DMatrix::identity(3, 3) * 10.0;

        for i in 0..1000 {
            kf.predict_dt(&model, 0.1);
            let z = DVector::from_column_slice(&[i as f64 * 0.1, 0.0, 0.0]);
            kf.update(&z, &h, &r);

            // Check all eigenvalues of P are non-negative
            let eigenvalues = kf.p.clone().symmetric_eigen().eigenvalues;
            for j in 0..eigenvalues.len() {
                assert!(
                    eigenvalues[j] >= -1e-10,
                    "Negative eigenvalue at step {i}: {}",
                    eigenvalues[j]
                );
            }
        }
    }
}
