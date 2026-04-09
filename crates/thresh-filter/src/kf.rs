//! Linear Kalman Filter with Joseph-form covariance update.

use nalgebra::{DMatrix, DVector};

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
    pub fn update(&mut self, z: &DVector<f64>, h: &DMatrix<f64>, r: &DMatrix<f64>) {
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

        // State update
        self.x = &self.x + &k * &y;

        // Joseph-form covariance update: P = (I - KH)P(I - KH)' + KRK'
        let n = self.x.len();
        let i_kh = DMatrix::identity(n, n) - &k * h;
        self.p = &i_kh * &self.p * i_kh.transpose() + &k * r * k.transpose();
    }

    /// Return the innovation (residual) for a measurement without updating.
    pub fn innovation(&self, z: &DVector<f64>, h: &DMatrix<f64>) -> DVector<f64> {
        z - h * &self.x
    }

    /// Return the innovation covariance S.
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
