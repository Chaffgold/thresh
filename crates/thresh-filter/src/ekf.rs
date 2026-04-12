//! Extended Kalman Filter with Jacobian-based linearization.

use nalgebra::{DMatrix, DVector};

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
    pub fn update(
        &mut self,
        z: &DVector<f64>,
        h_of_x: &DVector<f64>,
        h_jac: &DMatrix<f64>,
        r: &DMatrix<f64>,
    ) {
        // Innovation
        let y = z - h_of_x;
        // Innovation covariance
        let s = h_jac * &self.p * h_jac.transpose() + r;

        let s_inv = s
            .clone()
            .try_inverse()
            .expect("Innovation covariance S is singular");
        let k = &self.p * h_jac.transpose() * &s_inv;

        // State update
        self.x = &self.x + &k * &y;

        // Joseph-form covariance
        let n = self.x.len();
        let i_kh = DMatrix::identity(n, n) - &k * h_jac;
        self.p = &i_kh * &self.p * i_kh.transpose() + &k * r * k.transpose();
    }

    /// Linear update (for sensors with linear observation models).
    pub fn update_linear(&mut self, z: &DVector<f64>, h: &DMatrix<f64>, r: &DMatrix<f64>) {
        let h_of_x = h * &self.x;
        self.update(z, &h_of_x, h, r);
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
}
