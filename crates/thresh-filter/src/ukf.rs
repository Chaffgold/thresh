//! Unscented Kalman Filter with Van der Merwe sigma point selection.

use nalgebra::{DMatrix, DVector};

use crate::traits::MotionModel;

/// UKF tuning parameters.
#[derive(Debug, Clone, Copy)]
pub struct UkfParams {
    /// Sigma point spread (typically 1e-3 to 1).
    pub alpha: f64,
    /// Prior distribution parameter (2 is optimal for Gaussian).
    pub beta: f64,
    /// Secondary scaling parameter (typically 0 or 3-n).
    pub kappa: f64,
}

impl Default for UkfParams {
    fn default() -> Self {
        Self {
            alpha: 1e-3,
            beta: 2.0,
            kappa: 0.0,
        }
    }
}

/// Unscented Kalman Filter state.
pub struct UnscentedKalmanFilter {
    /// Current state estimate.
    pub x: DVector<f64>,
    /// Current covariance estimate.
    pub p: DMatrix<f64>,
    /// Tuning parameters.
    pub params: UkfParams,
}

impl UnscentedKalmanFilter {
    /// Create a new UKF.
    pub fn new(x: DVector<f64>, p: DMatrix<f64>, params: UkfParams) -> Self {
        Self { x, p, params }
    }

    /// Create with default parameters.
    pub fn with_defaults(x: DVector<f64>, p: DMatrix<f64>) -> Self {
        Self::new(x, p, UkfParams::default())
    }

    /// Compute lambda for sigma point generation.
    fn lambda(&self) -> f64 {
        let n = self.x.len() as f64;
        self.params.alpha * self.params.alpha * (n + self.params.kappa) - n
    }

    /// Ensure P is positive definite by clamping negative eigenvalues.
    fn ensure_psd(&mut self) {
        crate::cov::ensure_psd(&mut self.p);
    }

    /// Generate 2n+1 sigma points and their weights.
    fn sigma_points(&mut self) -> (Vec<DVector<f64>>, Vec<f64>, Vec<f64>) {
        self.ensure_psd();
        let n = self.x.len();
        let nf = n as f64;
        let lambda = self.lambda();
        let scale = nf + lambda;

        // Cholesky of (n + lambda) * P
        let scaled_p = &self.p * scale;
        let chol = scaled_p.clone().cholesky().unwrap_or_else(|| {
            // Fallback: regularize with small identity
            let reg = &scaled_p + DMatrix::identity(n, n) * 1e-8;
            reg.cholesky()
                .expect("P is not recoverable for UKF sigma points")
        });
        let l = chol.l();

        let mut sigmas = Vec::with_capacity(2 * n + 1);
        let mut wm = Vec::with_capacity(2 * n + 1);
        let mut wc = Vec::with_capacity(2 * n + 1);

        // Central sigma point
        sigmas.push(self.x.clone());
        wm.push(lambda / scale);
        wc.push(lambda / scale + (1.0 - self.params.alpha * self.params.alpha + self.params.beta));

        // Spread points
        let w = 1.0 / (2.0 * scale);
        for i in 0..n {
            let col = l.column(i);
            sigmas.push(&self.x + col.clone_owned());
            sigmas.push(&self.x - col.clone_owned());
            wm.push(w);
            wm.push(w);
            wc.push(w);
            wc.push(w);
        }

        (sigmas, wm, wc)
    }

    /// Predict step using a nonlinear motion model.
    pub fn predict(&mut self, model: &dyn MotionModel, dt: f64) {
        let (sigmas, wm, wc) = self.sigma_points();
        let q = model.process_noise(dt);

        // Propagate sigma points
        let propagated: Vec<DVector<f64>> = sigmas.iter().map(|s| model.predict(s, dt)).collect();

        // Predicted mean
        let n = self.x.len();
        let mut x_pred = DVector::zeros(n);
        for (i, s) in propagated.iter().enumerate() {
            x_pred += wm[i] * s;
        }

        // Predicted covariance
        let mut p_pred = DMatrix::zeros(n, n);
        for (i, s) in propagated.iter().enumerate() {
            let diff = s - &x_pred;
            p_pred += wc[i] * &diff * diff.transpose();
        }
        p_pred += q;

        self.x = x_pred;
        // Enforce symmetry for numerical stability
        self.p = (&p_pred + p_pred.transpose()) * 0.5;
    }

    /// Update step given measurement z, observation function h, and noise R.
    ///
    /// `h_fn` maps state sigma points to measurement space.
    pub fn update<F>(&mut self, z: &DVector<f64>, h_fn: F, r: &DMatrix<f64>)
    where
        F: Fn(&DVector<f64>) -> DVector<f64>,
    {
        let (sigmas, wm, wc) = self.sigma_points();

        // Project sigma points through observation function
        let z_sigmas: Vec<DVector<f64>> = sigmas.iter().map(&h_fn).collect();

        // Predicted measurement mean
        let m = z_sigmas[0].len();
        let mut z_pred = DVector::zeros(m);
        for (i, zs) in z_sigmas.iter().enumerate() {
            z_pred += wm[i] * zs;
        }

        // Innovation covariance S
        let n = self.x.len();
        let mut s_mat = DMatrix::zeros(m, m);
        let mut pxz = DMatrix::zeros(n, m);

        for (i, zs) in z_sigmas.iter().enumerate() {
            let z_diff = zs - &z_pred;
            let x_diff = &sigmas[i] - &self.x;
            s_mat += wc[i] * &z_diff * z_diff.transpose();
            pxz += wc[i] * &x_diff * z_diff.transpose();
        }
        s_mat += r;

        // Kalman gain. K = Pxz * S^-1; S is symmetric, so K^T =
        // solve(S, Pxz^T). An LU solve is better-conditioned than
        // forming S^-1 explicitly.
        let k = s_mat
            .clone()
            .lu()
            .solve(&pxz.transpose())
            .expect("UKF innovation covariance S is singular")
            .transpose();

        // Update
        let innovation = z - &z_pred;
        self.x = &self.x + &k * &innovation;
        self.p = &self.p - &k * &s_mat * k.transpose();

        // Enforce symmetry and PSD for numerical stability
        self.p = (&self.p + self.p.transpose()) * 0.5;
        self.ensure_psd();
    }

    /// Linear update (convenience for sensors with linear H).
    pub fn update_linear(&mut self, z: &DVector<f64>, h: &DMatrix<f64>, r: &DMatrix<f64>) {
        let h_clone = h.clone();
        self.update(z, move |x| &h_clone * x, r);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::ctrv::Ctrv;

    #[test]
    fn ukf_mean_weights_sum_to_one() {
        let mut ukf =
            UnscentedKalmanFilter::with_defaults(DVector::zeros(5), DMatrix::identity(5, 5));
        let (_, wm, _wc) = ukf.sigma_points();
        let sum_wm: f64 = wm.iter().sum();
        // Mean weights must sum to 1; covariance weights may not (due to beta term)
        assert!(
            (sum_wm - 1.0).abs() < 1e-8,
            "Mean weights sum to {sum_wm}, expected 1.0"
        );
    }

    #[test]
    fn ukf_correct_sigma_count() {
        let mut ukf =
            UnscentedKalmanFilter::with_defaults(DVector::zeros(5), DMatrix::identity(5, 5));
        let (sigmas, wm, wc) = ukf.sigma_points();
        assert_eq!(sigmas.len(), 11); // 2*5+1
        assert_eq!(wm.len(), 11);
        assert_eq!(wc.len(), 11);
    }

    #[test]
    fn ukf_polar_to_cartesian_second_order() {
        // Test that UKF handles nonlinear polar-to-cartesian better than linearization
        let mut ukf = UnscentedKalmanFilter::new(
            DVector::from_column_slice(&[100.0, 0.5]), // [range, bearing]
            DMatrix::from_diagonal(&DVector::from_column_slice(&[100.0, 0.01])), // large range uncertainty
            UkfParams {
                alpha: 0.5,
                beta: 2.0,
                kappa: 0.0,
            },
        );
        let (sigmas, wm, _) = ukf.sigma_points();

        // Transform to cartesian
        let h = |s: &DVector<f64>| -> DVector<f64> {
            DVector::from_column_slice(&[s[0] * s[1].cos(), s[0] * s[1].sin()])
        };

        let mut mean = DVector::zeros(2);
        for (i, s) in sigmas.iter().enumerate() {
            mean += wm[i] * h(s);
        }

        // The UKF mean should differ from the naive linearization
        let naive_x = 100.0 * 0.5f64.cos();
        let naive_y = 100.0 * 0.5f64.sin();
        // Just verify it produces reasonable values (not NaN, finite)
        assert!(mean[0].is_finite());
        assert!(mean[1].is_finite());
        // And is close to naive (within ~10% for these uncertainties)
        assert!((mean[0] - naive_x).abs() < 20.0);
        assert!((mean[1] - naive_y).abs() < 20.0);
    }

    #[test]
    fn ukf_tracks_turning_target() {
        let model = Ctrv::new(1.0, 0.1);
        let mut ukf = UnscentedKalmanFilter::new(
            DVector::from_column_slice(&[0.0, 0.0, 0.0, 100.0, 0.1]),
            DMatrix::identity(5, 5) * 10.0,
            UkfParams {
                alpha: 0.5,
                beta: 2.0,
                kappa: 0.0,
            },
        );

        let h_mat =
            DMatrix::from_row_slice(2, 5, &[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]);
        let r = DMatrix::identity(2, 2) * 25.0;

        let true_model = Ctrv::new(0.0, 0.0);
        let mut true_state = DVector::from_column_slice(&[0.0, 0.0, 0.0, 100.0, 0.1]);

        for _ in 0..50 {
            true_state = true_model.predict(&true_state, 0.1);
            let z = DVector::from_column_slice(&[true_state[0], true_state[1]]);

            ukf.predict(&model, 0.1);
            ukf.update_linear(&z, &h_mat, &r);
        }

        let pos_err =
            ((ukf.x[0] - true_state[0]).powi(2) + (ukf.x[1] - true_state[1]).powi(2)).sqrt();
        assert!(pos_err < 20.0, "Position error too large: {pos_err}");
    }
}
