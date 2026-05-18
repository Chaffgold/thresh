//! Cubature Kalman Filter (third-order spherical-radial rule).
//!
//! The CKF uses `2n` equally-weighted cubature points drawn from the
//! third-order spherical-radial cubature rule. Compared with the UKF it is
//! parameter-free (no `alpha`/`beta`/`kappa`), uses `2n` points instead of
//! `2n + 1`, and the equal positive weights keep the predicted covariance
//! positive-definite under the standard rule.
//!
//! Reference: I. Arasaratnam and S. Haykin, "Cubature Kalman Filters",
//! IEEE Transactions on Automatic Control, vol. 54, no. 6, pp. 1254-1269,
//! June 2009.

use nalgebra::{DMatrix, DVector};

use crate::traits::MotionModel;

/// Cubature Kalman Filter state.
///
/// Parameter-free by design: third-order spherical-radial cubature is
/// uniquely determined by the state dimension, so there is no `CkfParams`
/// analogue to [`crate::ukf::UkfParams`].
pub struct CubatureKalmanFilter {
    /// Current state estimate.
    pub x: DVector<f64>,
    /// Current covariance estimate.
    pub p: DMatrix<f64>,
}

impl CubatureKalmanFilter {
    /// Create a new CKF with initial state and covariance.
    ///
    /// Matches [`crate::ekf::ExtendedKalmanFilter::new`] /
    /// [`crate::ukf::UnscentedKalmanFilter::new`]: state and covariance are
    /// stored as-is, with no dimension validation (callers are trusted to
    /// pass a consistent `(x, p)`, same convention as the sibling filters).
    pub fn new(x: DVector<f64>, p: DMatrix<f64>) -> Self {
        Self { x, p }
    }

    /// Generate the `2n` cubature points for the current `(x, p)` state.
    ///
    /// Third-order spherical-radial rule: `S = chol(P)`, then emit
    /// `x ± √n · S_{:,i}` for each column `i`. All points carry the implicit
    /// weight `1 / (2n)`, applied by the callers.
    fn cubature_points(&mut self) -> Vec<DVector<f64>> {
        crate::cov::ensure_psd(&mut self.p);
        let n = self.x.len();
        let sqrt_n = (n as f64).sqrt();

        let chol = self.p.clone().cholesky().unwrap_or_else(|| {
            // Fallback: regularize with a small identity. Mirrors
            // `ukf::UnscentedKalmanFilter::sigma_points` so both filters
            // degrade identically on a near-degenerate P that survives
            // `ensure_psd` but still trips the Cholesky pivot tolerance.
            let reg = &self.p + DMatrix::identity(n, n) * 1e-8;
            reg.cholesky()
                .expect("P is not recoverable for CKF cubature points")
        });
        let l = chol.l();

        let mut points = Vec::with_capacity(2 * n);
        for i in 0..n {
            let col = l.column(i).clone_owned() * sqrt_n;
            points.push(&self.x + &col);
            points.push(&self.x - &col);
        }
        points
    }

    /// Predict step using a nonlinear motion model.
    ///
    /// Propagates each cubature point through the model, recomputes the mean
    /// and covariance with equal `1 / (2n)` weights, and adds the model's
    /// process noise.
    pub fn predict(&mut self, model: &dyn MotionModel, dt: f64) {
        let points = self.cubature_points();
        let w = 1.0 / (points.len() as f64);
        let q = model.process_noise(dt);

        let propagated: Vec<DVector<f64>> = points.iter().map(|p| model.predict(p, dt)).collect();

        let n = self.x.len();
        let mut x_pred = DVector::zeros(n);
        for s in &propagated {
            x_pred += w * s;
        }

        let mut p_pred = DMatrix::zeros(n, n);
        for s in &propagated {
            let diff = s - &x_pred;
            p_pred += w * &diff * diff.transpose();
        }
        p_pred += q;

        self.x = x_pred;
        self.p = crate::cov::symmetrize(&p_pred);
    }

    /// Update step given measurement `z`, observation function `h_fn`, and
    /// measurement noise `r`.
    ///
    /// `h_fn` maps a state vector into measurement space.
    pub fn update<F>(&mut self, z: &DVector<f64>, h_fn: F, r: &DMatrix<f64>)
    where
        F: Fn(&DVector<f64>) -> DVector<f64>,
    {
        let points = self.cubature_points();
        let w = 1.0 / (points.len() as f64);

        let z_points: Vec<DVector<f64>> = points.iter().map(&h_fn).collect();

        let m = z_points[0].len();
        let mut z_pred = DVector::zeros(m);
        for zp in &z_points {
            z_pred += w * zp;
        }

        let n = self.x.len();
        let mut s_mat = DMatrix::zeros(m, m);
        let mut pxz = DMatrix::zeros(n, m);
        for (point, zp) in points.iter().zip(z_points.iter()) {
            let z_diff = zp - &z_pred;
            let x_diff = point - &self.x;
            s_mat += w * &z_diff * z_diff.transpose();
            pxz += w * &x_diff * z_diff.transpose();
        }
        s_mat += r;

        // K = Pxz * S^-1. S is symmetric, so K^T = S^-1 * Pxz^T =
        // solve(S, Pxz^T); an LU solve is better-conditioned than
        // forming S^-1 explicitly.
        let k = s_mat
            .clone()
            .lu()
            .solve(&pxz.transpose())
            .expect("CKF innovation covariance S is singular")
            .transpose();

        let innovation = z - &z_pred;
        self.x = &self.x + &k * &innovation;
        self.p = &self.p - &k * &s_mat * k.transpose();

        self.p = crate::cov::symmetrize(&self.p);
        crate::cov::ensure_psd(&mut self.p);
    }

    /// Linear update (convenience for sensors with a linear `H`).
    pub fn update_linear(&mut self, z: &DVector<f64>, h: &DMatrix<f64>, r: &DMatrix<f64>) {
        let h_clone = h.clone();
        self.update(z, move |x| &h_clone * x, r);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::cv::ConstantVelocity;
    use crate::ukf::{UkfParams, UnscentedKalmanFilter};

    /// Deterministic linear congruential generator for reproducible noise.
    struct Lcg(u64);
    impl Lcg {
        fn new(seed: u64) -> Self {
            Self(seed)
        }
        /// Uniform in [0, 1).
        fn next_unit(&mut self) -> f64 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((self.0 >> 11) as f64) / ((1u64 << 53) as f64)
        }
        /// Approximately standard-normal via sum of 12 uniforms (mean 0, var 1).
        fn next_gauss(&mut self) -> f64 {
            let s: f64 = (0..12).map(|_| self.next_unit()).sum();
            s - 6.0
        }
    }

    fn smallest_eigenvalue(m: &DMatrix<f64>) -> f64 {
        m.clone()
            .symmetric_eigen()
            .eigenvalues
            .iter()
            .copied()
            .fold(f64::INFINITY, f64::min)
    }

    /// Position-only observation matrix for the 6D `[x,vx,y,vy,z,vz]` state
    /// (observes `x`, `y`, `z`). Shared by the multi-step CKF tests.
    fn pos_obs_h() -> DMatrix<f64> {
        DMatrix::from_row_slice(
            3,
            6,
            &[
                1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                1.0, 0.0,
            ],
        )
    }

    // 2.1
    #[test]
    fn cubature_points_are_2n_with_equal_weights() {
        let x = DVector::from_column_slice(&[1.0, -2.0, 3.0, 0.5]);
        // Non-degenerate, non-diagonal SPD covariance.
        let a = DMatrix::from_row_slice(
            4,
            4,
            &[
                2.0, 0.3, 0.1, 0.0, 0.3, 1.5, 0.2, 0.1, 0.1, 0.2, 3.0, 0.4, 0.0, 0.1, 0.4, 1.0,
            ],
        );
        let p = &a * a.transpose();
        let mut ckf = CubatureKalmanFilter::new(x.clone(), p.clone());
        let points = ckf.cubature_points();

        assert_eq!(points.len(), 8, "expected 2n = 8 cubature points");

        let w = 1.0 / points.len() as f64;
        let mut mean = DVector::zeros(4);
        for pt in &points {
            mean += w * pt;
        }
        assert!((mean - &x).norm() < 1e-9, "weighted mean must equal x");

        let mut cov = DMatrix::zeros(4, 4);
        for pt in &points {
            let d = pt - &x;
            cov += w * &d * d.transpose();
        }
        assert!(
            (cov - &p).norm() < 1e-9,
            "weighted sample covariance must equal P"
        );
    }

    // 2.2
    #[test]
    fn predict_zero_noise_identity_motion_preserves_state() {
        struct Identity;
        impl MotionModel for Identity {
            fn state_dim(&self) -> usize {
                4
            }
            fn predict(&self, s: &DVector<f64>, _dt: f64) -> DVector<f64> {
                s.clone()
            }
            fn jacobian(&self, _s: &DVector<f64>, _dt: f64) -> DMatrix<f64> {
                DMatrix::identity(4, 4)
            }
            fn process_noise(&self, _dt: f64) -> DMatrix<f64> {
                DMatrix::zeros(4, 4)
            }
        }

        let x = DVector::from_column_slice(&[5.0, -1.0, 2.0, 7.0]);
        let p = DMatrix::from_diagonal(&DVector::from_column_slice(&[3.0, 1.0, 2.0, 4.0]));
        let mut ckf = CubatureKalmanFilter::new(x.clone(), p.clone());
        ckf.predict(&Identity, 1.0);

        assert!((&ckf.x - &x).norm() < 1e-9, "state drifted under identity");
        assert!(
            (&ckf.p - &p).norm() < 1e-9,
            "covariance drifted under identity"
        );
    }

    // 2.3
    #[test]
    fn update_linear_converges_to_truth() {
        let truth = DVector::from_column_slice(&[10.0, 1.0, -5.0, 0.5]);
        let x0 = DVector::from_column_slice(&[0.0, 0.0, 0.0, 0.0]);
        let p0 = DMatrix::identity(4, 4) * 100.0;
        let mut ckf = CubatureKalmanFilter::new(x0, p0);

        let h = DMatrix::identity(4, 4);
        let r = DMatrix::identity(4, 4) * 0.25;
        let mut rng = Lcg::new(42);

        let mut prev_trace = ckf.p.trace();
        for _ in 0..30 {
            let z = DVector::from_iterator(4, truth.iter().map(|&t| t + 0.5 * rng.next_gauss()));
            ckf.update_linear(&z, &h, &r);
            let trace = ckf.p.trace();
            assert!(
                trace <= prev_trace + 1e-9,
                "covariance trace must not increase under updates"
            );
            prev_trace = trace;
        }

        assert!(
            (&ckf.x - &truth).norm() < 0.5,
            "state should converge to truth, got {:?}",
            ckf.x
        );
    }

    // 2.4
    #[test]
    fn update_nonlinear_bearings_only_reduces_uncertainty() {
        // State [px, py]; bearings-only measurements from several sensors.
        let truth: DVector<f64> = DVector::from_column_slice(&[40.0, 30.0]);
        let x0 = DVector::from_column_slice(&[0.0, 0.0]);
        let p0 = DMatrix::identity(2, 2) * 500.0;
        let mut ckf = CubatureKalmanFilter::new(x0, p0);

        let sensors = [
            DVector::from_column_slice(&[0.0, 0.0]),
            DVector::from_column_slice(&[100.0, 0.0]),
            DVector::from_column_slice(&[0.0, 100.0]),
            DVector::from_column_slice(&[-80.0, 60.0]),
        ];
        let r = DMatrix::from_element(1, 1, 1e-4);

        let initial_pos_var = ckf.p[(0, 0)] + ckf.p[(1, 1)];
        for _ in 0..6 {
            for sensor in &sensors {
                let s = sensor.clone();
                let bearing = (truth[1] - s[1]).atan2(truth[0] - s[0]);
                let z = DVector::from_element(1, bearing);
                ckf.update(
                    &z,
                    move |x| DVector::from_element(1, (x[1] - s[1]).atan2(x[0] - s[0])),
                    &r,
                );
            }
        }
        let final_pos_var = ckf.p[(0, 0)] + ckf.p[(1, 1)];

        // Spec scenario 2.4: the position covariance must shrink over the
        // bearings-only measurement sequence.
        assert!(
            final_pos_var < initial_pos_var,
            "position covariance should shrink: {initial_pos_var} -> {final_pos_var}"
        );
        // And the estimate must move meaningfully toward truth relative to the
        // (origin) prior — not a tight convergence bound, since bearings-only
        // observability depends on sensor geometry.
        let prior_err = truth.norm();
        let post_err = (&ckf.x - &truth).norm();
        assert!(
            ckf.x.iter().all(|v| v.is_finite()) && post_err < prior_err,
            "estimate should improve on the prior: prior_err={prior_err}, post_err={post_err}"
        );
    }

    // 2.5
    #[test]
    fn covariance_stays_symmetric_pd_under_perturbation() {
        let mut ckf = CubatureKalmanFilter::new(
            DVector::from_column_slice(&[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            DMatrix::identity(6, 6) * 10.0,
        );
        let model = ConstantVelocity::new(1.0);
        let h = pos_obs_h();
        let r = DMatrix::identity(3, 3) * 4.0;
        let mut rng = Lcg::new(7);

        for _ in 0..100 {
            ckf.predict(&model, 0.5);
            let asym = (&ckf.p - ckf.p.transpose()).norm();
            assert!(asym < 1e-9, "asymmetry after predict: {asym}");
            assert!(
                smallest_eigenvalue(&ckf.p) > 0.0,
                "P lost positive-definiteness after predict"
            );

            let z = DVector::from_column_slice(&[
                10.0 * rng.next_gauss(),
                10.0 * rng.next_gauss(),
                10.0 * rng.next_gauss(),
            ]);
            ckf.update_linear(&z, &h, &r);
            let asym = (&ckf.p - ckf.p.transpose()).norm();
            assert!(asym < 1e-9, "asymmetry after update: {asym}");
            assert!(
                smallest_eigenvalue(&ckf.p) > 0.0,
                "P lost positive-definiteness after update"
            );
        }
    }

    // 2.6
    #[test]
    fn ckf_and_ukf_agree_on_linear_problem() {
        let x0 = DVector::from_column_slice(&[0.0, 5.0, 0.0, -3.0, 0.0, 1.0]);
        let p0 = DMatrix::identity(6, 6) * 50.0;
        let mut ckf = CubatureKalmanFilter::new(x0.clone(), p0.clone());
        let mut ukf = UnscentedKalmanFilter::new(x0, p0, UkfParams::default());

        let model = ConstantVelocity::new(1.0);
        let h = pos_obs_h();
        let r = DMatrix::identity(3, 3) * 9.0;
        let mut rng = Lcg::new(2024);

        for step in 0..15 {
            let t = (step + 1) as f64;
            let z = DVector::from_column_slice(&[
                5.0 * t + rng.next_gauss(),
                -3.0 * t + rng.next_gauss(),
                1.0 * t + rng.next_gauss(),
            ]);
            ckf.predict(&model, 1.0);
            ukf.predict(&model, 1.0);
            ckf.update_linear(&z, &h, &r);
            ukf.update_linear(&z, &h, &r);
        }

        assert!(
            (&ckf.x - &ukf.x).amax() < 1e-6,
            "posterior means diverge: ckf={:?} ukf={:?}",
            ckf.x,
            ukf.x
        );
        assert!(
            (&ckf.p - &ukf.p).amax() < 1e-4,
            "posterior covariances diverge by {}",
            (&ckf.p - &ukf.p).amax()
        );
    }
}
