//! Kalman filter family (KF, EKF, UKF, CKF) with configurable motion models.

pub mod ckf;
pub mod cov;
pub mod ekf;
pub mod imm;
#[cfg(feature = "learned-imm")]
pub mod imm_adapter;
pub mod kf;
pub mod models;
pub mod numeric;
pub mod traits;
pub mod ukf;

use nalgebra::{DMatrix, DVector};

/// Diagnostics from a completed measurement update.
///
/// Returned by value from every `update` / `update_linear` on
/// [`kf::KalmanFilter`], [`ekf::ExtendedKalmanFilter`],
/// [`ukf::UnscentedKalmanFilter`], and [`ckf::CubatureKalmanFilter`] (and
/// through the [`imm::LeafFilter`] trait). Nothing is stored on the filters:
/// diagnostics exist **only** as each `update` call's return value, so no
/// stale or pre-first-update query surface exists — before the first update
/// there is simply no outcome to read.
///
/// NEES contract: the predicted (pre-update) state and covariance are read
/// from the filters' `pub` `x` / `p` fields after `predict` and before the
/// subsequent `update`; the posterior pair is read from the same fields after
/// `update` returns. No accessors alias those fields.
#[derive(Debug, Clone)]
pub struct UpdateOutcome {
    /// Innovation `y = z − ẑ` actually applied (pre-fusion residual).
    pub innovation: DVector<f64>,
    /// Innovation covariance `S` actually used to form the gain
    /// (`H P Hᵀ + R` for KF/EKF; the sigma-/cubature-point moment for
    /// UKF/CKF).
    pub innovation_covariance: DMatrix<f64>,
    /// Normalized innovation squared `yᵀ S⁻¹ y`, computed inside the update
    /// where `S`'s factorization already exists.
    pub nis: f64,
}
