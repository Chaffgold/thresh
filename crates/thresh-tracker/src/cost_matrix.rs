//! Shared helpers for building tracker cost matrices.
//!
//! Extracted to deduplicate the predict + Mahalanobis cost-matrix pattern
//! used across the various tracker variants (Cartesian, ECEF, stereographic, …).

use nalgebra::{DMatrix, DVector};
use thresh_association::gating::mahalanobis_squared;

/// Build a Mahalanobis-distance cost matrix between predicted track
/// observations and raw detection vectors.
///
/// `predicted_obs` is the per-track predicted observation `H x_hat`.
/// `innovation_cov` is the per-track innovation covariance `H P H^T + R`.
/// Entries above `gate_threshold` are clamped to `gate_threshold` so the
/// caller can use a single value to mean "ungated".
pub fn build_cost_matrix(
    predicted_obs: &[DVector<f64>],
    innovation_cov: &[DMatrix<f64>],
    detections: &[DVector<f64>],
    gate_threshold: f64,
) -> Vec<Vec<f64>> {
    let mut cost = vec![vec![gate_threshold; detections.len()]; predicted_obs.len()];
    for (ai, (z_hat, s)) in predicted_obs.iter().zip(innovation_cov.iter()).enumerate() {
        for (dj, det) in detections.iter().enumerate() {
            let d2 = mahalanobis_squared(det, z_hat, s);
            if d2 < gate_threshold {
                cost[ai][dj] = d2;
            }
        }
    }
    cost
}

/// Apply a linear-Gaussian predict step to a state vector and covariance.
///
/// Returns the predicted state and covariance: `(F x, F P F^T + Q)`.
pub fn predict_linear(
    state: &DVector<f64>,
    covariance: &DMatrix<f64>,
    f: &DMatrix<f64>,
    q: &DMatrix<f64>,
) -> (DVector<f64>, DMatrix<f64>) {
    let new_state = f * state;
    let new_cov = f * covariance * f.transpose() + q;
    (new_state, new_cov)
}
