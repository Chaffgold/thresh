//! Shared helpers for building tracker cost matrices.
//!
//! Extracted to deduplicate the predict + Mahalanobis cost-matrix pattern
//! used across the various tracker variants (Cartesian, ECEF, stereographic, …).

use nalgebra::{DMatrix, DVector};
use thresh_association::gating::mahalanobis_squared;
use thresh_core::track::TrackState;

/// Trait used by [`predict_all`] and [`build_track_cost_matrix`] so the same
/// helpers can drive multiple tracker variants whose track types only differ
/// in fields beyond the linear-Gaussian core.
pub trait LinearTrack {
    fn is_alive(&self) -> bool;
    fn state(&self) -> &DVector<f64>;
    fn state_mut(&mut self) -> &mut DVector<f64>;
    fn covariance(&self) -> &DMatrix<f64>;
    fn covariance_mut(&mut self) -> &mut DMatrix<f64>;

    // Lifecycle accessors so the shared lifecycle helpers can drive any
    // tracker variant. Default implementations panic so existing impls don't
    // have to provide them, but trackers using `record_hit_and_promote` /
    // `record_miss_and_age` must override these.
    fn hits(&self) -> usize {
        unimplemented!("LinearTrack::hits not implemented for this track type")
    }
    fn hits_mut(&mut self) -> &mut usize {
        unimplemented!("LinearTrack::hits_mut not implemented for this track type")
    }
    fn misses(&self) -> usize {
        unimplemented!("LinearTrack::misses not implemented for this track type")
    }
    fn misses_mut(&mut self) -> &mut usize {
        unimplemented!("LinearTrack::misses_mut not implemented for this track type")
    }
    fn lifecycle(&self) -> TrackState {
        unimplemented!("LinearTrack::lifecycle not implemented for this track type")
    }
    fn set_lifecycle(&mut self, _state: TrackState) {
        unimplemented!("LinearTrack::set_lifecycle not implemented for this track type")
    }
}

/// Record an association hit on a track and promote tentative/coasting
/// tracks to confirmed when appropriate.
pub fn record_hit_and_promote<T: LinearTrack>(track: &mut T, confirm_hits: usize) {
    *track.hits_mut() += 1;
    *track.misses_mut() = 0;
    let lc = track.lifecycle();
    if (lc == TrackState::Tentative && track.hits() >= confirm_hits) || lc == TrackState::Coasting {
        track.set_lifecycle(TrackState::Confirmed);
    }
}

/// Record a miss on a track and update lifecycle (coast or delete).
pub fn record_miss_and_age<T: LinearTrack>(track: &mut T, max_misses: usize) {
    *track.misses_mut() += 1;
    if track.misses() >= max_misses {
        track.set_lifecycle(TrackState::Deleted);
    } else if track.lifecycle() == TrackState::Confirmed {
        track.set_lifecycle(TrackState::Coasting);
    }
}

/// Apply a linear-Gaussian predict step to every alive track in `tracks`.
pub fn predict_all<T: LinearTrack>(tracks: &mut [T], f: &DMatrix<f64>, q: &DMatrix<f64>) {
    for track in tracks.iter_mut() {
        if !track.is_alive() {
            continue;
        }
        let (s, c) = predict_linear(track.state(), track.covariance(), f, q);
        *track.state_mut() = s;
        *track.covariance_mut() = c;
    }
}

/// Collect indices of alive tracks.
pub fn alive_indices<T: LinearTrack>(tracks: &[T]) -> Vec<usize> {
    tracks
        .iter()
        .enumerate()
        .filter(|(_, t)| t.is_alive())
        .map(|(i, _)| i)
        .collect()
}

/// Build a Mahalanobis cost matrix for a set of alive tracks given the
/// observation matrix and measurement noise that apply uniformly to all tracks.
pub fn build_track_cost_matrix<T: LinearTrack>(
    tracks: &[T],
    alive: &[usize],
    h: &DMatrix<f64>,
    r: &DMatrix<f64>,
    detections: &[DVector<f64>],
    gate_threshold: f64,
) -> Vec<Vec<f64>> {
    let predicted_obs: Vec<DVector<f64>> = alive.iter().map(|&ti| h * tracks[ti].state()).collect();
    let innovation_covs: Vec<DMatrix<f64>> = alive
        .iter()
        .map(|&ti| h * tracks[ti].covariance() * h.transpose() + r)
        .collect();
    build_cost_matrix(&predicted_obs, &innovation_covs, detections, gate_threshold)
}

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
