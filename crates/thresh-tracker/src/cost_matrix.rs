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
}

/// Record an association hit and promote tentative/coasting → confirmed.
///
/// Free function so each tracker can pass `&mut` references to its hits,
/// misses, and lifecycle fields directly without needing trait boilerplate.
pub fn record_hit(
    hits: &mut usize,
    misses: &mut usize,
    lifecycle: &mut TrackState,
    confirm_hits: usize,
) {
    *hits += 1;
    *misses = 0;
    if (*lifecycle == TrackState::Tentative && *hits >= confirm_hits)
        || *lifecycle == TrackState::Coasting
    {
        *lifecycle = TrackState::Confirmed;
    }
}

/// Record a miss and update lifecycle (coast or delete).
pub fn record_miss(misses: &mut usize, lifecycle: &mut TrackState, max_misses: usize) {
    *misses += 1;
    if *misses >= max_misses {
        *lifecycle = TrackState::Deleted;
    } else if *lifecycle == TrackState::Confirmed {
        *lifecycle = TrackState::Coasting;
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

/// Parallel version of [`predict_all`] using rayon.
///
/// Falls back to the sequential path when fewer than 32 tracks are alive.
/// Requires the `parallel` feature.
#[cfg(feature = "parallel")]
pub fn predict_all_parallel<T: LinearTrack + Send>(
    tracks: &mut [T],
    f: &DMatrix<f64>,
    q: &DMatrix<f64>,
) {
    let alive_count = tracks.iter().filter(|t| t.is_alive()).count();
    if alive_count < 32 {
        predict_all(tracks, f, q);
        return;
    }
    use rayon::prelude::*;
    tracks.par_iter_mut().for_each(|track| {
        if !track.is_alive() {
            return;
        }
        let (s, c) = predict_linear(track.state(), track.covariance(), f, q);
        *track.state_mut() = s;
        *track.covariance_mut() = c;
    });
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

/// Parallel version of [`build_track_cost_matrix`] using rayon.
///
/// Each row (one track vs all detections) is computed independently.
/// Requires the `parallel` feature.
#[cfg(feature = "parallel")]
pub fn build_track_cost_matrix_parallel<T: LinearTrack + Sync>(
    tracks: &[T],
    alive: &[usize],
    h: &DMatrix<f64>,
    r: &DMatrix<f64>,
    detections: &[DVector<f64>],
    gate_threshold: f64,
) -> Vec<Vec<f64>> {
    use rayon::prelude::*;
    use thresh_association::gating::mahalanobis_squared as mah_sq;

    alive
        .par_iter()
        .map(|&ti| {
            let z_hat = h * tracks[ti].state();
            let s = h * tracks[ti].covariance() * h.transpose() + r;
            let mut row = vec![gate_threshold; detections.len()];
            for (dj, det) in detections.iter().enumerate() {
                let d2 = mah_sq(det, &z_hat, &s);
                if d2 < gate_threshold {
                    row[dj] = d2;
                }
            }
            row
        })
        .collect()
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

/// Run a single Kalman filter update on the given prior state and covariance.
///
/// This is a thin wrapper around [`thresh_filter::kf::KalmanFilter`] that
/// avoids per-call clone-and-discard boilerplate at tracker callsites.
pub fn kf_update(
    state: &DVector<f64>,
    covariance: &DMatrix<f64>,
    detection: &DVector<f64>,
    h: &DMatrix<f64>,
    r: &DMatrix<f64>,
) -> (DVector<f64>, DMatrix<f64>) {
    let mut kf = thresh_filter::kf::KalmanFilter::new(state.clone(), covariance.clone());
    kf.update(detection, h, r);
    (kf.x, kf.p)
}

/// Default initial covariance for a 6-state interleaved [pos, vel] track
/// (`[x, vx, y, vy, z/alt, vz/valt]`) born from a position-only detection.
///
/// Used by tracker variants that don't have measured velocity at birth time:
/// 10 km position std on the horizontal axes, 1 km on altitude, plus
/// generously wide velocity priors.
pub fn default_birth_covariance_6() -> DMatrix<f64> {
    DMatrix::from_diagonal(&DVector::from_column_slice(&[
        1.0e8, // x position (10 km std)
        1.0e4, // vx
        1.0e8, // y position
        1.0e4, // vy
        1.0e6, // z/alt (1 km std)
        1.0e2, // vz/valt
    ]))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn record_hit_promotes_tentative_after_threshold() {
        let mut hits = 0;
        let mut misses = 5;
        let mut lc = TrackState::Tentative;
        for _ in 0..3 {
            record_hit(&mut hits, &mut misses, &mut lc, 3);
        }
        assert_eq!(hits, 3);
        assert_eq!(misses, 0);
        assert_eq!(lc, TrackState::Confirmed);
    }

    #[test]
    fn record_hit_revives_coasting_track() {
        let mut hits = 5;
        let mut misses = 2;
        let mut lc = TrackState::Coasting;
        record_hit(&mut hits, &mut misses, &mut lc, 3);
        assert_eq!(lc, TrackState::Confirmed);
        assert_eq!(misses, 0);
    }

    #[test]
    fn record_miss_transitions_confirmed_to_coasting() {
        let mut misses = 0;
        let mut lc = TrackState::Confirmed;
        record_miss(&mut misses, &mut lc, 5);
        assert_eq!(misses, 1);
        assert_eq!(lc, TrackState::Coasting);
    }

    #[test]
    fn record_miss_deletes_after_max() {
        let mut misses = 4;
        let mut lc = TrackState::Coasting;
        record_miss(&mut misses, &mut lc, 5);
        assert_eq!(misses, 5);
        assert_eq!(lc, TrackState::Deleted);
    }
}
