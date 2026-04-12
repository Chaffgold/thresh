//! Track-to-track fusion: distributed/federated fusion of track-level outputs.
//!
//! Enables multiple tracker instances to merge track estimates without sharing
//! raw measurements. Provides:
//! - [`TrackExchange`] — lightweight track state for inter-site exchange
//! - [`t2t_association`] — augmented-state Mahalanobis + Hungarian matching
//! - [`fuse_naive`] — inverse-covariance-weighted average (assumes independence)
//! - [`fuse_covariance_intersection`] — CI fusion (safe when cross-covariances unknown)
//! - [`FederatedFusionManager`] — multi-site fusion orchestrator

use nalgebra::{DMatrix, DVector};
use thresh_association::hungarian::{AssignmentResult, hungarian_assignment};

// ---------------------------------------------------------------------------
// Core type
// ---------------------------------------------------------------------------

/// Lightweight track state exchanged between fusion sites.
#[derive(Debug, Clone)]
pub struct TrackExchange {
    /// Unique track identifier at the originating site.
    pub track_id: u64,
    /// State vector (e.g. [x, vx, y, vy, …]).
    pub state: DVector<f64>,
    /// Error covariance matrix.
    pub covariance: DMatrix<f64>,
    /// Timestamp of the estimate (seconds, epoch-relative).
    pub timestamp: f64,
    /// Identifier of the originating tracker / site.
    pub source_id: u32,
}

// ---------------------------------------------------------------------------
// T2T association
// ---------------------------------------------------------------------------

/// Augmented-state Mahalanobis distance between two track estimates.
///
/// `d² = (x_a - x_b)ᵀ (P_a + P_b)⁻¹ (x_a - x_b)`
fn augmented_mahalanobis(a: &TrackExchange, b: &TrackExchange) -> f64 {
    let dx = &a.state - &b.state;
    let s = &a.covariance + &b.covariance;
    match s.try_inverse() {
        Some(s_inv) => {
            let d2 = (&dx.transpose() * s_inv * &dx)[(0, 0)];
            d2.max(0.0) // numerical floor
        }
        None => f64::INFINITY,
    }
}

/// Associate tracks from two sites using augmented-state Mahalanobis distance
/// and Hungarian optimal assignment.
///
/// Returns `(row, col)` index pairs into `tracks_a` / `tracks_b` for matched
/// tracks. Pairs with Mahalanobis distance ≥ `gate` are rejected.
pub fn t2t_association(
    tracks_a: &[TrackExchange],
    tracks_b: &[TrackExchange],
    gate: f64,
) -> Vec<(usize, usize)> {
    if tracks_a.is_empty() || tracks_b.is_empty() {
        return Vec::new();
    }

    // Build cost matrix of Mahalanobis distances.
    let cost: Vec<Vec<f64>> = tracks_a
        .iter()
        .map(|a| {
            tracks_b
                .iter()
                .map(|b| augmented_mahalanobis(a, b))
                .collect()
        })
        .collect();

    let AssignmentResult { matches, .. } = hungarian_assignment(&cost, gate);
    matches
}

// ---------------------------------------------------------------------------
// Fusion algorithms
// ---------------------------------------------------------------------------

/// Covariance Intersection fusion of two track estimates.
///
/// Optimises ω ∈ [0, 1] to minimise trace(P_fused). Safe when
/// cross-covariances between the two estimates are unknown.
pub fn fuse_covariance_intersection(a: &TrackExchange, b: &TrackExchange) -> TrackExchange {
    let (x_fused, p_fused) = crate::covariance_intersection::covariance_intersection(
        &a.state,
        &a.covariance,
        &b.state,
        &b.covariance,
    );

    TrackExchange {
        track_id: a.track_id, // keep the first site's ID by convention
        state: x_fused,
        covariance: p_fused,
        timestamp: a.timestamp.max(b.timestamp),
        source_id: 0, // fused — no single source
    }
}

/// Naive inverse-covariance-weighted fusion.
///
/// Assumes the two estimation errors are independent:
/// ```text
/// P_f⁻¹ = P_a⁻¹ + P_b⁻¹
/// x_f   = P_f (P_a⁻¹ x_a + P_b⁻¹ x_b)
/// ```
///
/// Fast but **overconfident** when the estimates share common process noise.
pub fn fuse_naive(a: &TrackExchange, b: &TrackExchange) -> TrackExchange {
    let p_a_inv = a
        .covariance
        .clone()
        .try_inverse()
        .expect("P_a singular in naive fusion");
    let p_b_inv = b
        .covariance
        .clone()
        .try_inverse()
        .expect("P_b singular in naive fusion");

    let p_fused_inv = &p_a_inv + &p_b_inv;
    let p_fused = p_fused_inv
        .try_inverse()
        .expect("fused P singular in naive fusion");
    let x_fused = &p_fused * (&p_a_inv * &a.state + &p_b_inv * &b.state);

    TrackExchange {
        track_id: a.track_id,
        state: x_fused,
        covariance: p_fused,
        timestamp: a.timestamp.max(b.timestamp),
        source_id: 0,
    }
}

// ---------------------------------------------------------------------------
// Federated fusion manager
// ---------------------------------------------------------------------------

/// Multi-site federated fusion manager.
///
/// Collects track-level outputs from N sites and produces a fused common
/// operating picture using Covariance Intersection (safe default).
pub struct FederatedFusionManager {
    /// Per-site track submissions, keyed by source_id.
    site_tracks: Vec<(u32, Vec<TrackExchange>)>,
}

impl FederatedFusionManager {
    /// Create a new empty manager.
    pub fn new() -> Self {
        Self {
            site_tracks: Vec::new(),
        }
    }

    /// Submit tracks from a single site. Replaces any previous submission from
    /// the same `source_id`.
    pub fn submit_tracks(&mut self, source_id: u32, tracks: Vec<TrackExchange>) {
        // Replace if source already submitted
        if let Some(entry) = self.site_tracks.iter_mut().find(|(id, _)| *id == source_id) {
            entry.1 = tracks;
        } else {
            self.site_tracks.push((source_id, tracks));
        }
    }

    /// Fuse all submitted tracks and return the unified track list.
    ///
    /// Workflow:
    /// 1. Start with the first site's tracks as the initial fused set.
    /// 2. For each subsequent site, associate its tracks against the current
    ///    fused set using [`t2t_association`].
    /// 3. Fuse matched pairs via Covariance Intersection.
    /// 4. Append unmatched tracks from the new site (track birth).
    /// 5. Pass through unmatched fused tracks unchanged (coasting).
    pub fn fuse(&self, gate: f64) -> Vec<TrackExchange> {
        if self.site_tracks.is_empty() {
            return Vec::new();
        }

        // Seed with first site
        let mut fused: Vec<TrackExchange> = self.site_tracks[0].1.clone();
        // Mark fused source_id = 0
        for t in &mut fused {
            t.source_id = 0;
        }

        // Sequentially incorporate each additional site
        for (_source_id, incoming) in self.site_tracks.iter().skip(1) {
            let matches = t2t_association(&fused, incoming, gate);

            let mut matched_fused: Vec<bool> = vec![false; fused.len()];
            let mut matched_incoming: Vec<bool> = vec![false; incoming.len()];
            let mut new_fused = Vec::new();

            // Fuse matched pairs
            for &(fi, ii) in &matches {
                matched_fused[fi] = true;
                matched_incoming[ii] = true;
                new_fused.push(fuse_covariance_intersection(&fused[fi], &incoming[ii]));
            }

            // Keep unmatched fused tracks (coasting)
            for (i, track) in fused.iter().enumerate() {
                if !matched_fused[i] {
                    new_fused.push(track.clone());
                }
            }

            // Birth unmatched incoming tracks
            for (i, track) in incoming.iter().enumerate() {
                if !matched_incoming[i] {
                    let mut t = track.clone();
                    t.source_id = 0; // now part of fused picture
                    new_fused.push(t);
                }
            }

            fused = new_fused;
        }

        fused
    }
}

impl Default for FederatedFusionManager {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{DMatrix, DVector};

    /// Helper: create a `TrackExchange` with diagonal covariance.
    fn make_track(track_id: u64, source_id: u32, state: &[f64], cov_diag: &[f64]) -> TrackExchange {
        TrackExchange {
            track_id,
            source_id,
            state: DVector::from_column_slice(state),
            covariance: DMatrix::from_diagonal(&DVector::from_column_slice(cov_diag)),
            timestamp: 0.0,
        }
    }

    // -- association tests ---------------------------------------------------

    #[test]
    fn test_t2t_association_matches_identical_tracks() {
        let site_a = vec![
            make_track(1, 1, &[10.0, 20.0], &[1.0, 1.0]),
            make_track(2, 1, &[30.0, 40.0], &[1.0, 1.0]),
        ];
        let site_b = vec![
            make_track(10, 2, &[10.0, 20.0], &[1.0, 1.0]),
            make_track(20, 2, &[30.0, 40.0], &[1.0, 1.0]),
        ];

        let matches = t2t_association(&site_a, &site_b, 20.0);
        assert_eq!(matches.len(), 2, "identical tracks should all match");

        // Verify correct pairing (0↔0, 1↔1)
        let mut sorted = matches.clone();
        sorted.sort();
        assert_eq!(sorted, vec![(0, 0), (1, 1)]);
    }

    #[test]
    fn test_t2t_association_rejects_distant_tracks() {
        let site_a = vec![make_track(1, 1, &[0.0, 0.0], &[1.0, 1.0])];
        let site_b = vec![make_track(10, 2, &[1000.0, 1000.0], &[1.0, 1.0])];

        // Tight gate — Mahalanobis distance will far exceed this
        let matches = t2t_association(&site_a, &site_b, 5.0);
        assert!(matches.is_empty(), "distant tracks should not match");
    }

    // -- naive fusion tests --------------------------------------------------

    #[test]
    fn test_fuse_naive_reduces_covariance() {
        let a = make_track(1, 1, &[10.0, 20.0], &[4.0, 9.0]);
        let b = make_track(2, 2, &[11.0, 19.0], &[4.0, 9.0]);

        let fused = fuse_naive(&a, &b);

        let det_a = a.covariance.determinant();
        let det_b = b.covariance.determinant();
        let det_f = fused.covariance.determinant();

        assert!(
            det_f < det_a,
            "fused det {det_f} should be < input A det {det_a}"
        );
        assert!(
            det_f < det_b,
            "fused det {det_f} should be < input B det {det_b}"
        );
    }

    // -- CI fusion tests -----------------------------------------------------

    #[test]
    fn test_fuse_ci_reduces_covariance() {
        let a = make_track(1, 1, &[10.0, 20.0], &[5.0, 10.0]);
        let b = make_track(2, 2, &[12.0, 18.0], &[8.0, 3.0]);

        let fused = fuse_covariance_intersection(&a, &b);

        let tr_a = a.covariance.trace();
        let tr_b = b.covariance.trace();
        let tr_f = fused.covariance.trace();
        let tr_min = tr_a.min(tr_b);

        assert!(
            tr_f <= tr_min + 1e-6,
            "CI fused trace {tr_f} should be <= min input trace {tr_min}"
        );
    }

    // -- federated manager tests ---------------------------------------------

    #[test]
    fn test_federated_manager_two_sites() {
        // 2 common targets + 1 unique to site B
        let mut mgr = FederatedFusionManager::new();

        mgr.submit_tracks(
            1,
            vec![
                make_track(1, 1, &[10.0, 20.0], &[2.0, 2.0]),
                make_track(2, 1, &[50.0, 60.0], &[2.0, 2.0]),
            ],
        );
        mgr.submit_tracks(
            2,
            vec![
                make_track(10, 2, &[10.1, 19.9], &[3.0, 3.0]), // ≈ track 1
                make_track(20, 2, &[50.2, 59.8], &[3.0, 3.0]), // ≈ track 2
                make_track(30, 2, &[200.0, 300.0], &[2.0, 2.0]), // unique
            ],
        );

        let fused = mgr.fuse(20.0);

        // Expect: 2 fused + 1 passed-through = 3 total
        assert_eq!(
            fused.len(),
            3,
            "expected 3 fused tracks, got {}",
            fused.len()
        );
    }

    #[test]
    fn test_federated_manager_no_overlap() {
        let mut mgr = FederatedFusionManager::new();

        mgr.submit_tracks(1, vec![make_track(1, 1, &[0.0, 0.0], &[1.0, 1.0])]);
        mgr.submit_tracks(2, vec![make_track(10, 2, &[1000.0, 1000.0], &[1.0, 1.0])]);

        let fused = mgr.fuse(5.0); // tight gate

        // Both tracks should pass through unfused
        assert_eq!(
            fused.len(),
            2,
            "expected 2 unfused tracks, got {}",
            fused.len()
        );
    }
}
