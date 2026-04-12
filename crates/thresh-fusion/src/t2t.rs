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
// Fusion mode
// ---------------------------------------------------------------------------

/// Selects which fusion algorithm the [`FederatedFusionManager`] uses.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FusionMode {
    /// Naive inverse-covariance-weighted fusion (assumes independence).
    Naive,
    /// Covariance Intersection (safe when cross-covariances are unknown).
    CovarianceIntersection,
}

// ---------------------------------------------------------------------------
// Temporal alignment
// ---------------------------------------------------------------------------

/// Extrapolate a track exchange to `target_time` using a linear state
/// transition matrix `f` and process noise `q`.
///
/// If `dt` ≈ 0 the original exchange is returned unchanged.
pub fn extrapolate_track(
    exchange: &TrackExchange,
    target_time: f64,
    f: &DMatrix<f64>,
    q: &DMatrix<f64>,
) -> TrackExchange {
    let dt = target_time - exchange.timestamp;
    if dt.abs() < 1e-10 {
        return exchange.clone();
    }
    // Simple linear extrapolation: x_new = F * x, P_new = F*P*F' + Q
    let new_state = f * &exchange.state;
    let new_cov = f * &exchange.covariance * f.transpose() + q;
    TrackExchange {
        track_id: exchange.track_id,
        state: new_state,
        covariance: new_cov,
        timestamp: target_time,
        source_id: exchange.source_id,
    }
}

/// Align all track exchanges to the latest timestamp in the slice.
///
/// Each track whose timestamp is earlier than the latest is extrapolated
/// forward using the provided state transition `f` and process noise `q`.
pub fn align_to_common_time(tracks: &mut [TrackExchange], f: &DMatrix<f64>, q: &DMatrix<f64>) {
    let latest = tracks
        .iter()
        .map(|t| t.timestamp)
        .fold(f64::NEG_INFINITY, f64::max);
    for track in tracks.iter_mut() {
        if (track.timestamp - latest).abs() > 1e-10 {
            *track = extrapolate_track(track, latest, f, q);
        }
    }
}

// ---------------------------------------------------------------------------
// Cross-covariance-aware Mahalanobis
// ---------------------------------------------------------------------------

/// Augmented Mahalanobis distance accounting for cross-covariance between
/// two estimates.
///
/// `d² = (x₁ - x₂)ᵀ (P₁ + P₂ - P₁₂ - P₁₂ᵀ)⁻¹ (x₁ - x₂)`
pub fn augmented_mahalanobis_with_cross_cov(
    x1: &DVector<f64>,
    p1: &DMatrix<f64>,
    x2: &DVector<f64>,
    p2: &DMatrix<f64>,
    p12: &DMatrix<f64>,
) -> f64 {
    let diff = x1 - x2;
    let s = p1 + p2 - p12 - p12.transpose();
    match s.try_inverse() {
        Some(s_inv) => (diff.transpose() * s_inv * &diff)[(0, 0)],
        None => f64::MAX,
    }
}

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
    /// Which fusion algorithm to use for matched pairs.
    pub mode: FusionMode,
}

impl FederatedFusionManager {
    /// Create a new empty manager.
    pub fn new() -> Self {
        Self {
            site_tracks: Vec::new(),
            mode: FusionMode::CovarianceIntersection,
        }
    }

    /// Create a new manager with the specified fusion mode.
    pub fn with_mode(mode: FusionMode) -> Self {
        Self {
            site_tracks: Vec::new(),
            mode,
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

            // Fuse matched pairs using the configured mode
            for &(fi, ii) in &matches {
                matched_fused[fi] = true;
                matched_incoming[ii] = true;
                let fused_pair = match self.mode {
                    FusionMode::Naive => fuse_naive(&fused[fi], &incoming[ii]),
                    FusionMode::CovarianceIntersection => {
                        fuse_covariance_intersection(&fused[fi], &incoming[ii])
                    }
                };
                new_fused.push(fused_pair);
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
    fn test_fusion_mode_selects_algorithm() {
        // Create a manager with Naive mode and verify it uses naive fusion
        // (which produces a different result from CI).
        let mut mgr_naive = FederatedFusionManager::with_mode(FusionMode::Naive);
        let mut mgr_ci = FederatedFusionManager::with_mode(FusionMode::CovarianceIntersection);

        let tracks_a = vec![make_track(1, 1, &[10.0, 20.0], &[4.0, 4.0])];
        let tracks_b = vec![make_track(10, 2, &[11.0, 19.0], &[4.0, 4.0])];

        mgr_naive.submit_tracks(1, tracks_a.clone());
        mgr_naive.submit_tracks(2, tracks_b.clone());
        mgr_ci.submit_tracks(1, tracks_a);
        mgr_ci.submit_tracks(2, tracks_b);

        let fused_naive = mgr_naive.fuse(50.0);
        let fused_ci = mgr_ci.fuse(50.0);

        assert_eq!(fused_naive.len(), 1);
        assert_eq!(fused_ci.len(), 1);

        // Naive and CI should produce different covariances for the same inputs
        let tr_naive = fused_naive[0].covariance.trace();
        let tr_ci = fused_ci[0].covariance.trace();
        assert!(
            (tr_naive - tr_ci).abs() > 1e-6,
            "Naive and CI should produce different covariances, got naive={tr_naive}, ci={tr_ci}"
        );
    }

    #[test]
    fn test_extrapolate_track_zero_dt() {
        let track = make_track(1, 1, &[10.0, 1.0], &[2.0, 2.0]);
        let f = DMatrix::identity(2, 2);
        let q = DMatrix::zeros(2, 2);
        let result = extrapolate_track(&track, track.timestamp, &f, &q);
        assert_eq!(result.state, track.state);
        assert_eq!(result.covariance, track.covariance);
    }

    #[test]
    fn test_extrapolate_track_positive_dt() {
        let mut track = make_track(1, 1, &[10.0, 1.0], &[2.0, 2.0]);
        track.timestamp = 0.0;
        // F = [[1, dt], [0, 1]] with dt=1.0 (constant velocity)
        let dt = 1.0;
        let f = DMatrix::from_row_slice(2, 2, &[1.0, dt, 0.0, 1.0]);
        let q = DMatrix::from_diagonal(&DVector::from_column_slice(&[0.1, 0.1]));

        let result = extrapolate_track(&track, dt, &f, &q);
        assert!((result.timestamp - dt).abs() < 1e-10);
        // x_new = F * [10, 1] = [10 + 1*1, 1] = [11, 1]
        assert!((result.state[0] - 11.0).abs() < 1e-10);
        assert!((result.state[1] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_align_to_common_time() {
        let mut tracks = vec![
            TrackExchange {
                track_id: 1,
                state: DVector::from_column_slice(&[10.0, 1.0]),
                covariance: DMatrix::from_diagonal(&DVector::from_column_slice(&[1.0, 1.0])),
                timestamp: 0.0,
                source_id: 1,
            },
            TrackExchange {
                track_id: 2,
                state: DVector::from_column_slice(&[20.0, 2.0]),
                covariance: DMatrix::from_diagonal(&DVector::from_column_slice(&[1.0, 1.0])),
                timestamp: 0.5,
                source_id: 2,
            },
            TrackExchange {
                track_id: 3,
                state: DVector::from_column_slice(&[30.0, 3.0]),
                covariance: DMatrix::from_diagonal(&DVector::from_column_slice(&[1.0, 1.0])),
                timestamp: 1.0,
                source_id: 3,
            },
        ];

        let f = DMatrix::identity(2, 2);
        let q = DMatrix::zeros(2, 2);
        align_to_common_time(&mut tracks, &f, &q);

        // All tracks should now be at timestamp 1.0
        for track in &tracks {
            assert!(
                (track.timestamp - 1.0).abs() < 1e-10,
                "track {} should be at t=1.0, got {}",
                track.track_id,
                track.timestamp
            );
        }
    }

    #[test]
    fn test_augmented_mahalanobis_with_zero_cross_cov() {
        // With zero cross-covariance, should equal regular augmented Mahalanobis
        let a = make_track(1, 1, &[10.0, 20.0], &[4.0, 9.0]);
        let b = make_track(2, 2, &[11.0, 19.0], &[4.0, 9.0]);

        let d_regular = augmented_mahalanobis(&a, &b);
        let d_cross = augmented_mahalanobis_with_cross_cov(
            &a.state,
            &a.covariance,
            &b.state,
            &b.covariance,
            &DMatrix::zeros(2, 2),
        );

        assert!(
            (d_regular - d_cross).abs() < 1e-10,
            "zero cross-cov should match regular: {d_regular} vs {d_cross}"
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
