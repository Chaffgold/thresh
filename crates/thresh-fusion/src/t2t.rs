//! Track-to-track fusion: distributed/federated fusion of track-level outputs.
//!
//! Enables multiple tracker instances to merge track estimates without sharing
//! raw measurements. Provides:
//! - [`TrackExchange`] — lightweight track state for inter-site exchange
//! - [`t2t_association`] — augmented-state Mahalanobis + Hungarian matching
//! - [`fuse_naive`] — inverse-covariance-weighted average (assumes independence)
//! - [`fuse_covariance_intersection`] — CI fusion (safe when cross-covariances unknown)
//! - [`fuse_optimal`] — optimal fusion with known cross-covariance P₁₂
//! - [`FederatedFusionManager`] — multi-site fusion orchestrator

use std::collections::HashMap;

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
    /// Optimal fusion using explicit cross-covariance P₁₂.
    ///
    /// Falls back to CI when cross-covariance is not available for a pair
    /// or when the innovation matrix S is singular.
    OptimalWithCrossCovariance,
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

/// Optimal fusion of two track estimates using a known cross-covariance P₁₂.
///
/// Uses the Bar-Shalom distributed fusion formula:
/// ```text
/// S   = P₁ + P₂ - P₁₂ - P₁₂ᵀ
/// K   = (P₁ - P₁₂) S⁻¹
/// x_f = x₁ + K (x₂ - x₁)
/// P_f = P₁ - K (P₁ - P₁₂)ᵀ
/// ```
///
/// Returns `None` if S is singular, in which case the caller should fall back
/// to covariance intersection.
pub fn fuse_optimal(
    a: &TrackExchange,
    b: &TrackExchange,
    p12: &DMatrix<f64>,
) -> Option<TrackExchange> {
    let s = &a.covariance + &b.covariance - p12 - p12.transpose();
    let s_inv = s.try_inverse()?;
    let p1_minus_p12 = &a.covariance - p12;
    let k = &p1_minus_p12 * &s_inv;
    let x_fused = &a.state + &k * (&b.state - &a.state);
    let p_fused = &a.covariance - &k * p1_minus_p12.transpose();

    Some(TrackExchange {
        track_id: a.track_id,
        state: x_fused,
        covariance: p_fused,
        timestamp: a.timestamp.max(b.timestamp),
        source_id: 0,
    })
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
    /// Persistent fused track state maintained across calls to `fuse()`.
    fused_tracks: Vec<TrackExchange>,
    /// Optional timeout (seconds): fused tracks older than
    /// `latest_time - timeout` are pruned after each `fuse()` call.
    pub fused_track_timeout_s: Option<f64>,
    /// Known cross-covariances between track pairs, keyed by
    /// `(track_a_id, track_b_id)` where `track_a_id < track_b_id`.
    cross_covariances: HashMap<(u64, u64), DMatrix<f64>>,
}

impl FederatedFusionManager {
    /// Create a new empty manager.
    pub fn new() -> Self {
        Self {
            site_tracks: Vec::new(),
            mode: FusionMode::CovarianceIntersection,
            fused_tracks: Vec::new(),
            fused_track_timeout_s: None,
            cross_covariances: HashMap::new(),
        }
    }

    /// Create a new manager with the specified fusion mode.
    pub fn with_mode(mode: FusionMode) -> Self {
        Self {
            site_tracks: Vec::new(),
            mode,
            fused_tracks: Vec::new(),
            fused_track_timeout_s: None,
            cross_covariances: HashMap::new(),
        }
    }

    /// Store a cross-covariance matrix between two tracks identified by
    /// their track IDs. The key is stored with the smaller ID first.
    pub fn set_cross_covariance(&mut self, track_a_id: u64, track_b_id: u64, p12: DMatrix<f64>) {
        let key = if track_a_id <= track_b_id {
            (track_a_id, track_b_id)
        } else {
            (track_b_id, track_a_id)
        };
        self.cross_covariances.insert(key, p12);
    }

    /// Look up a stored cross-covariance between two track IDs.
    fn get_cross_covariance(&self, id_a: u64, id_b: u64) -> Option<&DMatrix<f64>> {
        let key = if id_a <= id_b {
            (id_a, id_b)
        } else {
            (id_b, id_a)
        };
        self.cross_covariances.get(&key)
    }

    /// Return a reference to the persistent fused track state.
    pub fn get_fused_tracks(&self) -> &[TrackExchange] {
        &self.fused_tracks
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
    pub fn fuse(&mut self, gate: f64) -> Vec<TrackExchange> {
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
                    FusionMode::OptimalWithCrossCovariance => {
                        if let Some(p12) =
                            self.get_cross_covariance(fused[fi].track_id, incoming[ii].track_id)
                        {
                            let p12 = p12.clone();
                            fuse_optimal(&fused[fi], &incoming[ii], &p12).unwrap_or_else(|| {
                                fuse_covariance_intersection(&fused[fi], &incoming[ii])
                            })
                        } else {
                            fuse_covariance_intersection(&fused[fi], &incoming[ii])
                        }
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

        // Prune fused tracks that have timed out.
        if let Some(timeout) = self.fused_track_timeout_s {
            let latest_time = fused
                .iter()
                .map(|t| t.timestamp)
                .fold(f64::NEG_INFINITY, f64::max);
            fused.retain(|t| latest_time - t.timestamp <= timeout);
        }

        // Persist for `get_fused_tracks()`.
        self.fused_tracks = fused.clone();

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

    // -- Task 3.3: manual extrapolation verification ---------------------------

    #[test]
    fn test_extrapolation_matches_manual() {
        // 6D constant-velocity state: [x, vx, y, vy, z, vz]
        let state = DVector::from_column_slice(&[100.0, 10.0, 200.0, -5.0, 300.0, 2.0]);
        let cov_diag = &[4.0, 1.0, 9.0, 2.0, 16.0, 3.0];
        let p = DMatrix::from_diagonal(&DVector::from_column_slice(cov_diag));
        let track = TrackExchange {
            track_id: 42,
            source_id: 1,
            state: state.clone(),
            covariance: p.clone(),
            timestamp: 5.0,
        };

        let dt = 2.0;
        // CV transition matrix for 3 position/velocity pairs:
        // [[1, dt, 0, 0,  0, 0 ],
        //  [0, 1,  0, 0,  0, 0 ],
        //  [0, 0,  1, dt, 0, 0 ],
        //  [0, 0,  0, 1,  0, 0 ],
        //  [0, 0,  0, 0,  1, dt],
        //  [0, 0,  0, 0,  0, 1 ]]
        #[rustfmt::skip]
        let f = DMatrix::from_row_slice(6, 6, &[
            1.0, dt,  0.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 1.0, dt,  0.0, 0.0,
            0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 1.0, dt,
            0.0, 0.0, 0.0, 0.0, 0.0, 1.0,
        ]);

        let q = DMatrix::from_diagonal(&DVector::from_column_slice(&[
            0.1, 0.01, 0.1, 0.01, 0.1, 0.01,
        ]));

        let result = extrapolate_track(&track, 5.0 + dt, &f, &q);

        // Manual: x_new = F * x
        let expected_state = &f * &state;
        // Manual: P_new = F * P * F' + Q
        let expected_cov = &f * &p * f.transpose() + &q;

        assert!(
            (result.timestamp - 7.0).abs() < 1e-10,
            "timestamp should be 7.0"
        );
        assert_eq!(result.state.len(), 6);
        for i in 0..6 {
            assert!(
                (result.state[i] - expected_state[i]).abs() < 1e-10,
                "state[{i}]: got {} expected {}",
                result.state[i],
                expected_state[i]
            );
        }
        for i in 0..6 {
            for j in 0..6 {
                assert!(
                    (result.covariance[(i, j)] - expected_cov[(i, j)]).abs() < 1e-10,
                    "cov[{i},{j}]: got {} expected {}",
                    result.covariance[(i, j)],
                    expected_cov[(i, j)]
                );
            }
        }

        // Spot-check expected values:
        // x_new[0] = 100 + 10*2 = 120
        assert!((result.state[0] - 120.0).abs() < 1e-10);
        // x_new[2] = 200 + (-5)*2 = 190
        assert!((result.state[2] - 190.0).abs() < 1e-10);
        // x_new[4] = 300 + 2*2 = 304
        assert!((result.state[4] - 304.0).abs() < 1e-10);
        // Velocities unchanged
        assert!((result.state[1] - 10.0).abs() < 1e-10);
        assert!((result.state[3] - -5.0).abs() < 1e-10);
        assert!((result.state[5] - 2.0).abs() < 1e-10);
    }

    // -- Task 5.1: track birth from unmatched incoming -------------------------

    #[test]
    fn test_federated_manager_births_unmatched_tracks() {
        let mut mgr = FederatedFusionManager::new();

        // Site 1: one track
        mgr.submit_tracks(1, vec![make_track(1, 1, &[10.0, 20.0], &[2.0, 2.0])]);

        // Site 2: one matching track + one completely different (should be birthed)
        mgr.submit_tracks(
            2,
            vec![
                make_track(10, 2, &[10.1, 19.9], &[3.0, 3.0]), // matches track 1
                make_track(20, 2, &[500.0, 600.0], &[2.0, 2.0]), // no match -> birth
            ],
        );

        let fused = mgr.fuse(20.0);

        // Should have 2 tracks: 1 fused pair + 1 birthed
        assert_eq!(
            fused.len(),
            2,
            "expected 2 fused tracks, got {}",
            fused.len()
        );

        // The birthed track should have source_id = 0 and state near [500, 600]
        let birthed = fused
            .iter()
            .find(|t| (t.state[0] - 500.0).abs() < 1.0)
            .expect("birthed track near [500, 600] should exist");
        assert_eq!(
            birthed.source_id, 0,
            "birthed track should have source_id=0"
        );
    }

    // -- Task 5.2: fused track timeout -----------------------------------------

    #[test]
    fn test_fused_track_timeout() {
        let mut mgr = FederatedFusionManager::with_mode(FusionMode::CovarianceIntersection);
        mgr.fused_track_timeout_s = Some(5.0);

        // Submit tracks at different times
        let mut track_old = make_track(1, 1, &[10.0, 20.0], &[2.0, 2.0]);
        track_old.timestamp = 0.0; // old

        let mut track_new = make_track(2, 1, &[50.0, 60.0], &[2.0, 2.0]);
        track_new.timestamp = 10.0; // recent

        mgr.submit_tracks(1, vec![track_old, track_new]);

        let fused = mgr.fuse(20.0);

        // The old track at t=0 should be pruned because latest=10, 10-0=10 > 5
        assert_eq!(
            fused.len(),
            1,
            "old track should have been pruned, got {} tracks",
            fused.len()
        );
        assert!(
            (fused[0].timestamp - 10.0).abs() < 1e-10,
            "remaining track should be the recent one"
        );
    }

    #[test]
    fn test_fused_track_timeout_no_prune_within_window() {
        let mut mgr = FederatedFusionManager::new();
        mgr.fused_track_timeout_s = Some(20.0);

        let mut t1 = make_track(1, 1, &[10.0, 20.0], &[2.0, 2.0]);
        t1.timestamp = 5.0;
        let mut t2 = make_track(2, 1, &[50.0, 60.0], &[2.0, 2.0]);
        t2.timestamp = 10.0;

        mgr.submit_tracks(1, vec![t1, t2]);
        let fused = mgr.fuse(20.0);

        // Both within window (10 - 5 = 5 <= 20)
        assert_eq!(fused.len(), 2, "both tracks should survive");
    }

    // -- Task 5.3: get_fused_tracks persists -----------------------------------

    #[test]
    fn test_get_fused_tracks_persists_across_calls() {
        let mut mgr = FederatedFusionManager::new();

        // Initially empty
        assert!(mgr.get_fused_tracks().is_empty());

        mgr.submit_tracks(1, vec![make_track(1, 1, &[10.0, 20.0], &[2.0, 2.0])]);
        mgr.submit_tracks(2, vec![make_track(10, 2, &[10.1, 19.9], &[3.0, 3.0])]);

        let result = mgr.fuse(20.0);
        assert_eq!(result.len(), 1);

        // get_fused_tracks should return same state
        let persisted = mgr.get_fused_tracks();
        assert_eq!(persisted.len(), 1);
        assert!((persisted[0].state[0] - result[0].state[0]).abs() < 1e-10);
    }

    // -- Task 6.1: integration — two sites, three overlapping targets ----------

    #[test]
    fn test_integration_two_sites_three_targets() {
        let mut mgr = FederatedFusionManager::with_mode(FusionMode::CovarianceIntersection);

        // True target positions (3 targets in 2D)
        let targets = [[100.0, 200.0], [500.0, 600.0], [900.0, 100.0]];

        // Site A: tracks with some noise
        let site_a: Vec<TrackExchange> = targets
            .iter()
            .enumerate()
            .map(|(i, &[x, y])| {
                let mut t = TrackExchange {
                    track_id: i as u64,
                    source_id: 1,
                    state: DVector::from_column_slice(&[x + 0.5, y - 0.3]),
                    covariance: DMatrix::from_diagonal(&DVector::from_column_slice(&[4.0, 4.0])),
                    timestamp: 10.0,
                };
                t.timestamp = 10.0;
                t
            })
            .collect();

        // Site B: tracks with different noise
        let site_b: Vec<TrackExchange> = targets
            .iter()
            .enumerate()
            .map(|(i, &[x, y])| TrackExchange {
                track_id: 10 + i as u64,
                source_id: 2,
                state: DVector::from_column_slice(&[x - 0.4, y + 0.6]),
                covariance: DMatrix::from_diagonal(&DVector::from_column_slice(&[5.0, 5.0])),
                timestamp: 10.0,
            })
            .collect();

        mgr.submit_tracks(1, site_a.clone());
        mgr.submit_tracks(2, site_b.clone());

        let fused = mgr.fuse(50.0);

        // Should have exactly 3 fused tracks
        assert_eq!(
            fused.len(),
            3,
            "expected 3 fused tracks, got {}",
            fused.len()
        );

        // Each fused track should have lower covariance than either input
        for ft in &fused {
            let tr_fused = ft.covariance.trace();
            // Site A trace = 8.0 (4+4), Site B trace = 10.0 (5+5).
            // CI is conservative, so fused trace will be less than the larger
            // input but may be slightly above the smaller — verify it beats
            // the worst-case input.
            assert!(
                tr_fused < 10.0,
                "fused trace {tr_fused} should be < largest input trace 10.0"
            );
        }
    }

    // -- Task 6.2: integration — asynchronous site updates ---------------------

    #[test]
    fn test_integration_asynchronous_updates() {
        let mut mgr = FederatedFusionManager::with_mode(FusionMode::CovarianceIntersection);

        // Simulate 4 time steps.
        // Site A submits every step (t=0,1,2,3).
        // Site B submits every other step (t=0,2).
        let cov = DMatrix::from_diagonal(&DVector::from_column_slice(&[4.0, 4.0]));

        for step in 0..4u64 {
            let t = step as f64;

            // Site A always submits
            let track_a = TrackExchange {
                track_id: 1,
                source_id: 1,
                state: DVector::from_column_slice(&[100.0 + t * 10.0, 200.0]),
                covariance: cov.clone(),
                timestamp: t,
            };
            mgr.submit_tracks(1, vec![track_a]);

            if step % 2 == 0 {
                // Site B submits every other step
                let track_b = TrackExchange {
                    track_id: 10,
                    source_id: 2,
                    state: DVector::from_column_slice(&[100.5 + t * 10.0, 199.5]),
                    covariance: cov.clone(),
                    timestamp: t,
                };
                mgr.submit_tracks(2, vec![track_b]);
            }

            let fused = mgr.fuse(50.0);

            // Should always produce exactly 1 fused track
            assert_eq!(
                fused.len(),
                1,
                "step {step}: expected 1 fused track, got {}",
                fused.len()
            );

            // Fused timestamp should be the latest of inputs
            assert!(
                fused[0].timestamp >= t - 1.0,
                "step {step}: fused timestamp {} too old",
                fused[0].timestamp
            );
        }
    }

    // -- Task 4.5: optimal fusion with cross-covariance -------------------------

    #[test]
    fn test_fuse_optimal_known_cross_cov() {
        let a = make_track(1, 1, &[10.0, 20.0], &[4.0, 9.0]);
        let b = make_track(2, 2, &[12.0, 18.0], &[5.0, 8.0]);

        // Moderate positive cross-covariance (correlated errors)
        let p12 = DMatrix::from_diagonal(&DVector::from_column_slice(&[1.5, 2.0]));

        let fused_opt = fuse_optimal(&a, &b, &p12).expect("S should be invertible");

        // Fused state should lie between the two inputs (component-wise)
        for i in 0..2 {
            let lo = a.state[i].min(b.state[i]);
            let hi = a.state[i].max(b.state[i]);
            assert!(
                fused_opt.state[i] >= lo - 1e-6 && fused_opt.state[i] <= hi + 1e-6,
                "fused state[{i}] = {} not between {} and {}",
                fused_opt.state[i],
                lo,
                hi,
            );
        }

        // Fused covariance should be tighter than CI
        let fused_ci = fuse_covariance_intersection(&a, &b);
        let tr_opt = fused_opt.covariance.trace();
        let tr_ci = fused_ci.covariance.trace();
        assert!(
            tr_opt < tr_ci + 1e-6,
            "optimal trace {tr_opt} should be <= CI trace {tr_ci}"
        );
    }

    #[test]
    fn test_fuse_optimal_falls_back_to_ci() {
        let a = make_track(1, 1, &[10.0, 20.0], &[4.0, 4.0]);
        let b = make_track(2, 2, &[12.0, 18.0], &[4.0, 4.0]);

        // P12 = (P1 + P2) / 2 makes S = P1 + P2 - P12 - P12' = 0 (singular)
        let p12 = (&a.covariance + &b.covariance) / 2.0;

        let result = fuse_optimal(&a, &b, &p12);
        assert!(result.is_none(), "should return None when S is singular");

        // Verify CI still works as fallback (no panic)
        let _ci = fuse_covariance_intersection(&a, &b);
    }

    #[test]
    fn test_manager_optimal_mode_with_cross_cov() {
        let mut mgr_opt = FederatedFusionManager::with_mode(FusionMode::OptimalWithCrossCovariance);
        let mut mgr_ci = FederatedFusionManager::with_mode(FusionMode::CovarianceIntersection);

        let tracks_a = vec![make_track(1, 1, &[10.0, 20.0], &[4.0, 9.0])];
        let tracks_b = vec![make_track(10, 2, &[12.0, 18.0], &[5.0, 8.0])];

        // Set cross-covariance for the optimal manager
        let p12 = DMatrix::from_diagonal(&DVector::from_column_slice(&[1.5, 2.0]));
        mgr_opt.set_cross_covariance(1, 10, p12);

        mgr_opt.submit_tracks(1, tracks_a.clone());
        mgr_opt.submit_tracks(2, tracks_b.clone());
        mgr_ci.submit_tracks(1, tracks_a);
        mgr_ci.submit_tracks(2, tracks_b);

        let fused_opt = mgr_opt.fuse(50.0);
        let fused_ci = mgr_ci.fuse(50.0);

        assert_eq!(fused_opt.len(), 1);
        assert_eq!(fused_ci.len(), 1);

        // Optimal fusion with known cross-covariance should be tighter than CI
        let tr_opt = fused_opt[0].covariance.trace();
        let tr_ci = fused_ci[0].covariance.trace();
        assert!(
            tr_opt < tr_ci + 1e-6,
            "optimal trace {tr_opt} should be <= CI trace {tr_ci}"
        );

        // Results should differ (optimal uses the cross-covariance info)
        let state_diff = (&fused_opt[0].state - &fused_ci[0].state).norm();
        assert!(
            state_diff > 1e-8 || (tr_opt - tr_ci).abs() > 1e-8,
            "optimal and CI should produce different results"
        );
    }
}
