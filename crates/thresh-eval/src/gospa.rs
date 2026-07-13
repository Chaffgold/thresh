//! GOSPA metric (Rahmathullah / García-Fernández / Svensson 2017), α = 2,
//! with exact localization / missed-target / false-track decomposition.
//!
//! Implementation of the `eval-consistency-metrics` OpenSpec change,
//! section 4.
//!
//! # Formulation
//!
//! For a ground-truth set `X` and an estimate set `Y`, cutoff `c > 0`, and
//! order `p >= 1`, the α = 2 GOSPA distance is
//!
//! ```text
//! gospa(X, Y) = ( min over assignments [ Σ_matched d(x, y)^p ]
//!                 + (c^p / 2) · (|unmatched X| + |unmatched Y|) )^(1/p)
//! ```
//!
//! where a pair may only be matched at base (Euclidean) distance `d < c`.
//! α = 2 is **fixed, not a parameter**: per Rahmathullah et al. 2017
//! (Proposition 1), α = 2 is the unique convention under which the metric
//! decomposes exactly into localization + missed + false components with
//! per-target penalty `c^p / 2` — and that decomposition is the contract
//! this module ships. Other α values are intentionally unsupported.
//!
//! # Assignment optimality via the existing Hungarian solver
//!
//! The optimal assignment is found with
//! `thresh_association::hungarian::hungarian_assignment`, passing the
//! **clamped** per-pair cost `min(d, c)^p` and gate `c^p`. Clamping is
//! load-bearing: the solver optimizes over the raw matrix entries and only
//! drops `cost >= gate` matches *after* optimization, so with unclamped
//! `d^p` costs an entry far above `c^p` can push the solver toward a
//! matching whose gated value is suboptimal (see the
//! `clamped_cost_yields_optimal_assignment` test for a concrete Euclidean
//! counter-example). With clamped costs the solver's padded square
//! objective is `Σ_matched d^p + c^p·(dim − m)`, which differs from the
//! GOSPA objective by the assignment-independent constant
//! `c^p·(dim − (|X| + |Y|)/2)`, so the minimizers coincide exactly
//! (design.md Decision 4).
//!
//! A pair at exactly `d = c` is gated out (counted as one miss plus one
//! false track, total `c^p`) rather than matched at cost `c^p`; the two
//! attributions have identical totals, so the metric value is unaffected
//! and the choice is deterministic.

use serde::{Deserialize, Serialize};
use thresh_association::hungarian::{AssignmentResult, hungarian_assignment};

use crate::matching::FrameData;

/// Parameters of the α = 2 GOSPA metric.
///
/// `c` has **no global default**: the repo's regimes span metres (nuScenes)
/// to tens of kilometres (orbital), so every caller must supply the cutoff
/// from its own context. The order `p` defaults to 2 via
/// [`GospaParams::new`].
///
/// Validity domain: `c > 0` and `p >= 1` (the metric axioms hold only
/// there). Values outside the domain are not validated and yield
/// meaningless results.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct GospaParams {
    /// Cutoff distance `c` in state units; pairs at `d >= c` cannot match.
    pub c: f64,
    /// Metric order `p >= 1` (default 2 via [`GospaParams::new`]).
    pub p: f64,
}

impl GospaParams {
    /// Parameters with the given cutoff `c` and the default order `p = 2`.
    pub fn new(c: f64) -> Self {
        Self { c, p: 2.0 }
    }

    /// The gate value `c^p` shared by the cost clamp, the Hungarian gate,
    /// and the `c^p / 2` per-target cardinality penalty.
    fn cutoff_power(&self) -> f64 {
        self.c.powf(self.p)
    }
}

/// Single-frame GOSPA value with its exact α = 2 decomposition.
///
/// Invariant (auditable, tested to floating-point tolerance):
/// `localization + missed + false_tracks == gospa^p`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GospaResult {
    /// Total GOSPA distance `(localization + missed + false_tracks)^(1/p)`.
    pub gospa: f64,
    /// Localization component: `Σ d^p` over assigned pairs.
    pub localization: f64,
    /// Missed-target component: `c^p / 2` per unassigned ground truth.
    pub missed: f64,
    /// False-track component: `c^p / 2` per unassigned estimate.
    pub false_tracks: f64,
    /// Number of assigned (matched) pairs.
    pub n_matched: usize,
    /// Number of unassigned ground truths.
    pub n_missed: usize,
    /// Number of unassigned estimates.
    pub n_false: usize,
}

/// Sequence-level GOSPA summary: per-frame results, the order-p mean of the
/// per-frame totals, and the summed decomposition components.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GospaSummary {
    /// Per-frame GOSPA results, in input frame order.
    pub frames: Vec<GospaResult>,
    /// Order-p mean of the per-frame totals:
    /// `((1/N) Σ gospa_t^p)^(1/p)`; `0.0` for an empty sequence.
    pub mean_gospa: f64,
    /// Sum of the per-frame localization components.
    pub localization: f64,
    /// Sum of the per-frame missed-target components.
    pub missed: f64,
    /// Sum of the per-frame false-track components.
    pub false_tracks: f64,
}

/// Compute the α = 2 GOSPA distance between a frame's ground-truth and
/// track position sets.
///
/// The assignment is optimal (minimum total cost) via the existing
/// Hungarian solver on the clamped cost `min(d, c)^p` gated at `c^p`; see
/// the module docs for why clamping (not raw `d^p`) is required for
/// optimality. Empty sets are pure cardinality cases: two empty sets yield
/// distance 0, an empty track set yields all-missed, an empty truth set
/// yields all-false.
pub fn gospa_frame(frame: &FrameData, params: &GospaParams) -> GospaResult {
    if frame.gt.is_empty() || frame.tracks.is_empty() {
        return finalize_result(0.0, 0, frame.gt.len(), frame.tracks.len(), params);
    }
    let cost = build_clamped_cost(&frame.gt, &frame.tracks, params);
    let assignment = hungarian_assignment(&cost, params.cutoff_power());
    assemble_result(&assignment, params)
}

/// Compute per-frame GOSPA over a sequence plus the sequence summary.
///
/// GOSPA is a set metric per time step: there are no cross-frame assignment
/// terms (that is trajectory-GOSPA, explicitly out of scope). The summary
/// carries the order-p mean of the per-frame totals and the summed
/// decomposition components. Evaluation is deterministic: identical inputs
/// and parameters produce bitwise-identical output, including when the
/// optimal assignment is non-unique (the Hungarian solver's tie-breaking is
/// deterministic).
pub fn gospa_sequence(frames: &[FrameData], params: &GospaParams) -> GospaSummary {
    let per_frame: Vec<GospaResult> = frames.iter().map(|f| gospa_frame(f, params)).collect();
    let (localization, missed, false_tracks) = sum_decomposition(&per_frame);
    let mean_gospa = order_p_mean(&per_frame, params.p);
    GospaSummary {
        frames: per_frame,
        mean_gospa,
        localization,
        missed,
        false_tracks,
    }
}

// ---------------------------------------------------------------------------
// Phase helpers
// ---------------------------------------------------------------------------

/// Euclidean distance between two 3D positions.
fn point_distance(a: &[f64; 3], b: &[f64; 3]) -> f64 {
    ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt()
}

/// Build the clamped GOSPA cost matrix `cost[i][j] = min(d(gt_i, track_j), c)^p`.
///
/// Clamping at the cutoff (rather than passing raw `d^p`) makes the
/// Hungarian objective equal the GOSPA objective up to an
/// assignment-independent constant, so the returned assignment is
/// GOSPA-optimal (module docs; design.md Decision 4).
fn build_clamped_cost(
    gt: &[(u64, [f64; 3])],
    tracks: &[(u64, [f64; 3])],
    params: &GospaParams,
) -> Vec<Vec<f64>> {
    gt.iter()
        .map(|(_, g)| {
            tracks
                .iter()
                .map(|(_, t)| point_distance(g, t).min(params.c).powf(params.p))
                .collect()
        })
        .collect()
}

/// Convert a gated Hungarian assignment into a [`GospaResult`].
///
/// `assignment.total_cost` sums exactly the kept (feasible) matches' clamped
/// costs, which is the localization component.
fn assemble_result(assignment: &AssignmentResult, params: &GospaParams) -> GospaResult {
    finalize_result(
        assignment.total_cost,
        assignment.matches.len(),
        assignment.unassigned_rows.len(),
        assignment.unassigned_cols.len(),
        params,
    )
}

/// Assemble a [`GospaResult`] from the localization sum and the three
/// cardinalities, applying the `c^p / 2` per-target penalty and the final
/// `1/p` root.
fn finalize_result(
    localization: f64,
    n_matched: usize,
    n_missed: usize,
    n_false: usize,
    params: &GospaParams,
) -> GospaResult {
    let half_cutoff_power = params.cutoff_power() / 2.0;
    let missed = half_cutoff_power * n_missed as f64;
    let false_tracks = half_cutoff_power * n_false as f64;
    let gospa = (localization + missed + false_tracks).powf(1.0 / params.p);
    GospaResult {
        gospa,
        localization,
        missed,
        false_tracks,
        n_matched,
        n_missed,
        n_false,
    }
}

/// Sum the decomposition components over per-frame results, in frame order.
fn sum_decomposition(results: &[GospaResult]) -> (f64, f64, f64) {
    results.iter().fold((0.0, 0.0, 0.0), |(l, m, f), r| {
        (l + r.localization, m + r.missed, f + r.false_tracks)
    })
}

/// Order-p mean of the per-frame totals: `((1/N) Σ gospa_t^p)^(1/p)`.
/// Returns `0.0` for an empty slice.
fn order_p_mean(results: &[GospaResult], p: f64) -> f64 {
    if results.is_empty() {
        return 0.0;
    }
    let sum_p: f64 = results.iter().map(|r| r.gospa.powf(p)).sum();
    (sum_p / results.len() as f64).powf(1.0 / p)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hota::compute_hota_at_threshold;
    use crate::metrics::{compute_idf1, compute_mot_metrics};
    use rand::prelude::*;

    // ---- test helpers -----------------------------------------------------

    /// Build a `FrameData` from bare point sets (IDs are irrelevant to GOSPA).
    fn frame_between(x: &[[f64; 3]], y: &[[f64; 3]]) -> FrameData {
        FrameData {
            gt: x.iter().enumerate().map(|(i, p)| (i as u64, *p)).collect(),
            tracks: y
                .iter()
                .enumerate()
                .map(|(i, p)| (100 + i as u64, *p))
                .collect(),
        }
    }

    /// Random point set with cardinality in `0..=max_card`, coordinates in
    /// a cube of the given span centred on the origin.
    fn random_points(rng: &mut StdRng, max_card: usize, span: f64) -> Vec<[f64; 3]> {
        let n = rng.random_range(0..=max_card);
        (0..n)
            .map(|_| {
                [
                    span * (rng.random::<f64>() - 0.5),
                    span * (rng.random::<f64>() - 0.5),
                    span * (rng.random::<f64>() - 0.5),
                ]
            })
            .collect()
    }

    /// Brute-force minimum of the GOSPA objective (`total^p`) by enumerating
    /// every partial assignment of ground truths to distinct tracks.
    fn brute_force_gospa_p(frame: &FrameData, params: &GospaParams) -> f64 {
        let mut used = vec![false; frame.tracks.len()];
        brute_force_recurse(frame, &mut used, 0, params)
    }

    /// Recursive core of the brute force: gt row `idx` is either left
    /// unmatched (cost `c^p / 2`) or matched to any unused track at clamped
    /// cost `min(d, c)^p`; unmatched tracks are charged at the leaf.
    fn brute_force_recurse(
        frame: &FrameData,
        used: &mut Vec<bool>,
        idx: usize,
        params: &GospaParams,
    ) -> f64 {
        let half_cp = params.c.powf(params.p) / 2.0;
        if idx == frame.gt.len() {
            let unused = used.iter().filter(|&&u| !u).count();
            return half_cp * unused as f64;
        }
        let mut best = half_cp + brute_force_recurse(frame, used, idx + 1, params);
        for j in 0..frame.tracks.len() {
            if used[j] {
                continue;
            }
            let d = point_distance(&frame.gt[idx].1, &frame.tracks[j].1);
            let pair = d.min(params.c).powf(params.p);
            used[j] = true;
            let candidate = pair + brute_force_recurse(frame, used, idx + 1, params);
            used[j] = false;
            best = best.min(candidate);
        }
        best
    }

    /// Assert two per-frame results are bitwise identical.
    fn assert_result_bitwise_eq(a: &GospaResult, b: &GospaResult) {
        assert_eq!(a.gospa.to_bits(), b.gospa.to_bits(), "gospa differs");
        assert_eq!(
            a.localization.to_bits(),
            b.localization.to_bits(),
            "localization differs"
        );
        assert_eq!(a.missed.to_bits(), b.missed.to_bits(), "missed differs");
        assert_eq!(
            a.false_tracks.to_bits(),
            b.false_tracks.to_bits(),
            "false_tracks differs"
        );
        assert_eq!(
            (a.n_matched, a.n_missed, a.n_false),
            (b.n_matched, b.n_missed, b.n_false),
            "counts differ"
        );
    }

    /// Assert two sequence summaries are bitwise identical.
    fn assert_summary_bitwise_eq(a: &GospaSummary, b: &GospaSummary) {
        assert_eq!(a.frames.len(), b.frames.len());
        for (x, y) in a.frames.iter().zip(&b.frames) {
            assert_result_bitwise_eq(x, y);
        }
        assert_eq!(a.mean_gospa.to_bits(), b.mean_gospa.to_bits());
        assert_eq!(a.localization.to_bits(), b.localization.to_bits());
        assert_eq!(a.missed.to_bits(), b.missed.to_bits());
        assert_eq!(a.false_tracks.to_bits(), b.false_tracks.to_bits());
    }

    // ---- 4.1: worked example and empty-set edge cases ----------------------

    // Spec scenario: "Known worked example".
    #[test]
    fn worked_example_c10_p2() {
        let frame = frame_between(&[[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]], &[[1.0, 0.0, 0.0]]);
        let r = gospa_frame(&frame, &GospaParams::new(10.0));
        // (0,0) pairs with (1,0) at cost 1^2 = 1; (10,0) is missed at
        // 10^2 / 2 = 50; total = sqrt(51).
        assert!((r.gospa - 51.0_f64.sqrt()).abs() < 1e-12, "{}", r.gospa);
        assert!((r.localization - 1.0).abs() < 1e-12);
        assert!((r.missed - 50.0).abs() < 1e-12);
        assert!((r.false_tracks - 0.0).abs() < 1e-12);
        assert_eq!((r.n_matched, r.n_missed, r.n_false), (1, 1, 0));
    }

    // Spec scenario: "Empty-set edge cases" (both empty).
    #[test]
    fn empty_sets_yield_zero_distance() {
        let frame = frame_between(&[], &[]);
        let r = gospa_frame(&frame, &GospaParams::new(10.0));
        assert_eq!(r.gospa.to_bits(), 0.0_f64.to_bits());
        assert_eq!(r.localization, 0.0);
        assert_eq!(r.missed, 0.0);
        assert_eq!(r.false_tracks, 0.0);
        assert_eq!((r.n_matched, r.n_missed, r.n_false), (0, 0, 0));
    }

    // Spec scenario: "Empty-set edge cases" (empty estimate → all missed).
    #[test]
    fn empty_estimate_is_all_missed() {
        let truths = [[0.0, 0.0, 0.0], [100.0, 0.0, 0.0], [0.0, 100.0, 0.0]];
        let frame = frame_between(&truths, &[]);
        let r = gospa_frame(&frame, &GospaParams::new(10.0));
        // (k · c^p / 2)^(1/p) = (3 · 50)^(1/2).
        assert!((r.gospa - 150.0_f64.sqrt()).abs() < 1e-12, "{}", r.gospa);
        assert!((r.missed - 150.0).abs() < 1e-12);
        assert_eq!((r.n_matched, r.n_missed, r.n_false), (0, 3, 0));
    }

    // Spec scenario: "Pure-miss configuration isolates the missed component".
    #[test]
    fn pure_miss_isolates_missed_component() {
        let truths = [[0.0, 0.0, 0.0], [100.0, 0.0, 0.0], [0.0, 100.0, 0.0]];
        let frame = frame_between(&truths, &[]);
        let params = GospaParams::new(10.0);
        let r = gospa_frame(&frame, &params);
        // Localization and false components exactly zero; missed accounts
        // for the entire total.
        assert_eq!(r.localization.to_bits(), 0.0_f64.to_bits());
        assert_eq!(r.false_tracks.to_bits(), 0.0_f64.to_bits());
        assert!((r.missed - r.gospa.powf(params.p)).abs() <= 1e-12 * (1.0 + r.missed));
    }

    // Spec scenario: "Empty-set edge cases" (symmetric all-false case).
    #[test]
    fn empty_truth_is_all_false() {
        let estimates = [[0.0, 0.0, 0.0], [100.0, 0.0, 0.0], [0.0, 100.0, 0.0]];
        let frame = frame_between(&[], &estimates);
        let r = gospa_frame(&frame, &GospaParams::new(10.0));
        // Same value as the all-missed case, attributed to false tracks.
        assert!((r.gospa - 150.0_f64.sqrt()).abs() < 1e-12, "{}", r.gospa);
        assert!((r.false_tracks - 150.0).abs() < 1e-12);
        assert_eq!(r.localization.to_bits(), 0.0_f64.to_bits());
        assert_eq!(r.missed.to_bits(), 0.0_f64.to_bits());
        assert_eq!((r.n_matched, r.n_missed, r.n_false), (0, 0, 3));
    }

    /// A pair at exactly d = c is gated out (one miss + one false, total
    /// c^p) rather than matched at cost c^p — identical totals, so the
    /// metric value is unaffected and the attribution is deterministic.
    #[test]
    fn pair_at_exactly_cutoff_counts_as_miss_and_false() {
        let frame = frame_between(&[[0.0, 0.0, 0.0]], &[[10.0, 0.0, 0.0]]);
        let r = gospa_frame(&frame, &GospaParams::new(10.0));
        assert!((r.gospa - 10.0).abs() < 1e-12);
        assert_eq!((r.n_matched, r.n_missed, r.n_false), (0, 1, 1));
        assert!((r.missed - 50.0).abs() < 1e-12);
        assert!((r.false_tracks - 50.0).abs() < 1e-12);
    }

    // ---- 4.2: assignment optimality and decomposition ----------------------

    // Hand-solved instances pinning the constant-offset equivalence between
    // the padded Hungarian objective and the GOSPA objective (design.md
    // Decision 4's proof): with clamped costs min(d,c)^p and gate c^p, the
    // solver pads to a dim x dim square with gate-valued dummies, so every
    // perfect matching's internal objective is
    //     Σ_matched d^p + c^p·(dim − m)          (m = feasible real matches)
    // while the GOSPA objective is
    //     Σ_matched d^p + (c^p/2)·(|X| + |Y| − 2m).
    // Their difference, c^p·(dim − (|X| + |Y|)/2), does not depend on the
    // matching, so the minimizers coincide.
    #[test]
    fn hungarian_gospa_objective_constant_offset() {
        let params = GospaParams::new(10.0);
        let cp = 100.0;

        // 3x3 (square): gt x-coords {0, 20, 40}, tracks {1, 22, 200}.
        // Clamped d^2 matrix: [[1,100,100],[100,4,100],[100,100,100]].
        // Hand-solved optimum: (g0,t0)=1, (g1,t1)=4, third pair gated →
        // loc 5, one miss, one false; GOSPA^p = 5 + 50 + 50 = 105.
        let gt = [[0.0, 0.0, 0.0], [20.0, 0.0, 0.0], [40.0, 0.0, 0.0]];
        let tracks3 = [[1.0, 0.0, 0.0], [22.0, 0.0, 0.0], [200.0, 0.0, 0.0]];
        let r3 = gospa_frame(&frame_between(&gt, &tracks3), &params);
        assert!((r3.localization - 5.0).abs() < 1e-9);
        assert_eq!((r3.n_matched, r3.n_missed, r3.n_false), (2, 1, 1));
        let gospa_p3 = r3.localization + r3.missed + r3.false_tracks;
        assert!((gospa_p3 - 105.0).abs() < 1e-9);
        // Internal padded objective = loc + c^p·(dim − m) = 5 + 100·1 = 105;
        // offset = c^p·(dim − (|X|+|Y|)/2) = 100·(3 − 3) = 0.
        let dim3 = 3.0;
        let internal3 = r3.localization + cp * (dim3 - r3.n_matched as f64);
        let offset3 = cp * (dim3 - (3.0 + 3.0) / 2.0);
        assert!((internal3 - gospa_p3 - offset3).abs() < 1e-9);
        // Cross-check optimality against exhaustive enumeration.
        let bf3 = brute_force_gospa_p(&frame_between(&gt, &tracks3), &params);
        assert!((gospa_p3 - bf3).abs() < 1e-9);

        // 3x2 (rectangular, non-zero offset): tracks {1, 22} only.
        // Optimum: (g0,t0)=1, (g1,t1)=4, g2 missed → GOSPA^p = 5 + 50 = 55.
        // Internal = 5 + 100·(3 − 2) = 105; offset = 100·(3 − 2.5) = 50.
        let tracks2 = [[1.0, 0.0, 0.0], [22.0, 0.0, 0.0]];
        let r2 = gospa_frame(&frame_between(&gt, &tracks2), &params);
        assert!((r2.localization - 5.0).abs() < 1e-9);
        assert_eq!((r2.n_matched, r2.n_missed, r2.n_false), (2, 1, 0));
        let gospa_p2 = r2.localization + r2.missed + r2.false_tracks;
        assert!((gospa_p2 - 55.0).abs() < 1e-9);
        let internal2 = r2.localization + cp * (dim3 - r2.n_matched as f64);
        let offset2 = cp * (dim3 - (3.0 + 2.0) / 2.0);
        assert!((internal2 - gospa_p2 - offset2).abs() < 1e-9);
    }

    // Euclidean counter-example proving the cost clamp is load-bearing:
    // raw d^2 matrix [[104.04, 16], [538.24, 81]] (gate 100). Raw costing
    // would pick the diagonal (185.04 < 554.24), whose gated value is
    // 100→dropped + 81 → GOSPA^p = 81 + 50 + 50 = 181. The GOSPA optimum
    // matches g0–t1 at 16, leaving g1 and t0 unmatched → 16 + 50 + 50 = 116.
    // Clamping at c^p makes the Hungarian objective the GOSPA objective, so
    // the solver returns the true optimum.
    #[test]
    fn clamped_cost_yields_optimal_assignment() {
        let params = GospaParams::new(10.0);
        let frame = frame_between(
            &[[0.0, 0.0, 0.0], [13.0, 0.0, 0.0]],
            &[[-10.2, 0.0, 0.0], [4.0, 0.0, 0.0]],
        );
        let r = gospa_frame(&frame, &params);
        let gospa_p = r.localization + r.missed + r.false_tracks;
        assert!((gospa_p - 116.0).abs() < 1e-9, "got {gospa_p}, want 116");
        assert!((r.localization - 16.0).abs() < 1e-9);
        assert_eq!((r.n_matched, r.n_missed, r.n_false), (1, 1, 1));
        let bf = brute_force_gospa_p(&frame, &params);
        assert!((gospa_p - bf).abs() < 1e-9);
    }

    // Spec scenario: "Decomposition sums to the total".
    #[test]
    fn decomposition_identity_on_random_frames() {
        let mut rng = StdRng::seed_from_u64(42);
        for _ in 0..200 {
            let x = random_points(&mut rng, 8, 100.0);
            let y = random_points(&mut rng, 8, 100.0);
            let frame = frame_between(&x, &y);
            for p in [1.0, 2.0] {
                let r = gospa_frame(&frame, &GospaParams { c: 15.0, p });
                assert!(r.localization >= 0.0);
                assert!(r.missed >= 0.0);
                assert!(r.false_tracks >= 0.0);
                let sum = r.localization + r.missed + r.false_tracks;
                let total_p = r.gospa.powf(p);
                assert!(
                    (sum - total_p).abs() <= 1e-12 * (1.0 + sum),
                    "p={p}: {sum} vs {total_p}"
                );
            }
        }
    }

    // Orbital-magnitude regression (design Risk: d^p magnitudes): km-scale
    // distances at LEO-scale coordinates, c = 5e3 m, cross-checked against
    // exhaustive assignment enumeration.
    #[test]
    fn orbital_magnitude_matches_brute_force() {
        let mut rng = StdRng::seed_from_u64(2026);
        let params = GospaParams::new(5.0e3);
        for _ in 0..50 {
            let frame = random_orbital_frame(&mut rng);
            let r = gospa_frame(&frame, &params);
            let gospa_p = r.localization + r.missed + r.false_tracks;
            let expected = brute_force_gospa_p(&frame, &params);
            assert!(
                (gospa_p - expected).abs() <= 1e-9 * (1.0 + expected),
                "{gospa_p} vs brute-force {expected}"
            );
        }
    }

    /// Random frame at LEO-like coordinate magnitudes (~7e6 m) with track
    /// offsets up to ±8e3 m per axis, straddling the c = 5e3 m cutoff.
    fn random_orbital_frame(rng: &mut StdRng) -> FrameData {
        let n_gt = rng.random_range(1..=4);
        let n_tr = rng.random_range(0..=4);
        let gt: Vec<(u64, [f64; 3])> = (0..n_gt)
            .map(|i| {
                (
                    i as u64,
                    [
                        7.0e6 + 5.0e4 * (rng.random::<f64>() - 0.5),
                        1.0e6 + 5.0e4 * (rng.random::<f64>() - 0.5),
                        2.0e5 + 5.0e4 * (rng.random::<f64>() - 0.5),
                    ],
                )
            })
            .collect();
        let tracks: Vec<(u64, [f64; 3])> = (0..n_tr)
            .map(|j| {
                let base = gt[rng.random_range(0..n_gt)].1;
                (
                    100 + j as u64,
                    [
                        base[0] + 1.6e4 * (rng.random::<f64>() - 0.5),
                        base[1] + 1.6e4 * (rng.random::<f64>() - 0.5),
                        base[2] + 1.6e4 * (rng.random::<f64>() - 0.5),
                    ],
                )
            })
            .collect();
        FrameData { gt, tracks }
    }

    // ---- 4.3: metric axioms -------------------------------------------------

    // Spec scenario: "Identity and symmetry" (identity half).
    #[test]
    fn self_distance_is_exactly_zero() {
        let x = [[1.5, -2.0, 3.0], [10.0, 4.0, -1.0], [-7.0, 0.5, 2.5]];
        let r = gospa_frame(&frame_between(&x, &x), &GospaParams::new(10.0));
        assert_eq!(r.gospa.to_bits(), 0.0_f64.to_bits(), "self-distance != 0");
        assert_eq!(r.localization.to_bits(), 0.0_f64.to_bits());
        assert_eq!(r.missed.to_bits(), 0.0_f64.to_bits());
        assert_eq!(r.false_tracks.to_bits(), 0.0_f64.to_bits());
        assert_eq!((r.n_matched, r.n_missed, r.n_false), (3, 0, 0));

        // Only-if direction: a distinct set has strictly positive distance.
        let shifted = [[2.5, -2.0, 3.0], [10.0, 4.0, -1.0], [-7.0, 0.5, 2.5]];
        let r2 = gospa_frame(&frame_between(&x, &shifted), &GospaParams::new(10.0));
        assert!(r2.gospa > 0.0);
    }

    // Spec scenario: "Identity and symmetry" (symmetry half).
    #[test]
    fn symmetry_under_argument_swap() {
        let params = GospaParams::new(10.0);
        // Generic (irrational-distance) points so the optimum is unique.
        let x = [[0.3, 1.7, -0.4], [12.1, 3.3, 0.9], [45.0, -8.0, 2.0]];
        let y = [[1.1, 2.2, 0.1], [13.0, 2.9, 1.4]];
        let fwd = gospa_frame(&frame_between(&x, &y), &params);
        let rev = gospa_frame(&frame_between(&y, &x), &params);
        assert!(
            (fwd.gospa - rev.gospa).abs() <= 1e-12 * (1.0 + fwd.gospa),
            "{} vs {}",
            fwd.gospa,
            rev.gospa
        );
        // The decomposition mirrors: misses in one order are false tracks
        // in the other.
        assert_eq!(fwd.n_matched, rev.n_matched);
        assert_eq!(fwd.n_missed, rev.n_false);
        assert_eq!(fwd.n_false, rev.n_missed);
        assert_eq!(fwd.missed.to_bits(), rev.false_tracks.to_bits());
        assert_eq!(fwd.false_tracks.to_bits(), rev.missed.to_bits());
    }

    // Spec scenario: "Triangle inequality on randomized sets".
    #[test]
    fn triangle_inequality_on_random_triples() {
        let mut rng = StdRng::seed_from_u64(7);
        for p in [1.0, 2.0] {
            let params = GospaParams { c: 10.0, p };
            for trial in 0..120 {
                let x = random_points(&mut rng, 5, 40.0);
                let y = random_points(&mut rng, 5, 40.0);
                let z = random_points(&mut rng, 5, 40.0);
                let d_xz = gospa_frame(&frame_between(&x, &z), &params).gospa;
                let d_xy = gospa_frame(&frame_between(&x, &y), &params).gospa;
                let d_yz = gospa_frame(&frame_between(&y, &z), &params).gospa;
                assert!(
                    d_xz <= d_xy + d_yz + 1e-9,
                    "p={p} trial={trial}: d(X,Z)={d_xz} > d(X,Y)+d(Y,Z)={}",
                    d_xy + d_yz
                );
            }
        }
    }

    // ---- 4.4: sequence and determinism --------------------------------------

    /// A small mixed sequence with matches, misses, false tracks, and one
    /// frame whose optimal assignment is non-unique (all four pairwise
    /// distances equal sqrt(26)).
    fn mixed_sequence() -> Vec<FrameData> {
        vec![
            frame_between(
                &[[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]],
                &[[1.0, 0.0, 0.0], [10.5, 0.2, 0.0]],
            ),
            frame_between(&[[0.0, 0.0, 0.0], [50.0, 0.0, 0.0]], &[[1.5, 0.0, 0.0]]),
            frame_between(&[[2.0, 0.0, 0.0]], &[[2.5, 0.0, 0.0], [90.0, 90.0, 0.0]]),
            // Tie frame: every gt/track pair is at distance sqrt(26).
            frame_between(
                &[[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]],
                &[[5.0, 1.0, 0.0], [5.0, -1.0, 0.0]],
            ),
            frame_between(&[], &[]),
        ]
    }

    // Spec scenario: "Sequence evaluation alongside MOT metrics".
    #[test]
    fn sequence_gospa_leaves_mot_metrics_unchanged() {
        let frames = mixed_sequence();
        let threshold = 5.0;
        let (mota_before, motp_before, idsw_before) = compute_mot_metrics(&frames, threshold);
        let idf1_before = compute_idf1(&frames, threshold);
        let hota_before = compute_hota_at_threshold(&frames, threshold);

        let params = GospaParams::new(5.0);
        let summary = gospa_sequence(&frames, &params);

        // Per-frame values and the summary are present and self-consistent.
        assert_eq!(summary.frames.len(), frames.len());
        let (loc, missed, false_tracks) = sum_decomposition(&summary.frames);
        assert_eq!(summary.localization.to_bits(), loc.to_bits());
        assert_eq!(summary.missed.to_bits(), missed.to_bits());
        assert_eq!(summary.false_tracks.to_bits(), false_tracks.to_bits());
        assert!(summary.mean_gospa > 0.0);

        // Every pre-existing MOT value is unchanged by scoring GOSPA.
        let (mota_after, motp_after, idsw_after) = compute_mot_metrics(&frames, threshold);
        let idf1_after = compute_idf1(&frames, threshold);
        let hota_after = compute_hota_at_threshold(&frames, threshold);
        assert_eq!(mota_before.to_bits(), mota_after.to_bits());
        assert_eq!(motp_before.to_bits(), motp_after.to_bits());
        assert_eq!(idsw_before, idsw_after);
        assert_eq!(idf1_before.to_bits(), idf1_after.to_bits());
        assert_eq!(hota_before.0.to_bits(), hota_after.0.to_bits());
        assert_eq!(hota_before.1.to_bits(), hota_after.1.to_bits());
        assert_eq!(hota_before.2.to_bits(), hota_after.2.to_bits());
    }

    // Spec scenario: "Deterministic evaluation" (includes the constructed
    // non-unique-assignment tie frame in `mixed_sequence`).
    #[test]
    fn double_evaluation_is_bitwise_identical() {
        let frames = mixed_sequence();
        let params = GospaParams::new(5.0);
        let first = gospa_sequence(&frames, &params);
        let second = gospa_sequence(&frames, &params);
        assert_summary_bitwise_eq(&first, &second);

        // The tie frame alone is also deterministic across repeated calls.
        let tie = frame_between(
            &[[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]],
            &[[5.0, 1.0, 0.0], [5.0, -1.0, 0.0]],
        );
        let tie_params = GospaParams::new(10.0);
        let a = gospa_frame(&tie, &tie_params);
        let b = gospa_frame(&tie, &tie_params);
        assert_result_bitwise_eq(&a, &b);
        // All four pairwise distances are sqrt(26) < 10, so both matches
        // land regardless of which optimal assignment is chosen.
        assert_eq!((a.n_matched, a.n_missed, a.n_false), (2, 0, 0));
    }

    // ---- per-helper unit tests (phase-helper convention) ---------------------

    #[test]
    fn build_clamped_cost_clamps_at_cutoff_power() {
        let params = GospaParams::new(10.0);
        let gt = vec![(0u64, [0.0, 0.0, 0.0])];
        let tracks = vec![(100u64, [3.0, 4.0, 0.0]), (101u64, [300.0, 0.0, 0.0])];
        let cost = build_clamped_cost(&gt, &tracks, &params);
        assert_eq!(cost.len(), 1);
        assert!((cost[0][0] - 25.0).abs() < 1e-12); // d = 5 → 25 (unclamped)
        assert!((cost[0][1] - 100.0).abs() < 1e-12); // d = 300 → clamped to c^p
    }

    #[test]
    fn order_p_mean_matches_hand_computation() {
        let mk = |gospa: f64| GospaResult {
            gospa,
            localization: 0.0,
            missed: 0.0,
            false_tracks: 0.0,
            n_matched: 0,
            n_missed: 0,
            n_false: 0,
        };
        let results = [mk(3.0), mk(4.0)];
        // p = 2: sqrt((9 + 16) / 2) = sqrt(12.5); p = 1: (3 + 4) / 2 = 3.5.
        assert!((order_p_mean(&results, 2.0) - 12.5_f64.sqrt()).abs() < 1e-12);
        assert!((order_p_mean(&results, 1.0) - 3.5).abs() < 1e-12);
        assert_eq!(order_p_mean(&[], 2.0).to_bits(), 0.0_f64.to_bits());
    }

    #[test]
    fn empty_sequence_summary_is_zero() {
        let summary = gospa_sequence(&[], &GospaParams::new(10.0));
        assert!(summary.frames.is_empty());
        assert_eq!(summary.mean_gospa.to_bits(), 0.0_f64.to_bits());
        assert_eq!(summary.localization.to_bits(), 0.0_f64.to_bits());
        assert_eq!(summary.missed.to_bits(), 0.0_f64.to_bits());
        assert_eq!(summary.false_tracks.to_bits(), 0.0_f64.to_bits());
    }
}
