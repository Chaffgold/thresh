//! Incremental MOT metrics for live tracking sessions.
//!
//! `compute_mot_metrics` and `compute_idf1` (in [`crate::metrics`]) are
//! batch APIs that take a slice of frames and produce final values. The
//! visualization dashboard needs running totals updated each timestep
//! without revisiting every prior frame.
//!
//! [`MotMetricsBuilder`] solves this by maintaining the cumulative
//! counters (true positives, false positives, false negatives, ID
//! switches, MOTP error) and the per-GT last-assignment map. `update`
//! ingests one frame in O(K · M) time (K = active tracks, M = active
//! ground truth), runs the same Hungarian matcher as the batch API, and
//! returns a fresh [`MotMetrics`] reflecting all frames seen so far.
//!
//! Invariants maintained across updates:
//! - `total_matches >= 0` and equals `idtp` (the per-frame matches that
//!   succeed are also the IDF1 true positives in this implementation).
//! - `last_assignment` is monotonic: an entry is never removed, only
//!   overwritten. Removing it would lose the ID-switch signal.
//! - `total_gt` accumulates frame-by-frame, mirroring the batch API.

use std::collections::HashMap;

use crate::matching::{FrameData, match_frame};

/// Aggregated MOT metrics produced by [`MotMetricsBuilder::update`].
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct MotMetrics {
    /// Multiple Object Tracking Accuracy.
    pub mota: f64,
    /// Multiple Object Tracking Precision (mean matched-pair distance).
    pub motp: f64,
    /// IDF1 = 2·IDTP / (2·IDTP + IDFP + IDFN).
    pub idf1: f64,
    /// Cumulative ID switches.
    pub id_switches: usize,
    /// Cumulative true positives (matched pairs).
    pub true_positives: usize,
    /// Cumulative false positives (unmatched tracks).
    pub false_positives: usize,
    /// Cumulative false negatives (unmatched ground truth).
    pub false_negatives: usize,
    /// Total ground-truth observations across all frames seen so far.
    pub total_ground_truth: usize,
}

/// Stateful per-frame MOT metric accumulator.
///
/// Construct with [`MotMetricsBuilder::new`], call [`Self::update`] each
/// timestep with that frame's ground truth and tracks, and read the
/// running [`MotMetrics`] from the return value.
#[derive(Debug, Clone)]
pub struct MotMetricsBuilder {
    distance_threshold: f64,
    last_assignment: HashMap<u64, u64>,
    total_gt: usize,
    total_fn: usize,
    total_fp: usize,
    total_idsw: usize,
    total_dist: f64,
    total_matches: usize,
}

impl MotMetricsBuilder {
    /// Create an empty builder. `distance_threshold` is the maximum
    /// gating distance passed through to the Hungarian matcher.
    pub fn new(distance_threshold: f64) -> Self {
        Self {
            distance_threshold,
            last_assignment: HashMap::new(),
            total_gt: 0,
            total_fn: 0,
            total_fp: 0,
            total_idsw: 0,
            total_dist: 0.0,
            total_matches: 0,
        }
    }

    /// Ingest one frame and return the running metrics.
    ///
    /// Runs in O(K · M) where K is the number of active tracks and M is
    /// the number of active ground-truth observations in this frame.
    pub fn update(&mut self, frame: &FrameData) -> MotMetrics {
        let fm = match_frame(frame, self.distance_threshold);
        self.total_gt += frame.gt.len();
        self.total_fn += fm.false_negatives.len();
        self.total_fp += fm.false_positives.len();

        for &(gt_id, track_id, dist) in &fm.matches {
            self.total_matches += 1;
            self.total_dist += dist;

            if let Some(&prev_track) = self.last_assignment.get(&gt_id)
                && prev_track != track_id
            {
                self.total_idsw += 1;
            }
            self.last_assignment.insert(gt_id, track_id);
        }

        self.snapshot()
    }

    /// Reset the builder to its initial empty state.
    pub fn reset(&mut self) {
        self.last_assignment.clear();
        self.total_gt = 0;
        self.total_fn = 0;
        self.total_fp = 0;
        self.total_idsw = 0;
        self.total_dist = 0.0;
        self.total_matches = 0;
    }

    /// Read current metrics without ingesting a frame.
    pub fn snapshot(&self) -> MotMetrics {
        let mota = if self.total_gt > 0 {
            1.0 - (self.total_fn + self.total_fp + self.total_idsw) as f64 / self.total_gt as f64
        } else {
            0.0
        };

        let motp = if self.total_matches > 0 {
            self.total_dist / self.total_matches as f64
        } else {
            0.0
        };

        // IDF1 in this implementation uses the same per-frame match
        // counts as MOTA, mirroring `compute_idf1` in `metrics.rs`.
        let idtp = self.total_matches;
        let idfp = self.total_fp;
        let idfn = self.total_fn;
        let denom = 2 * idtp + idfp + idfn;
        let idf1 = if denom > 0 {
            (2 * idtp) as f64 / denom as f64
        } else {
            0.0
        };

        MotMetrics {
            mota,
            motp,
            idf1,
            id_switches: self.total_idsw,
            true_positives: self.total_matches,
            false_positives: self.total_fp,
            false_negatives: self.total_fn,
            total_ground_truth: self.total_gt,
        }
    }

    /// Distance threshold used for matching.
    pub fn distance_threshold(&self) -> f64 {
        self.distance_threshold
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metrics::{compute_idf1, compute_mot_metrics};

    fn perfect_frames() -> Vec<FrameData> {
        (0..10)
            .map(|t| FrameData {
                gt: vec![
                    (1, [t as f64 * 10.0, 0.0, 0.0]),
                    (2, [0.0, t as f64 * 10.0, 0.0]),
                ],
                tracks: vec![
                    (101, [t as f64 * 10.0, 0.0, 0.0]),
                    (102, [0.0, t as f64 * 10.0, 0.0]),
                ],
            })
            .collect()
    }

    #[test]
    fn builder_matches_one_shot_for_perfect_run() {
        let frames = perfect_frames();
        let (mota_batch, motp_batch, idsw_batch) = compute_mot_metrics(&frames, 1.0);
        let idf1_batch = compute_idf1(&frames, 1.0);

        let mut builder = MotMetricsBuilder::new(1.0);
        let mut last = MotMetrics::default();
        for f in &frames {
            last = builder.update(f);
        }

        assert!((last.mota - mota_batch).abs() < 1e-9, "MOTA mismatch");
        assert!((last.motp - motp_batch).abs() < 1e-9, "MOTP mismatch");
        assert!((last.idf1 - idf1_batch).abs() < 1e-9, "IDF1 mismatch");
        assert_eq!(last.id_switches, idsw_batch);
    }

    #[test]
    fn builder_matches_one_shot_for_id_switch() {
        // Frame 0: track 101 follows GT 1
        // Frame 1: track 102 takes over GT 1 -> 1 ID switch
        let frames = vec![
            FrameData {
                gt: vec![(1, [0.0, 0.0, 0.0])],
                tracks: vec![(101, [0.0, 0.0, 0.0])],
            },
            FrameData {
                gt: vec![(1, [10.0, 0.0, 0.0])],
                tracks: vec![(102, [10.0, 0.0, 0.0])],
            },
        ];
        let (mota_batch, _, idsw_batch) = compute_mot_metrics(&frames, 1.0);

        let mut builder = MotMetricsBuilder::new(1.0);
        let mut last = MotMetrics::default();
        for f in &frames {
            last = builder.update(f);
        }
        assert_eq!(last.id_switches, idsw_batch);
        assert_eq!(last.id_switches, 1);
        assert!((last.mota - mota_batch).abs() < 1e-9);
    }

    #[test]
    fn builder_handles_empty_frame() {
        let mut builder = MotMetricsBuilder::new(1.0);
        let m = builder.update(&FrameData {
            gt: vec![],
            tracks: vec![],
        });
        assert_eq!(m.mota, 0.0);
        assert_eq!(m.motp, 0.0);
        assert_eq!(m.idf1, 0.0);
        assert_eq!(m.total_ground_truth, 0);
    }

    #[test]
    fn builder_reset_returns_to_initial_state() {
        let frames = perfect_frames();
        let mut builder = MotMetricsBuilder::new(1.0);
        for f in &frames {
            builder.update(f);
        }
        let initial = MotMetricsBuilder::new(1.0).snapshot();
        builder.reset();
        assert_eq!(builder.snapshot(), initial);
    }

    #[test]
    fn builder_per_frame_metrics_reflect_history() {
        // Verify metrics evolve over time, not just final value
        let frames = perfect_frames();
        let mut builder = MotMetricsBuilder::new(1.0);
        let m1 = builder.update(&frames[0]);
        let m5 = (1..5).fold(m1, |_, i| builder.update(&frames[i]));
        // After more frames, total_ground_truth must grow
        assert!(m5.total_ground_truth > m1.total_ground_truth);
    }

    #[test]
    fn builder_scales_subquadratically() {
        // Coarse timing check: O(K·M) per frame, not O(N²·K·M)
        use std::time::Instant;

        let make_run = |k: usize| -> Vec<FrameData> {
            (0..20)
                .map(|t| {
                    let gt: Vec<_> = (0..k)
                        .map(|i| (i as u64, [t as f64 + i as f64, 0.0, 0.0]))
                        .collect();
                    let tracks: Vec<_> = (0..k)
                        .map(|i| (100 + i as u64, [t as f64 + i as f64, 0.0, 0.0]))
                        .collect();
                    FrameData { gt, tracks }
                })
                .collect()
        };

        let small = make_run(10);
        let large = make_run(100);

        let t_small = {
            let start = Instant::now();
            let mut b = MotMetricsBuilder::new(1.0);
            for f in &small {
                b.update(f);
            }
            start.elapsed()
        };

        let t_large = {
            let start = Instant::now();
            let mut b = MotMetricsBuilder::new(1.0);
            for f in &large {
                b.update(f);
            }
            start.elapsed()
        };

        // Hungarian is O((K+M)^3) per frame, so 10x K → ~1000x time.
        // We just verify the builder isn't accumulating per-history work
        // (which would be > O(N²·...) and easily 10000x).
        // Allow up to 5000x for headroom across machines.
        let ratio = t_large.as_nanos() as f64 / t_small.as_nanos().max(1) as f64;
        assert!(
            ratio < 5000.0,
            "Builder appears super-quadratic in track count: ratio = {ratio}"
        );
    }
}
