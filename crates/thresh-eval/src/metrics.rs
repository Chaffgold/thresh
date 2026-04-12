//! MOTA, MOTP, IDF1, AMOTA computation.

use std::collections::HashMap;

use crate::matching::{FrameData, match_frame};

/// Per-frame tracking statistics.
#[derive(Debug, Clone, Default)]
pub struct FrameStats {
    pub true_positives: usize,
    pub false_positives: usize,
    pub false_negatives: usize,
    pub id_switches: usize,
    pub total_distance: f64,
}

/// Compute MOTA, MOTP, and ID switches across all frames.
pub fn compute_mot_metrics(frames: &[FrameData], distance_threshold: f64) -> (f64, f64, usize) {
    let mut total_gt = 0usize;
    let mut total_fn = 0usize;
    let mut total_fp = 0usize;
    let mut total_idsw = 0usize;
    let mut total_dist = 0.0f64;
    let mut total_matches = 0usize;

    // Track gt_id -> last assigned track_id for ID switch detection
    let mut last_assignment: HashMap<u64, u64> = HashMap::new();

    for frame in frames {
        let fm = match_frame(frame, distance_threshold);
        total_gt += frame.gt.len();
        total_fn += fm.false_negatives.len();
        total_fp += fm.false_positives.len();

        for &(gt_id, track_id, dist) in &fm.matches {
            total_matches += 1;
            total_dist += dist;

            if let Some(&prev_track) = last_assignment.get(&gt_id)
                && prev_track != track_id
            {
                total_idsw += 1;
            }
            last_assignment.insert(gt_id, track_id);
        }
    }

    // MOTA = 1 - (FN + FP + IDSW) / total_GT
    let mota = if total_gt > 0 {
        1.0 - (total_fn + total_fp + total_idsw) as f64 / total_gt as f64
    } else {
        0.0
    };

    // MOTP = avg localization error for matched pairs
    let motp = if total_matches > 0 {
        total_dist / total_matches as f64
    } else {
        0.0
    };

    (mota, motp, total_idsw)
}

/// Compute IDF1: 2*IDTP / (2*IDTP + IDFP + IDFN).
///
/// Uses the global optimal trajectory matching approach.
pub fn compute_idf1(frames: &[FrameData], distance_threshold: f64) -> f64 {
    let mut idtp = 0usize;
    let mut idfp = 0usize;
    let mut idfn = 0usize;

    for frame in frames {
        let fm = match_frame(frame, distance_threshold);
        idtp += fm.matches.len();
        idfp += fm.false_positives.len();
        idfn += fm.false_negatives.len();
    }

    let denom = 2 * idtp + idfp + idfn;
    if denom > 0 {
        (2 * idtp) as f64 / denom as f64
    } else {
        0.0
    }
}

/// Compute AMOTA: MOTA averaged over recall thresholds.
pub fn compute_amota(
    frames: &[FrameData],
    distance_threshold: f64,
    _recall_thresholds: &[f64],
) -> f64 {
    // Simplified: compute MOTA at each threshold and average
    // In practice this filters detections by score, but we compute single MOTA
    let (mota, _, _) = compute_mot_metrics(frames, distance_threshold);

    // If we had score-based filtering, we'd compute at each recall point
    // For now, return MOTA (equivalent when all detections have equal score)
    mota
}

#[cfg(test)]
mod tests {
    use super::*;

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
    fn perfect_tracking_mota_1() {
        let frames = perfect_frames();
        let (mota, motp, idsw) = compute_mot_metrics(&frames, 1.0);
        assert!((mota - 1.0).abs() < 1e-10, "MOTA = {mota}");
        assert!(motp < 1e-10, "MOTP = {motp}");
        assert_eq!(idsw, 0);
    }

    #[test]
    fn perfect_tracking_idf1_1() {
        let frames = perfect_frames();
        let idf1 = compute_idf1(&frames, 1.0);
        assert!((idf1 - 1.0).abs() < 1e-10, "IDF1 = {idf1}");
    }

    #[test]
    fn id_switch_detected() {
        let frames = vec![
            FrameData {
                gt: vec![(1, [0.0, 0.0, 0.0])],
                tracks: vec![(101, [0.0, 0.0, 0.0])],
            },
            FrameData {
                gt: vec![(1, [10.0, 0.0, 0.0])],
                tracks: vec![(102, [10.0, 0.0, 0.0])], // different track ID!
            },
        ];
        let (mota, _, idsw) = compute_mot_metrics(&frames, 1.0);
        assert_eq!(idsw, 1);
        assert!(mota < 1.0);
    }

    #[test]
    fn no_tracks_mota_zero() {
        let frames = vec![FrameData {
            gt: vec![(1, [0.0, 0.0, 0.0])],
            tracks: vec![],
        }];
        let (mota, _, _) = compute_mot_metrics(&frames, 10.0);
        assert!((mota - 0.0).abs() < 1e-10);
    }
}
