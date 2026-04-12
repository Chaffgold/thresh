//! HOTA (Higher Order Tracking Accuracy) metric.
//!
//! HOTA = sqrt(DetA * AssA), integrated over IoU thresholds.

use crate::matching::{FrameData, match_frame};

/// Compute HOTA with DetA and AssA at a single distance threshold.
pub fn compute_hota_at_threshold(frames: &[FrameData], distance_threshold: f64) -> (f64, f64, f64) {
    let mut total_tp = 0usize;
    let mut total_fp = 0usize;
    let mut total_fn = 0usize;

    // For AssA: track how many times each (gt_id, track_id) pair matches
    let mut pair_counts: std::collections::HashMap<(u64, u64), usize> =
        std::collections::HashMap::new();
    let mut gt_counts: std::collections::HashMap<u64, usize> = std::collections::HashMap::new();
    let mut track_counts: std::collections::HashMap<u64, usize> = std::collections::HashMap::new();

    for frame in frames {
        let fm = match_frame(frame, distance_threshold);
        total_tp += fm.matches.len();
        total_fp += fm.false_positives.len();
        total_fn += fm.false_negatives.len();

        for &(gt_id, track_id, _) in &fm.matches {
            *pair_counts.entry((gt_id, track_id)).or_insert(0) += 1;
            *gt_counts.entry(gt_id).or_insert(0) += 1;
            *track_counts.entry(track_id).or_insert(0) += 1;
        }
    }

    // DetA = TP / (TP + FP + FN)  [with 0.5 weighting to avoid double-counting, simplified]
    let det_a = if total_tp + total_fp + total_fn > 0 {
        total_tp as f64 / (total_tp + total_fp + total_fn) as f64
    } else {
        0.0
    };

    // AssA = average over TPs of |TPA(c)| / (|TPA(c)| + |FPA(c)| + |FNA(c)|)
    let mut ass_a_sum = 0.0;
    for (&(gt_id, track_id), &count) in &pair_counts {
        let gt_total = gt_counts.get(&gt_id).copied().unwrap_or(0);
        let tr_total = track_counts.get(&track_id).copied().unwrap_or(0);
        // TPA = count, FPA = tr_total - count, FNA = gt_total - count
        let denom = gt_total + tr_total - count;
        if denom > 0 {
            let a = count as f64 / denom as f64;
            ass_a_sum += a * count as f64; // weighted by number of TPs in this pair
        }
    }
    let ass_a = if total_tp > 0 {
        ass_a_sum / total_tp as f64
    } else {
        0.0
    };

    let hota = (det_a * ass_a).sqrt();
    (hota, det_a, ass_a)
}

/// Compute HOTA integrated over distance thresholds from 0.5 to 9.5 (step 0.5).
/// Adapted for 3D tracking: uses position error thresholds in meters.
pub fn compute_hota(frames: &[FrameData], thresholds: &[f64]) -> (f64, Vec<(f64, f64, f64)>) {
    let mut per_threshold = Vec::new();
    let mut hota_sum = 0.0;

    for &thresh in thresholds {
        let (hota, det_a, ass_a) = compute_hota_at_threshold(frames, thresh);
        per_threshold.push((hota, det_a, ass_a));
        hota_sum += hota;
    }

    let avg_hota = if thresholds.is_empty() {
        0.0
    } else {
        hota_sum / thresholds.len() as f64
    };

    (avg_hota, per_threshold)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn perfect_frames() -> Vec<FrameData> {
        (0..10)
            .map(|t| FrameData {
                gt: vec![(1, [t as f64 * 10.0, 0.0, 0.0])],
                tracks: vec![(101, [t as f64 * 10.0, 0.0, 0.0])],
            })
            .collect()
    }

    #[test]
    fn perfect_hota_1() {
        let frames = perfect_frames();
        let (hota, det_a, ass_a) = compute_hota_at_threshold(&frames, 1.0);
        assert!((hota - 1.0).abs() < 1e-10, "HOTA = {hota}");
        assert!((det_a - 1.0).abs() < 1e-10, "DetA = {det_a}");
        assert!((ass_a - 1.0).abs() < 1e-10, "AssA = {ass_a}");
    }

    #[test]
    fn hota_decomposition() {
        // High DetA (all detections match) but low AssA (ID switches every frame)
        let frames: Vec<FrameData> = (0..10)
            .map(|t| FrameData {
                gt: vec![(1, [t as f64 * 10.0, 0.0, 0.0])],
                tracks: vec![((100 + t) as u64, [t as f64 * 10.0, 0.0, 0.0])],
            })
            .collect();

        let (hota, det_a, ass_a) = compute_hota_at_threshold(&frames, 1.0);
        // DetA should be 1.0 (all matched)
        assert!((det_a - 1.0).abs() < 1e-10, "DetA = {det_a}");
        // AssA should be low (each pair only matches once)
        assert!(ass_a < 0.3, "AssA should be low: {ass_a}");
        // HOTA = sqrt(1.0 * low) should be moderate
        assert!(hota < 1.0);
        assert!(hota > 0.0);
    }
}
