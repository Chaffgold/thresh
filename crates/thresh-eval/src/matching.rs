//! Ground-truth to track matching via Hungarian assignment.

use thresh_association::hungarian::hungarian_assignment;

/// A frame's ground truth and track positions.
pub struct FrameData {
    /// Ground truth positions [(id, [x,y,z])].
    pub gt: Vec<(u64, [f64; 3])>,
    /// Track positions [(id, [x,y,z])].
    pub tracks: Vec<(u64, [f64; 3])>,
}

/// Match result for a single frame.
#[derive(Debug, Clone)]
pub struct FrameMatch {
    /// (gt_id, track_id, distance) pairs.
    pub matches: Vec<(u64, u64, f64)>,
    /// Unmatched ground truth IDs (false negatives).
    pub false_negatives: Vec<u64>,
    /// Unmatched track IDs (false positives).
    pub false_positives: Vec<u64>,
}

/// Match ground truths to tracks using Euclidean distance + Hungarian.
pub fn match_frame(frame: &FrameData, distance_threshold: f64) -> FrameMatch {
    if frame.gt.is_empty() || frame.tracks.is_empty() {
        return FrameMatch {
            matches: vec![],
            false_negatives: frame.gt.iter().map(|(id, _)| *id).collect(),
            false_positives: frame.tracks.iter().map(|(id, _)| *id).collect(),
        };
    }

    let mut cost = vec![vec![0.0; frame.tracks.len()]; frame.gt.len()];
    for (i, (_, gt_pos)) in frame.gt.iter().enumerate() {
        for (j, (_, tr_pos)) in frame.tracks.iter().enumerate() {
            let d = ((gt_pos[0] - tr_pos[0]).powi(2)
                + (gt_pos[1] - tr_pos[1]).powi(2)
                + (gt_pos[2] - tr_pos[2]).powi(2))
            .sqrt();
            cost[i][j] = d;
        }
    }

    let result = hungarian_assignment(&cost, distance_threshold);

    let matches: Vec<(u64, u64, f64)> = result
        .matches
        .iter()
        .map(|&(gi, ti)| (frame.gt[gi].0, frame.tracks[ti].0, cost[gi][ti]))
        .collect();

    let fn_ids: Vec<u64> = result
        .unassigned_rows
        .iter()
        .map(|&i| frame.gt[i].0)
        .collect();

    let fp_ids: Vec<u64> = result
        .unassigned_cols
        .iter()
        .map(|&j| frame.tracks[j].0)
        .collect();

    FrameMatch {
        matches,
        false_negatives: fn_ids,
        false_positives: fp_ids,
    }
}
