//! Cost matrix construction and cascaded association.

use crate::gating::{chi2_threshold, mahalanobis_squared};
use crate::hungarian::{AssignmentResult, hungarian_assignment};
use nalgebra::{DMatrix, DVector};

/// Build a fused cost matrix: C = alpha * d_motion + (1-alpha) * d_appearance.
pub fn fused_cost_matrix(
    motion_costs: &[Vec<f64>],
    appearance_costs: &[Vec<f64>],
    alpha: f64,
) -> Vec<Vec<f64>> {
    let n_rows = motion_costs.len();
    if n_rows == 0 {
        return vec![];
    }
    let n_cols = motion_costs[0].len();

    let mut fused = vec![vec![0.0; n_cols]; n_rows];
    for i in 0..n_rows {
        for j in 0..n_cols {
            fused[i][j] = alpha * motion_costs[i][j] + (1.0 - alpha) * appearance_costs[i][j];
        }
    }
    fused
}

/// Build a motion cost matrix from predicted measurements and their innovation covariances.
pub fn motion_cost_matrix(
    predicted: &[(DVector<f64>, DMatrix<f64>)],
    measurements: &[DVector<f64>],
    gate: Option<f64>,
) -> Vec<Vec<f64>> {
    let n_tracks = predicted.len();
    let n_dets = measurements.len();
    let measurement_dim = if n_dets > 0 { measurements[0].len() } else { 2 };
    let threshold = gate.unwrap_or_else(|| chi2_threshold(measurement_dim));

    let mut costs = vec![vec![threshold; n_dets]; n_tracks];

    for (i, (pred_z, s)) in predicted.iter().enumerate() {
        for (j, z) in measurements.iter().enumerate() {
            let d2 = mahalanobis_squared(z, pred_z, s);
            if d2 <= threshold {
                costs[i][j] = d2;
            }
        }
    }
    costs
}

/// Cascaded association: first match high-confidence detections, then low-confidence.
pub fn cascaded_association(
    cost_matrix: &[Vec<f64>],
    detection_scores: &[f64],
    score_threshold: f64,
    gate: f64,
) -> AssignmentResult {
    let n_rows = cost_matrix.len();
    let n_cols = if n_rows > 0 { cost_matrix[0].len() } else { 0 };

    // Split detections by confidence
    let high_conf: Vec<usize> = (0..n_cols)
        .filter(|&j| detection_scores[j] >= score_threshold)
        .collect();
    let low_conf: Vec<usize> = (0..n_cols)
        .filter(|&j| detection_scores[j] < score_threshold)
        .collect();

    // First pass: high confidence
    let high_cost: Vec<Vec<f64>> = (0..n_rows)
        .map(|i| high_conf.iter().map(|&j| cost_matrix[i][j]).collect())
        .collect();

    let first = hungarian_assignment(&high_cost, gate);

    let mut all_matches = Vec::new();
    let mut matched_rows = vec![false; n_rows];
    let mut matched_cols = vec![false; n_cols];
    let mut total_cost = 0.0;

    for &(r, hc) in &first.matches {
        let real_col = high_conf[hc];
        all_matches.push((r, real_col));
        matched_rows[r] = true;
        matched_cols[real_col] = true;
        total_cost += cost_matrix[r][real_col];
    }

    // Second pass: low confidence on unmatched tracks
    let remaining_rows: Vec<usize> = (0..n_rows).filter(|&i| !matched_rows[i]).collect();
    let low_cost: Vec<Vec<f64>> = remaining_rows
        .iter()
        .map(|&i| low_conf.iter().map(|&j| cost_matrix[i][j]).collect())
        .collect();

    let second = hungarian_assignment(&low_cost, gate);

    for &(lr, lc) in &second.matches {
        let real_row = remaining_rows[lr];
        let real_col = low_conf[lc];
        all_matches.push((real_row, real_col));
        matched_rows[real_row] = true;
        matched_cols[real_col] = true;
        total_cost += cost_matrix[real_row][real_col];
    }

    let unassigned_rows: Vec<usize> = (0..n_rows).filter(|&i| !matched_rows[i]).collect();
    let unassigned_cols: Vec<usize> = (0..n_cols).filter(|&j| !matched_cols[j]).collect();

    AssignmentResult {
        matches: all_matches,
        unassigned_rows,
        unassigned_cols,
        total_cost,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fused_cost_50_50() {
        let motion = vec![vec![2.0, 10.0], vec![10.0, 3.0]];
        let appear = vec![vec![4.0, 8.0], vec![8.0, 5.0]];
        let fused = fused_cost_matrix(&motion, &appear, 0.5);
        assert!((fused[0][0] - 3.0).abs() < 1e-10);
        assert!((fused[1][1] - 4.0).abs() < 1e-10);
    }

    #[test]
    fn cascaded_high_first() {
        let cost = vec![vec![1.0, 5.0, 2.0], vec![5.0, 1.0, 5.0]];
        let scores = vec![0.9, 0.8, 0.3]; // det 2 is low-conf

        let result = cascaded_association(&cost, &scores, 0.5, 100.0);
        assert_eq!(result.matches.len(), 2);
    }
}
