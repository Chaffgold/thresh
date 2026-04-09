//! Hungarian (Munkres) algorithm for optimal linear assignment.

/// Result of running the Hungarian algorithm.
#[derive(Debug, Clone)]
pub struct AssignmentResult {
    /// (row, col) pairs of matched assignments.
    pub matches: Vec<(usize, usize)>,
    /// Row indices with no assignment (unassigned tracks).
    pub unassigned_rows: Vec<usize>,
    /// Column indices with no assignment (unassigned detections).
    pub unassigned_cols: Vec<usize>,
    /// Total cost of the assignment.
    pub total_cost: f64,
}

/// Solve the linear assignment problem using the Hungarian algorithm.
///
/// Finds the assignment that minimizes total cost. Handles rectangular matrices.
/// Entries with cost >= `gate` are considered infeasible.
pub fn hungarian_assignment(cost: &[Vec<f64>], gate: f64) -> AssignmentResult {
    let n_rows = cost.len();
    if n_rows == 0 {
        return AssignmentResult {
            matches: vec![],
            unassigned_rows: vec![],
            unassigned_cols: vec![],
            total_cost: 0.0,
        };
    }
    let n_cols = cost[0].len();
    if n_cols == 0 {
        return AssignmentResult {
            matches: vec![],
            unassigned_rows: (0..n_rows).collect(),
            unassigned_cols: vec![],
            total_cost: 0.0,
        };
    }

    // Pad to square if needed
    let dim = n_rows.max(n_cols);
    let mut c = vec![vec![0.0f64; dim]; dim];
    for i in 0..dim {
        for j in 0..dim {
            if i < n_rows && j < n_cols {
                c[i][j] = cost[i][j];
            } else {
                c[i][j] = gate; // dummy entries at gate cost
            }
        }
    }

    // Step 1: Row reduction
    for row in c.iter_mut() {
        let min = row.iter().cloned().fold(f64::INFINITY, f64::min);
        if min.is_finite() {
            for v in row.iter_mut() {
                *v -= min;
            }
        }
    }

    // Step 2: Column reduction
    for j in 0..dim {
        let min = (0..dim).map(|i| c[i][j]).fold(f64::INFINITY, f64::min);
        if min.is_finite() {
            for row in c.iter_mut() {
                row[j] -= min;
            }
        }
    }

    // Kuhn-Munkres assignment
    let mut row_assign = vec![None::<usize>; dim];
    let mut col_assign = vec![None::<usize>; dim];

    // Greedy initial assignment on zeros
    for i in 0..dim {
        for j in 0..dim {
            if c[i][j].abs() < 1e-10 && row_assign[i].is_none() && col_assign[j].is_none() {
                row_assign[i] = Some(j);
                col_assign[j] = Some(i);
            }
        }
    }

    loop {
        // Find uncovered rows/cols
        let mut row_covered = vec![false; dim];
        let mut col_covered = vec![false; dim];

        // Mark columns with assignments
        for (i, assign) in row_assign.iter().enumerate() {
            if assign.is_some() {
                row_covered[i] = false;
            }
        }

        // Augmenting path search
        let n_assigned: usize = row_assign.iter().filter(|a| a.is_some()).count();
        if n_assigned == dim {
            break;
        }

        // Mark rows without assignments
        let mut unmarked_rows: Vec<usize> = (0..dim).filter(|&i| row_assign[i].is_none()).collect();

        let mut row_marked = vec![false; dim];
        let mut col_marked = vec![false; dim];
        for &r in &unmarked_rows {
            row_marked[r] = true;
        }

        let mut changed = true;
        while changed {
            changed = false;
            // Mark columns with zeros in marked rows
            for &r in &unmarked_rows {
                for j in 0..dim {
                    if !col_marked[j] && c[r][j].abs() < 1e-10 {
                        col_marked[j] = true;
                        changed = true;
                    }
                }
            }
            // Mark rows assigned to marked columns
            let mut new_rows = vec![];
            for j in 0..dim {
                if let Some(r) = col_assign[j]
                    && col_marked[j]
                    && !row_marked[r]
                {
                    row_marked[r] = true;
                    new_rows.push(r);
                    changed = true;
                }
            }
            unmarked_rows.extend(new_rows);
        }

        // Cover: rows NOT marked, columns marked
        for (i, &m) in row_marked.iter().enumerate() {
            row_covered[i] = !m;
        }
        for (j, &m) in col_marked.iter().enumerate() {
            col_covered[j] = m;
        }

        // Check if we can find augmenting path
        let covered_count: usize =
            row_covered.iter().filter(|&&c| c).count() + col_covered.iter().filter(|&&c| c).count();
        if covered_count >= dim {
            break;
        }

        // Find minimum uncovered value
        let mut min_val = f64::INFINITY;
        for i in 0..dim {
            if !row_covered[i] {
                for j in 0..dim {
                    if !col_covered[j] && c[i][j] < min_val {
                        min_val = c[i][j];
                    }
                }
            }
        }

        if !min_val.is_finite() || min_val.abs() < 1e-15 {
            break;
        }

        // Subtract from uncovered, add to doubly covered
        for i in 0..dim {
            for j in 0..dim {
                if !row_covered[i] && !col_covered[j] {
                    c[i][j] -= min_val;
                }
                if row_covered[i] && col_covered[j] {
                    c[i][j] += min_val;
                }
            }
        }

        // Re-do greedy assignment on zeros
        row_assign = vec![None; dim];
        col_assign = vec![None; dim];
        for i in 0..dim {
            for j in 0..dim {
                if c[i][j].abs() < 1e-10 && row_assign[i].is_none() && col_assign[j].is_none() {
                    row_assign[i] = Some(j);
                    col_assign[j] = Some(i);
                }
            }
        }
    }

    // Extract real assignments, filtering dummy and gated
    let mut matches = vec![];
    let mut matched_rows = vec![false; n_rows];
    let mut matched_cols = vec![false; n_cols];
    let mut total_cost = 0.0;

    for i in 0..n_rows {
        if let Some(j) = row_assign[i]
            && j < n_cols
            && cost[i][j] < gate
        {
            matches.push((i, j));
            matched_rows[i] = true;
            matched_cols[j] = true;
            total_cost += cost[i][j];
        }
    }

    let unassigned_rows: Vec<usize> = (0..n_rows).filter(|&i| !matched_rows[i]).collect();
    let unassigned_cols: Vec<usize> = (0..n_cols).filter(|&j| !matched_cols[j]).collect();

    AssignmentResult {
        matches,
        unassigned_rows,
        unassigned_cols,
        total_cost,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn known_5x5_assignment() {
        // Classic example: should find minimum-cost perfect matching
        let cost = vec![
            vec![10.0, 5.0, 13.0, 15.0, 16.0],
            vec![3.0, 9.0, 18.0, 13.0, 6.0],
            vec![10.0, 7.0, 2.0, 4.0, 2.0],
            vec![5.0, 11.0, 9.0, 4.0, 12.0],
            vec![7.0, 5.0, 11.0, 7.0, 3.0],
        ];
        let result = hungarian_assignment(&cost, f64::INFINITY);
        assert_eq!(result.matches.len(), 5);
        assert!(result.unassigned_rows.is_empty());
        assert!(result.unassigned_cols.is_empty());
        // Optimal cost = 5 + 3 + 2 + 4 + 3 = 17
        assert!(
            (result.total_cost - 17.0).abs() < 1e-10,
            "Expected cost 17, got {}",
            result.total_cost
        );
    }

    #[test]
    fn rectangular_more_detections() {
        // 2 tracks, 4 detections
        let cost = vec![vec![1.0, 10.0, 20.0, 30.0], vec![30.0, 20.0, 2.0, 10.0]];
        let result = hungarian_assignment(&cost, f64::INFINITY);
        assert_eq!(result.matches.len(), 2);
        assert!(result.unassigned_rows.is_empty());
        assert_eq!(result.unassigned_cols.len(), 2);
    }

    #[test]
    fn rectangular_more_tracks() {
        // 4 tracks, 2 detections
        let cost = vec![
            vec![1.0, 10.0],
            vec![10.0, 2.0],
            vec![20.0, 30.0],
            vec![30.0, 20.0],
        ];
        let result = hungarian_assignment(&cost, f64::INFINITY);
        assert_eq!(result.matches.len(), 2);
        assert_eq!(result.unassigned_rows.len(), 2);
        assert!(result.unassigned_cols.is_empty());
    }

    #[test]
    fn gating_rejects_expensive() {
        let cost = vec![vec![1.0, 100.0], vec![100.0, 2.0]];
        let result = hungarian_assignment(&cost, 50.0);
        assert_eq!(result.matches.len(), 2);
        assert_eq!(result.total_cost, 3.0);
    }

    #[test]
    fn empty_cost_matrix() {
        let cost: Vec<Vec<f64>> = vec![];
        let result = hungarian_assignment(&cost, f64::INFINITY);
        assert!(result.matches.is_empty());
    }
}
