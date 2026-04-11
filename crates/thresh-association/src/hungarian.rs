//! Hungarian (Munkres) algorithm for optimal linear assignment.
//!
//! The public entry point [`hungarian_assignment`] is a short sequence of
//! phase helpers so its cognitive complexity stays well under 15. Each helper
//! implements one logically distinct phase of the Kuhn-Munkres iteration:
//!
//! 1. `build_square_cost` — pad a rectangular cost matrix to a square with
//!    `gate`-valued dummy entries.
//! 2. `reduce_cost_matrix` — classical row and column reduction.
//! 3. `greedy_zero_assignment` — greedy initial matching on zero entries.
//! 4. `mark_cover` — marking pass that yields a minimum vertex cover of the
//!    current zero graph.
//! 5. `min_uncovered_value` — smallest uncovered entry used for the update.
//! 6. `update_labels` — subtract from uncovered, add to doubly covered.
//! 7. `extract_assignment` — drop dummy / gated matches and build the
//!    [`AssignmentResult`].
//!
//! Correctness is validated by hand-crafted regression tests covering
//! square, rectangular (more rows than cols and vice versa), gated, and
//! edge-case inputs, plus per-helper unit tests that exercise each phase
//! in isolation.

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

/// Tolerance used when comparing cost entries against zero.
const ZERO_EPS: f64 = 1e-10;

/// Solve the linear assignment problem using the Hungarian algorithm.
///
/// Finds the assignment that minimizes total cost. Handles rectangular matrices.
/// Entries with cost >= `gate` are considered infeasible.
pub fn hungarian_assignment(cost: &[Vec<f64>], gate: f64) -> AssignmentResult {
    let n_rows = cost.len();
    if n_rows == 0 {
        return empty_result();
    }
    let n_cols = cost[0].len();
    if n_cols == 0 {
        return all_rows_unassigned(n_rows);
    }

    let dim = n_rows.max(n_cols);
    let mut c = build_square_cost(cost, gate, dim, n_rows, n_cols);
    reduce_cost_matrix(&mut c, dim);

    let (mut row_assign, mut col_assign) = greedy_zero_assignment(&c, dim);
    run_munkres_loop(&mut c, &mut row_assign, &mut col_assign, dim);

    extract_assignment(cost, gate, &row_assign, n_rows, n_cols)
}

// ---------------------------------------------------------------------------
// Phase helpers
// ---------------------------------------------------------------------------

fn empty_result() -> AssignmentResult {
    AssignmentResult {
        matches: vec![],
        unassigned_rows: vec![],
        unassigned_cols: vec![],
        total_cost: 0.0,
    }
}

fn all_rows_unassigned(n_rows: usize) -> AssignmentResult {
    AssignmentResult {
        matches: vec![],
        unassigned_rows: (0..n_rows).collect(),
        unassigned_cols: vec![],
        total_cost: 0.0,
    }
}

/// Pad `cost` to a `dim x dim` square, filling dummy entries with `gate`.
fn build_square_cost(
    cost: &[Vec<f64>],
    gate: f64,
    dim: usize,
    n_rows: usize,
    n_cols: usize,
) -> Vec<Vec<f64>> {
    let mut c = vec![vec![0.0f64; dim]; dim];
    for i in 0..dim {
        for j in 0..dim {
            if i < n_rows && j < n_cols {
                c[i][j] = cost[i][j];
            } else {
                c[i][j] = gate;
            }
        }
    }
    c
}

/// Subtract the row minimum from each row, then the column minimum from each
/// column. Non-finite minima (all-infinity rows/columns) are left untouched.
fn reduce_cost_matrix(c: &mut [Vec<f64>], dim: usize) {
    for row in c.iter_mut() {
        let min = row.iter().copied().fold(f64::INFINITY, f64::min);
        if min.is_finite() {
            for v in row.iter_mut() {
                *v -= min;
            }
        }
    }
    for j in 0..dim {
        let min = (0..dim).map(|i| c[i][j]).fold(f64::INFINITY, f64::min);
        if min.is_finite() {
            for row in c.iter_mut() {
                row[j] -= min;
            }
        }
    }
}

/// Greedy initial matching: claim zero entries in row-major order whenever
/// both endpoints are still unmatched.
fn greedy_zero_assignment(c: &[Vec<f64>], dim: usize) -> (Vec<Option<usize>>, Vec<Option<usize>>) {
    let mut row_assign = vec![None::<usize>; dim];
    let mut col_assign = vec![None::<usize>; dim];
    for i in 0..dim {
        for j in 0..dim {
            if c[i][j].abs() < ZERO_EPS && row_assign[i].is_none() && col_assign[j].is_none() {
                row_assign[i] = Some(j);
                col_assign[j] = Some(i);
            }
        }
    }
    (row_assign, col_assign)
}

/// Run the main Munkres improvement loop until every row is assigned or no
/// further progress is possible.
fn run_munkres_loop(
    c: &mut [Vec<f64>],
    row_assign: &mut [Option<usize>],
    col_assign: &mut [Option<usize>],
    dim: usize,
) {
    loop {
        let n_assigned = row_assign.iter().filter(|a| a.is_some()).count();
        if n_assigned == dim {
            return;
        }

        let (row_covered, col_covered) = mark_cover(c, row_assign, col_assign, dim);
        let covered_count = count_true(&row_covered) + count_true(&col_covered);
        if covered_count >= dim {
            return;
        }

        let min_val = min_uncovered_value(c, &row_covered, &col_covered, dim);
        if !min_val.is_finite() || min_val.abs() < 1e-15 {
            return;
        }

        update_labels(c, &row_covered, &col_covered, dim, min_val);
        let (new_rows, new_cols) = greedy_zero_assignment(c, dim);
        row_assign.copy_from_slice(&new_rows);
        col_assign.copy_from_slice(&new_cols);
    }
}

fn count_true(flags: &[bool]) -> usize {
    flags.iter().filter(|&&f| f).count()
}

/// Compute the minimum vertex cover of the zero graph via the classical
/// marking procedure. Returns `(row_covered, col_covered)` boolean vectors.
fn mark_cover(
    c: &[Vec<f64>],
    row_assign: &[Option<usize>],
    col_assign: &[Option<usize>],
    dim: usize,
) -> (Vec<bool>, Vec<bool>) {
    let mut unmarked_rows: Vec<usize> = (0..dim).filter(|&i| row_assign[i].is_none()).collect();
    let mut row_marked = vec![false; dim];
    let mut col_marked = vec![false; dim];
    for &r in &unmarked_rows {
        row_marked[r] = true;
    }

    let mut changed = true;
    while changed {
        changed = false;
        changed |= mark_cols_with_zeros_in_marked_rows(c, &unmarked_rows, &mut col_marked, dim);
        changed |= mark_rows_assigned_to_marked_cols(
            col_assign,
            &col_marked,
            &mut row_marked,
            &mut unmarked_rows,
            dim,
        );
    }

    let row_covered: Vec<bool> = row_marked.iter().map(|&m| !m).collect();
    let col_covered: Vec<bool> = col_marked;
    (row_covered, col_covered)
}

fn mark_cols_with_zeros_in_marked_rows(
    c: &[Vec<f64>],
    unmarked_rows: &[usize],
    col_marked: &mut [bool],
    dim: usize,
) -> bool {
    let mut changed = false;
    for &r in unmarked_rows {
        for j in 0..dim {
            if !col_marked[j] && c[r][j].abs() < ZERO_EPS {
                col_marked[j] = true;
                changed = true;
            }
        }
    }
    changed
}

fn mark_rows_assigned_to_marked_cols(
    col_assign: &[Option<usize>],
    col_marked: &[bool],
    row_marked: &mut [bool],
    unmarked_rows: &mut Vec<usize>,
    dim: usize,
) -> bool {
    let mut changed = false;
    for j in 0..dim {
        if let Some(r) = col_assign[j]
            && col_marked[j]
            && !row_marked[r]
        {
            row_marked[r] = true;
            unmarked_rows.push(r);
            changed = true;
        }
    }
    changed
}

/// Smallest cost entry that lies in an uncovered row and uncovered column.
fn min_uncovered_value(
    c: &[Vec<f64>],
    row_covered: &[bool],
    col_covered: &[bool],
    dim: usize,
) -> f64 {
    let mut min_val = f64::INFINITY;
    for i in 0..dim {
        if row_covered[i] {
            continue;
        }
        for j in 0..dim {
            if !col_covered[j] && c[i][j] < min_val {
                min_val = c[i][j];
            }
        }
    }
    min_val
}

/// Update potentials: subtract `min_val` from every uncovered entry and add
/// `min_val` to every doubly-covered entry.
fn update_labels(
    c: &mut [Vec<f64>],
    row_covered: &[bool],
    col_covered: &[bool],
    dim: usize,
    min_val: f64,
) {
    for i in 0..dim {
        for j in 0..dim {
            if !row_covered[i] && !col_covered[j] {
                c[i][j] -= min_val;
            } else if row_covered[i] && col_covered[j] {
                c[i][j] += min_val;
            }
        }
    }
}

/// Build the final [`AssignmentResult`], discarding assignments that involve
/// dummy rows/cols or whose original cost is `>= gate`.
fn extract_assignment(
    cost: &[Vec<f64>],
    gate: f64,
    row_assign: &[Option<usize>],
    n_rows: usize,
    n_cols: usize,
) -> AssignmentResult {
    let mut matches = Vec::new();
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

    // ---- Phase-helper unit tests -----------------------------------------

    #[test]
    fn reduce_cost_matrix_subtracts_row_and_column_minima() {
        let mut c = vec![
            vec![4.0, 1.0, 3.0],
            vec![2.0, 0.0, 5.0],
            vec![3.0, 2.0, 2.0],
        ];
        reduce_cost_matrix(&mut c, 3);
        // Row mins: 1, 0, 2 → after row reduction:
        //   [3, 0, 2]
        //   [2, 0, 5]
        //   [1, 0, 0]
        // Col mins: 1, 0, 0 → after col reduction:
        //   [2, 0, 2]
        //   [1, 0, 5]
        //   [0, 0, 0]
        assert_eq!(c[0], vec![2.0, 0.0, 2.0]);
        assert_eq!(c[1], vec![1.0, 0.0, 5.0]);
        assert_eq!(c[2], vec![0.0, 0.0, 0.0]);
    }

    #[test]
    fn greedy_zero_assignment_matches_independent_zeros() {
        let c = vec![
            vec![0.0, 1.0, 2.0],
            vec![1.0, 0.0, 2.0],
            vec![2.0, 2.0, 0.0],
        ];
        let (row_assign, col_assign) = greedy_zero_assignment(&c, 3);
        assert_eq!(row_assign, vec![Some(0), Some(1), Some(2)]);
        assert_eq!(col_assign, vec![Some(0), Some(1), Some(2)]);
    }

    #[test]
    fn mark_cover_identifies_minimum_vertex_cover() {
        // All zeros in row 0; greedy claims (0,0). Rows 1 and 2 have no zeros.
        let c = vec![
            vec![0.0, 0.0, 0.0],
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
        ];
        let (row_assign, col_assign) = greedy_zero_assignment(&c, 3);
        let (row_cov, col_cov) = mark_cover(&c, &row_assign, &col_assign, 3);
        // Unassigned rows 1 and 2 are marked → uncovered. Assigned row 0 is
        // covered. No columns are marked (no zeros in marked rows), so all
        // columns stay uncovered.
        assert!(row_cov[0]);
        assert!(!row_cov[1]);
        assert!(!row_cov[2]);
        assert!(!col_cov.iter().any(|&c| c));
    }

    #[test]
    fn min_uncovered_value_skips_covered_entries() {
        let c = vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
            vec![7.0, 8.0, 9.0],
        ];
        let row_cov = vec![true, false, false];
        let col_cov = vec![false, true, false];
        // Uncovered entries: (1,0)=4, (1,2)=6, (2,0)=7, (2,2)=9 → min = 4
        let m = min_uncovered_value(&c, &row_cov, &col_cov, 3);
        assert_eq!(m, 4.0);
    }

    #[test]
    fn update_labels_adjusts_uncovered_and_double_covered() {
        let mut c = vec![vec![10.0, 10.0], vec![10.0, 10.0]];
        let row_cov = vec![true, false];
        let col_cov = vec![true, false];
        update_labels(&mut c, &row_cov, &col_cov, 2, 2.0);
        // (0,0) doubly covered → +2; (0,1) row covered only → unchanged;
        // (1,0) col covered only → unchanged; (1,1) uncovered → -2.
        assert_eq!(c[0][0], 12.0);
        assert_eq!(c[0][1], 10.0);
        assert_eq!(c[1][0], 10.0);
        assert_eq!(c[1][1], 8.0);
    }
}
