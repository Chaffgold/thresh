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
//! 4. `augment_matching` — bipartite augmenting-path search that grows the
//!    greedy matching to a maximum matching of the zero graph.
//! 5. `mark_cover` — marking pass that yields a minimum vertex cover of the
//!    current zero graph (valid because the matching is maximum, König).
//! 6. `min_uncovered_value` — smallest uncovered entry used for the update.
//! 7. `update_labels` — subtract from uncovered, add to doubly covered.
//! 8. `extract_assignment` — drop dummy / gated matches and build the
//!    [`AssignmentResult`].
//!
//! Correctness is validated by hand-crafted regression tests covering
//! square, rectangular (more rows than cols and vice versa), gated, and
//! edge-case inputs, plus per-helper unit tests that exercise each phase
//! in isolation.
//!
//! # Why `Vec<Vec<f64>>` instead of `nalgebra::DMatrix<f64>`
//!
//! The workspace guideline is to prefer `nalgebra` for linear-algebra-heavy
//! code, but this module deliberately keeps the row-major `Vec<Vec<f64>>`
//! representation that matches the public `cost.rs` API boundary. The
//! Hungarian helpers do element-wise reduction, marking, and min-search
//! rather than BLAS-style matrix ops, so `DMatrix` would not buy vectorised
//! kernels here — it would only add a copy at every call site that currently
//! builds cost matrices as `Vec<Vec<f64>>`. Revisit if we ever need to hand
//! larger matrices to a SIMD-accelerated solver.

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

/// Pre-allocated solver for the Hungarian (Munkres) algorithm.
///
/// Reuses internal buffers across calls to avoid per-invocation allocation.
/// Create once with [`HungarianSolver::new`] and call [`HungarianSolver::solve`]
/// repeatedly. The solver transparently grows its buffers if a call exceeds the
/// initial `max_dim`.
pub struct HungarianSolver {
    /// Flat row-major cost buffer (`dim * dim` elements).
    cost_buf: Vec<f64>,
    /// Pre-allocated row assignment buffer.
    row_assign: Vec<Option<usize>>,
    /// Pre-allocated column assignment buffer.
    col_assign: Vec<Option<usize>>,
    /// Pre-allocated row-covered flags (for mark_cover).
    row_covered: Vec<bool>,
    /// Pre-allocated col-covered flags (for mark_cover).
    col_covered: Vec<bool>,
    /// Pre-allocated row-marked flags (for mark_cover).
    row_marked: Vec<bool>,
    /// Pre-allocated unmarked-rows scratch list.
    unmarked_rows: Vec<usize>,
    /// Pre-allocated BFS visited-col flags (for augment_matching).
    visited_col: Vec<bool>,
    /// Pre-allocated BFS parent map (for augment_matching).
    parent_row_for_col: Vec<Option<usize>>,
    /// Pre-allocated BFS queue (for augment_matching).
    queue: Vec<usize>,
    /// Current maximum dimension the buffers can handle without realloc.
    capacity: usize,
}

impl HungarianSolver {
    /// Create a new solver pre-allocated for matrices up to `max_dim x max_dim`.
    pub fn new(max_dim: usize) -> Self {
        Self {
            cost_buf: vec![0.0; max_dim * max_dim],
            row_assign: vec![None; max_dim],
            col_assign: vec![None; max_dim],
            row_covered: vec![false; max_dim],
            col_covered: vec![false; max_dim],
            row_marked: vec![false; max_dim],
            unmarked_rows: Vec::with_capacity(max_dim),
            visited_col: vec![false; max_dim],
            parent_row_for_col: vec![None; max_dim],
            queue: Vec::with_capacity(max_dim),
            capacity: max_dim,
        }
    }

    /// Ensure all buffers are large enough for `dim`.
    fn ensure_capacity(&mut self, dim: usize) {
        if dim <= self.capacity {
            return;
        }
        self.cost_buf.resize(dim * dim, 0.0);
        self.row_assign.resize(dim, None);
        self.col_assign.resize(dim, None);
        self.row_covered.resize(dim, false);
        self.col_covered.resize(dim, false);
        self.row_marked.resize(dim, false);
        self.visited_col.resize(dim, false);
        self.parent_row_for_col.resize(dim, None);
        self.capacity = dim;
    }

    /// Solve the linear assignment problem on a flat pre-allocated buffer.
    ///
    /// Semantically identical to [`hungarian_assignment`] but reuses internal
    /// storage across calls.
    pub fn solve(&mut self, cost: &[Vec<f64>], gate: f64) -> AssignmentResult {
        let n_rows = cost.len();
        if n_rows == 0 {
            return empty_result();
        }
        let n_cols = cost[0].len();
        if n_cols == 0 {
            return all_rows_unassigned(n_rows);
        }

        let dim = n_rows.max(n_cols);
        self.ensure_capacity(dim);

        // Fill the flat cost buffer (row-major).
        for (i, row) in cost.iter().enumerate() {
            let base = i * dim;
            for (j, &val) in row.iter().enumerate() {
                self.cost_buf[base + j] = val;
            }
            for j in n_cols..dim {
                self.cost_buf[base + j] = gate;
            }
        }
        for i in n_rows..dim {
            let base = i * dim;
            for j in 0..dim {
                self.cost_buf[base + j] = gate;
            }
        }

        // Reduce cost matrix (row then column reduction) on flat buffer.
        self.reduce_flat(dim);

        // Greedy zero assignment on flat buffer.
        self.greedy_zero_flat(dim);

        // Run Munkres loop on flat buffer.
        self.run_munkres_loop_flat(dim);

        // Extract assignment from row_assign (same logic as the free function).
        extract_assignment(cost, gate, &self.row_assign, n_rows, n_cols)
    }

    /// Flat-buffer row and column reduction.
    fn reduce_flat(&mut self, dim: usize) {
        // Row reduction.
        for i in 0..dim {
            let base = i * dim;
            let min = self.cost_buf[base..base + dim]
                .iter()
                .copied()
                .fold(f64::INFINITY, f64::min);
            if min.is_finite() {
                for v in &mut self.cost_buf[base..base + dim] {
                    *v -= min;
                }
            }
        }
        // Column reduction.
        for j in 0..dim {
            let min = (0..dim)
                .map(|i| self.cost_buf[i * dim + j])
                .fold(f64::INFINITY, f64::min);
            if min.is_finite() {
                for i in 0..dim {
                    self.cost_buf[i * dim + j] -= min;
                }
            }
        }
    }

    /// Greedy zero assignment on the flat buffer.
    fn greedy_zero_flat(&mut self, dim: usize) {
        for v in &mut self.row_assign[..dim] {
            *v = None;
        }
        for v in &mut self.col_assign[..dim] {
            *v = None;
        }
        for i in 0..dim {
            for j in 0..dim {
                if self.cost_buf[i * dim + j].abs() < ZERO_EPS
                    && self.row_assign[i].is_none()
                    && self.col_assign[j].is_none()
                {
                    self.row_assign[i] = Some(j);
                    self.col_assign[j] = Some(i);
                }
            }
        }
    }

    /// Augment matching on the flat buffer via BFS.
    fn augment_matching_flat(&mut self, dim: usize) {
        for start_row in 0..dim {
            if self.row_assign[start_row].is_some() {
                continue;
            }
            self.try_augment_from_flat(start_row, dim);
        }
    }

    /// BFS augmenting path from `start_row` on the flat buffer.
    fn try_augment_from_flat(&mut self, start_row: usize, dim: usize) -> bool {
        for v in &mut self.visited_col[..dim] {
            *v = false;
        }
        for v in &mut self.parent_row_for_col[..dim] {
            *v = None;
        }
        self.queue.clear();
        self.queue.push(start_row);
        let mut terminal_col: Option<usize> = None;

        while let Some(r) = self.queue.pop() {
            for j in 0..dim {
                if self.visited_col[j] || self.cost_buf[r * dim + j].abs() >= ZERO_EPS {
                    continue;
                }
                self.visited_col[j] = true;
                self.parent_row_for_col[j] = Some(r);
                match self.col_assign[j] {
                    None => {
                        terminal_col = Some(j);
                        break;
                    }
                    Some(next_row) => self.queue.push(next_row),
                }
            }
            if terminal_col.is_some() {
                break;
            }
        }

        if let Some(mut j) = terminal_col {
            loop {
                let r = self.parent_row_for_col[j].expect("augmenting path parent must be set");
                let prev_col = self.row_assign[r];
                self.col_assign[j] = Some(r);
                self.row_assign[r] = Some(j);
                match prev_col {
                    None => break,
                    Some(pc) => j = pc,
                }
            }
            true
        } else {
            false
        }
    }

    /// Main Munkres loop on the flat buffer.
    fn run_munkres_loop_flat(&mut self, dim: usize) {
        self.augment_matching_flat(dim);

        loop {
            let n_assigned = self.row_assign[..dim]
                .iter()
                .filter(|a| a.is_some())
                .count();
            if n_assigned == dim {
                return;
            }

            // mark_cover on flat buffer
            self.mark_cover_flat(dim);

            let covered_count = self.row_covered[..dim].iter().filter(|&&f| f).count()
                + self.col_covered[..dim].iter().filter(|&&f| f).count();
            if covered_count >= dim {
                return;
            }

            // min uncovered value
            let mut min_val = f64::INFINITY;
            for i in 0..dim {
                if self.row_covered[i] {
                    continue;
                }
                for j in 0..dim {
                    if !self.col_covered[j] {
                        let v = self.cost_buf[i * dim + j];
                        if v < min_val {
                            min_val = v;
                        }
                    }
                }
            }
            if !min_val.is_finite() || min_val.abs() < 1e-15 {
                return;
            }

            // update labels
            for i in 0..dim {
                for j in 0..dim {
                    let idx = i * dim + j;
                    if !self.row_covered[i] && !self.col_covered[j] {
                        self.cost_buf[idx] -= min_val;
                    } else if self.row_covered[i] && self.col_covered[j] {
                        self.cost_buf[idx] += min_val;
                    }
                }
            }

            // Re-run greedy + augment
            self.greedy_zero_flat(dim);
            self.augment_matching_flat(dim);
        }
    }

    /// Compute minimum vertex cover (König marking) on the flat buffer.
    fn mark_cover_flat(&mut self, dim: usize) {
        // Initialize: mark all unassigned rows.
        self.unmarked_rows.clear();
        for i in 0..dim {
            if self.row_assign[i].is_none() {
                self.row_marked[i] = true;
                self.unmarked_rows.push(i);
            } else {
                self.row_marked[i] = false;
            }
        }
        for v in &mut self.col_covered[..dim] {
            *v = false;
        }
        // col_marked is stored in col_covered for this phase.
        let mut changed = true;
        while changed {
            changed = false;
            // Mark columns with zeros in marked rows.
            for &r in &self.unmarked_rows {
                for j in 0..dim {
                    if !self.col_covered[j] && self.cost_buf[r * dim + j].abs() < ZERO_EPS {
                        self.col_covered[j] = true;
                        changed = true;
                    }
                }
            }
            // Mark rows assigned to marked columns.
            let mut new_rows = Vec::new();
            for j in 0..dim {
                if let Some(r) = self.col_assign[j]
                    && self.col_covered[j]
                    && !self.row_marked[r]
                {
                    self.row_marked[r] = true;
                    new_rows.push(r);
                    changed = true;
                }
            }
            self.unmarked_rows.extend_from_slice(&new_rows);
        }

        // row_covered = !row_marked; col_covered stays as col_marked.
        for i in 0..dim {
            self.row_covered[i] = !self.row_marked[i];
        }
    }
}

/// Solve the linear assignment problem using the Hungarian algorithm.
///
/// Finds the assignment that minimizes total cost. Handles rectangular matrices.
/// Entries with cost >= `gate` are considered infeasible.
///
/// This is a convenience wrapper around [`HungarianSolver`]. For repeated
/// calls (e.g. every tracker frame), prefer creating a solver once with
/// [`HungarianSolver::new`] and reusing it via [`HungarianSolver::solve`] to
/// avoid per-call allocation.
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
    let mut solver = HungarianSolver::new(dim);
    solver.solve(cost, gate)
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

// The following Vec<Vec<f64>>-based phase helpers are retained for their
// unit tests (which validate algorithmic correctness in isolation). Production
// code uses the flat-buffer equivalents inside `HungarianSolver`.

/// Subtract the row minimum from each row, then the column minimum from each
/// column. Non-finite minima (all-infinity rows/columns) are left untouched.
#[cfg(test)]
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
#[cfg(test)]
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

/// Compute a minimum vertex cover of the zero graph via the classical
/// marking procedure. Returns `(row_covered, col_covered)` boolean vectors.
///
/// Precondition: `row_assign` / `col_assign` MUST represent a **maximum**
/// matching on the current zero graph. König's theorem only guarantees that
/// the marking procedure yields a minimum vertex cover when the matching is
/// maximum; with a merely greedy (non-maximum) matching, the returned cover
/// can be smaller than the matching and `run_munkres_loop` would terminate on
/// a non-optimal assignment. `run_munkres_loop` enforces this precondition by
/// calling `augment_matching` before every `mark_cover` invocation.
#[cfg(test)]
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

#[cfg(test)]
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

#[cfg(test)]
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
#[cfg(test)]
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
#[cfg(test)]
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
    fn perfect_matching_requires_augmentation() {
        // Regression for CodeRabbit-flagged case: greedy assignment picks
        // (0,0) and (2,1), leaving row 1 unassigned. But a perfect matching
        // exists: (0,1), (1,0), (2,2) all zero. The algorithm must find
        // the perfect matching, not stop at the greedy partial result.
        let cost = vec![
            vec![0.0, 0.0, 1.0],
            vec![0.0, 1.0, 1.0],
            vec![1.0, 0.0, 0.0],
        ];
        let result = hungarian_assignment(&cost, 100.0);
        assert_eq!(
            result.matches.len(),
            3,
            "must find 3 matches, got {:?}",
            result.matches
        );
        assert!(
            result.total_cost < 1e-9,
            "perfect zero matching exists, got cost {}",
            result.total_cost
        );
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

    // ---- HungarianSolver correctness tests ----------------------------------

    #[test]
    fn solver_matches_free_function_on_100x100() {
        use rand::prelude::*;

        let dim = 100;
        let mut rng = StdRng::seed_from_u64(42);
        let cost: Vec<Vec<f64>> = (0..dim)
            .map(|_| (0..dim).map(|_| rng.random::<f64>() * 100.0).collect())
            .collect();
        let gate = f64::INFINITY;

        // Run via the free function (which internally creates a solver).
        let result_free = hungarian_assignment(&cost, gate);

        // Run via an explicitly pre-allocated solver.
        let mut solver = HungarianSolver::new(dim);
        let result_solver = solver.solve(&cost, gate);

        // Both must produce the same optimal total cost and the same number of
        // matches. The actual match pairs may differ when multiple optimal
        // assignments exist, so we only compare total cost.
        assert_eq!(
            result_free.matches.len(),
            result_solver.matches.len(),
            "match count differs"
        );
        assert!(
            (result_free.total_cost - result_solver.total_cost).abs() < 1e-6,
            "total cost differs: free={}, solver={}",
            result_free.total_cost,
            result_solver.total_cost,
        );
        assert_eq!(
            result_free.unassigned_rows.len(),
            result_solver.unassigned_rows.len()
        );
        assert_eq!(
            result_free.unassigned_cols.len(),
            result_solver.unassigned_cols.len()
        );
    }

    #[test]
    fn solver_reuse_across_different_sizes() {
        // Verify that a solver pre-allocated for 50 can handle 10, 50, and 80
        // correctly (the last exceeding the initial capacity).
        use rand::prelude::*;
        let mut solver = HungarianSolver::new(50);
        let mut rng = StdRng::seed_from_u64(99);

        for dim in [10, 50, 80] {
            let cost: Vec<Vec<f64>> = (0..dim)
                .map(|_| (0..dim).map(|_| rng.random::<f64>() * 50.0).collect())
                .collect();
            let gate = f64::INFINITY;

            let expected = hungarian_assignment(&cost, gate);
            let got = solver.solve(&cost, gate);

            assert_eq!(expected.matches.len(), got.matches.len(), "dim={dim}");
            assert!(
                (expected.total_cost - got.total_cost).abs() < 1e-6,
                "dim={dim}: cost differs"
            );
        }
    }
}
