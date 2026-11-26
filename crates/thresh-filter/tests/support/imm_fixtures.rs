//! Shared IMM fixtures for the `imm_reference` and `imm_current_policy`
//! targets (test support; no `#[test]` here).
//!
//! Holds the fixture data once — the asymmetric-mixing estimates, the worked
//! measurement, the transition matrices and the zero-destination
//! configuration — together with the native bank builders, the CV+CV
//! end-to-end driver and the tolerance helpers, so the two targets do not
//! duplicate setup code.
//!
//! Tolerances follow the `imm-estimation` capability: absolute 1e-10 plus
//! relative 1e-9, and probability sums within 1e-12. Set the environment
//! variable `IMM_REFERENCE_RESIDUALS` and run with `--nocapture` to print
//! one `imm-residual <kind> abs <a> rel <r> <label>` line per comparison:
//! the worst absolute residual and the worst residual relative to
//! `max(|expected|, 1)`.

use nalgebra::{DMatrix, DVector};
use thresh_filter::imm::{CvMapping, ImmConfig, ImmFilter, ImmStepResult, StateMapping};
use thresh_filter::models::cv::ConstantVelocity;
use thresh_filter::traits::MotionModel;

use super::imm_oracle::{self as oracle, Gaussian, LinearMode, Measurement, SpecifiedCycle};

/// Absolute comparison tolerance.
pub const ATOL: f64 = 1e-10;
/// Relative comparison tolerance.
pub const RTOL: f64 = 1e-9;
/// Probability-sum tolerance.
pub const PROBABILITY_SUM_TOL: f64 = 1e-12;

/// CV acceleration noise of the worked CV mode and of CV+CV mode 1.
pub const SIGMA_A_MODE_1: f64 = 1.0;
/// CV acceleration noise of CV+CV mode 2.
pub const SIGMA_A_MODE_2: f64 = 3.0;
/// CA jerk noise of the worked CA mode.
pub const SIGMA_J: f64 = 1.0;
/// Elapsed times of the specified grid, in seconds.
pub const ELAPSED_TIMES: [f64; 3] = [0.1, 1.0, 2.3];

// ---------------------------------------------------------------------------
// Fixture data
// ---------------------------------------------------------------------------

/// Identity transition matrix.
pub fn pi_identity() -> DMatrix<f64> {
    DMatrix::identity(2, 2)
}

/// Symmetric transition matrix `[[0.95, 0.05], [0.05, 0.95]]`.
pub fn pi_symmetric() -> DMatrix<f64> {
    DMatrix::from_row_slice(2, 2, &[0.95, 0.05, 0.05, 0.95])
}

/// Asymmetric transition matrix `[[0.9, 0.1], [0.2, 0.8]]`.
pub fn pi_asymmetric() -> DMatrix<f64> {
    DMatrix::from_row_slice(2, 2, &[0.9, 0.1, 0.2, 0.8])
}

/// Zero-destination transition matrix `[[1, 0], [1, 0]]` (`c_2 = 0`).
pub fn pi_zero_destination() -> DMatrix<f64> {
    DMatrix::from_row_slice(2, 2, &[1.0, 0.0, 1.0, 0.0])
}

/// Worked mode probabilities `[0.6, 0.4]`.
pub fn mu_worked() -> DVector<f64> {
    DVector::from_column_slice(&[0.6, 0.4])
}

/// Mode 1's estimate: `x = [0, 10, 0, 5, 100, 0]`, `P = diag(4, 1, 4, 1, 4, 1)`.
pub fn worked_cv_leaf() -> Gaussian {
    let p = DMatrix::from_diagonal(&DVector::from_column_slice(&[4.0, 1.0, 4.0, 1.0, 4.0, 1.0]));
    Gaussian::new(&[0.0, 10.0, 0.0, 5.0, 100.0, 0.0], p)
}

/// Mode 2's CA estimate `[x, vx, ax, y, vy, ay, z, vz, az]`. Its acceleration
/// entries and the vx–ax covariance 0.5 lie outside the common state.
pub fn worked_ca_leaf() -> Gaussian {
    let mut p = DMatrix::from_diagonal(&DVector::from_column_slice(&[
        9.0, 4.0, 1.0, 9.0, 4.0, 1.0, 9.0, 4.0, 1.0,
    ]));
    p[(0, 1)] = 1.0;
    p[(1, 0)] = 1.0;
    p[(1, 2)] = 0.5;
    p[(2, 1)] = 0.5;
    Gaussian::new(&[2.0, 12.0, 1.5, -1.0, 4.0, 0.5, 101.0, 0.0, 0.0], p)
}

/// Mode 2's common estimate, written out by hand: `x = [2, 12, −1, 4, 101, 0]`,
/// `P = diag(9, 4, 9, 4, 9, 4)` plus a covariance of 1 between x and vx.
pub fn worked_ca_common() -> Gaussian {
    let mut p =
        DMatrix::from_diagonal(&DVector::from_column_slice(&[9.0, 4.0, 9.0, 4.0, 9.0, 4.0]));
    p[(0, 1)] = 1.0;
    p[(1, 0)] = 1.0;
    Gaussian::new(&[2.0, 12.0, -1.0, 4.0, 101.0, 0.0], p)
}

/// Position observation `H` (3 x 6) of x, y and z.
pub fn position_observation() -> DMatrix<f64> {
    let mut h = DMatrix::zeros(3, 6);
    h[(0, 0)] = 1.0;
    h[(1, 2)] = 1.0;
    h[(2, 4)] = 1.0;
    h
}

/// Position measurement `z` with `R = 4I`.
pub fn position_measurement(z: &[f64]) -> Measurement {
    Measurement {
        z: DVector::from_column_slice(z),
        h: position_observation(),
        r: DMatrix::identity(3, 3) * 4.0,
    }
}

/// Worked measurement `z = [11, 6, 100.5]`, `R = 4I`.
pub fn worked_measurement() -> Measurement {
    position_measurement(&[11.0, 6.0, 100.5])
}

/// Underflow measurement `z = [2011, 6, 100.5]`: both linear likelihoods
/// underflow to zero in float64.
pub fn underflow_measurement() -> Measurement {
    position_measurement(&[2011.0, 6.0, 100.5])
}

/// Exact combined mean of the zero-destination case (`Π = [[1, 0], [1, 0]]`,
/// worked estimates, `μ` and measurement, floor 0): mode 1's posterior mean.
pub fn zero_destination_exact_mean() -> DVector<f64> {
    DVector::from_column_slice(&[
        ratio(1689797, 152447),
        ratio(1612044, 152447),
        ratio(207755578, 37959303),
        ratio(63716985, 12653101),
        ratio(2541809541, 25306202),
        ratio(494019, 12653101),
    ])
}

/// Fixed per-cycle offsets added to the truth position `[10 t, 5 t, 100]`.
const MEASUREMENT_OFFSETS: [[f64; 3]; 5] = [
    [1.0, 1.0, 0.5],
    [-0.5, 0.75, -0.25],
    [0.25, -1.0, 0.75],
    [-1.0, 0.5, -0.5],
    [0.75, -0.25, 0.25],
];

/// Number of cycles in each CV+CV end-to-end run.
pub const END_TO_END_CYCLES: usize = MEASUREMENT_OFFSETS.len();

/// Deterministic measurements at `t_k = k dt`, `k = 1..=END_TO_END_CYCLES`:
/// the truth position `[10 t, 5 t, 100]` plus a fixed offset. At `dt = 1`
/// the first measurement is the worked measurement.
pub fn measurement_sequence(dt: f64) -> Vec<Measurement> {
    MEASUREMENT_OFFSETS
        .iter()
        .enumerate()
        .map(|(k, off)| {
            let t = (k + 1) as f64 * dt;
            position_measurement(&[10.0 * t + off[0], 5.0 * t + off[1], 100.0 + off[2]])
        })
        .collect()
}

/// Two-cycle underflow sequence: the underflow measurement, then the same
/// point advanced by one second of the truth velocity.
pub fn underflow_sequence() -> Vec<Measurement> {
    vec![
        underflow_measurement(),
        position_measurement(&[2021.0, 11.0, 100.5]),
    ]
}

// ---------------------------------------------------------------------------
// Oracle banks
// ---------------------------------------------------------------------------

/// Oracle CV+CA modes (σ_a = 1, σ_j = 1) over `dt`.
pub fn oracle_cv_ca_modes(dt: f64) -> Vec<LinearMode> {
    vec![
        LinearMode::cv(dt, SIGMA_A_MODE_1),
        LinearMode::ca(dt, SIGMA_J),
    ]
}

/// Oracle CV+CV modes (σ_a = 1 and 3) over `dt`.
pub fn oracle_cv_cv_modes(dt: f64) -> Vec<LinearMode> {
    vec![
        LinearMode::cv(dt, SIGMA_A_MODE_1),
        LinearMode::cv(dt, SIGMA_A_MODE_2),
    ]
}

/// Worked CV+CA leaves (CV 6-D, CA 9-D).
pub fn worked_cv_ca_leaves() -> Vec<Gaussian> {
    vec![worked_cv_leaf(), worked_ca_leaf()]
}

/// Worked CV+CV leaves: the asymmetric-mixing common estimates.
pub fn worked_cv_cv_leaves() -> Vec<Gaussian> {
    vec![worked_cv_leaf(), worked_ca_common()]
}

// ---------------------------------------------------------------------------
// Native banks
// ---------------------------------------------------------------------------

/// Overwrite every native leaf with the given estimates (model space).
pub fn set_native_leaves(bank: &mut ImmFilter, leaves: &[Gaussian]) {
    for (f, g) in bank.filters.iter_mut().zip(leaves) {
        f.leaf.set_x(g.mean.clone());
        f.leaf.set_p(g.cov.clone());
    }
}

/// Native leaf `j` in its model space.
pub fn native_leaf(bank: &ImmFilter, j: usize) -> Gaussian {
    let leaf = &bank.filters[j].leaf;
    Gaussian {
        mean: leaf.x().clone(),
        cov: leaf.p().clone(),
    }
}

fn finish_bank(config: ImmConfig, leaves: &[Gaussian], floor: f64) -> ImmFilter {
    let mut bank = ImmFilter::new(config, &DVector::zeros(6), &DMatrix::identity(6, 6));
    bank.mode_probability_floor = floor;
    set_native_leaves(&mut bank, leaves);
    bank
}

/// Native `ImmConfig::cv_ca(1, 1)` bank with the given `Π`, `μ` and floor,
/// holding the worked leaves.
pub fn native_cv_ca_bank(pi: &DMatrix<f64>, mu: &DVector<f64>, floor: f64) -> ImmFilter {
    let mut config = ImmConfig::cv_ca(SIGMA_A_MODE_1, SIGMA_J);
    config.transition_matrix = pi.clone();
    config.initial_mode_probabilities = mu.clone();
    finish_bank(config, &worked_cv_ca_leaves(), floor)
}

/// Native bank of CV modes (identity mappings) with the given noises, `Π`,
/// `μ`, leaves and floor.
pub fn native_cv_bank(
    sigmas: &[f64],
    pi: &DMatrix<f64>,
    mu: &DVector<f64>,
    leaves: &[Gaussian],
    floor: f64,
) -> ImmFilter {
    let config = ImmConfig {
        models: sigmas
            .iter()
            .map(|&s| Box::new(ConstantVelocity::new(s)) as Box<dyn MotionModel>)
            .collect(),
        mappings: sigmas
            .iter()
            .map(|_| Box::new(CvMapping) as Box<dyn StateMapping>)
            .collect(),
        transition_matrix: pi.clone(),
        initial_mode_probabilities: mu.clone(),
    };
    finish_bank(config, leaves, floor)
}

/// Native CV+CV bank (σ_a = 1 and 3) holding the worked common estimates.
pub fn native_cv_cv_bank(pi: &DMatrix<f64>, mu: &DVector<f64>, floor: f64) -> ImmFilter {
    native_cv_bank(
        &[SIGMA_A_MODE_1, SIGMA_A_MODE_2],
        pi,
        mu,
        &worked_cv_cv_leaves(),
        floor,
    )
}

/// One native step on a measurement.
pub fn native_step(bank: &mut ImmFilter, dt: f64, m: &Measurement) -> ImmStepResult {
    bank.step(dt, &m.z, &m.h, &m.r)
}

// ---------------------------------------------------------------------------
// CV+CV end-to-end driver
// ---------------------------------------------------------------------------

/// What one compared end-to-end cycle produced.
#[derive(Clone, Debug)]
pub struct CycleRecord {
    /// Mode probabilities before the cycle (oracle).
    pub mu_prior: DVector<f64>,
    /// Native posterior mode probabilities.
    pub native_mu: DVector<f64>,
    /// The oracle's specified cycle.
    pub oracle: SpecifiedCycle,
}

/// A native CV+CV bank and the independently evolving oracle state.
struct EndToEndRun {
    label: String,
    dt: f64,
    pi: DMatrix<f64>,
    native: ImmFilter,
    modes: Vec<LinearMode>,
    leaves: Vec<Gaussian>,
    mu: DVector<f64>,
}

impl EndToEndRun {
    fn new(label: &str, pi: &DMatrix<f64>, dt: f64) -> Self {
        Self {
            label: label.to_owned(),
            dt,
            pi: pi.clone(),
            native: native_cv_cv_bank(pi, &mu_worked(), 0.0),
            modes: oracle_cv_cv_modes(dt),
            leaves: worked_cv_cv_leaves(),
            mu: mu_worked(),
        }
    }

    /// Run one cycle on both sides, assert the probability guard and
    /// agreement, then advance the oracle from its own posterior.
    fn cycle(&mut self, k: usize, m: &Measurement) -> CycleRecord {
        let label = format!("{} cycle {}", self.label, k + 1);
        let oracle = oracle::specified_cycle(&self.modes, &self.leaves, &self.mu, &self.pi, m);
        assert_predicted_probabilities_above_bound(&label, &oracle.c);
        let result = native_step(&mut self.native, self.dt, m);
        assert_cycle_agrees(&label, &self.native, &result, &oracle);
        let record = CycleRecord {
            mu_prior: self.mu.clone(),
            native_mu: result.mode_probabilities,
            oracle,
        };
        self.leaves = record.oracle.posterior_leaves();
        self.mu = record.oracle.posterior_mu.clone();
        record
    }
}

/// Compare native `μ⁺`, every leaf posterior and the combined moments with
/// one oracle cycle, and check both posteriors sum to one.
fn assert_cycle_agrees(
    label: &str,
    native: &ImmFilter,
    result: &ImmStepResult,
    oracle: &SpecifiedCycle,
) {
    let kind = Residual::NativeVsOracle;
    assert_probabilities_sum_to_one(&format!("{label} native μ⁺"), &result.mode_probabilities);
    assert_probabilities_sum_to_one(&format!("{label} oracle μ⁺"), &oracle.posterior_mu);
    assert_vector_close(
        kind,
        &format!("{label} μ⁺"),
        &result.mode_probabilities,
        &oracle.posterior_mu,
    );
    for (j, expected) in oracle.posterior_leaves().iter().enumerate() {
        assert_gaussian_close(
            kind,
            &format!("{label} leaf {j}"),
            &native_leaf(native, j),
            expected,
        );
    }
    let combined = Gaussian {
        mean: result.state.clone(),
        cov: result.covariance.clone(),
    };
    assert_gaussian_close(
        kind,
        &format!("{label} combined"),
        &combined,
        &oracle.combined,
    );
}

/// Run a CV+CV bank (σ_a = 1 and 3, floor 0, worked leaves and `μ`) through
/// `measurements` at elapsed time `dt`, comparing native with the oracle at
/// every cycle and asserting that every predicted probability exceeds 1e-12.
pub fn run_cv_cv_end_to_end(
    label: &str,
    pi: &DMatrix<f64>,
    dt: f64,
    measurements: &[Measurement],
) -> Vec<CycleRecord> {
    let mut run = EndToEndRun::new(label, pi, dt);
    measurements
        .iter()
        .enumerate()
        .map(|(k, m)| run.cycle(k, m))
        .collect()
}

// ---------------------------------------------------------------------------
// Tolerance helpers
// ---------------------------------------------------------------------------

/// What a comparison checks, for residual reporting.
#[derive(Clone, Copy, Debug)]
pub enum Residual {
    /// Production output against the oracle.
    NativeVsOracle,
    /// Oracle against a hand-derived or committed specified value.
    OracleVsHand,
    /// Production output against a hand-derived specified value.
    NativeVsHand,
    /// Production output against a pinned current-policy value.
    NativeVsPinned,
    /// Oracle (current-policy layer) against a pinned current-policy value.
    OracleVsPinned,
}

/// Both operands must be finite: with an infinite expected value the
/// relative bound is infinite too, and any finite actual value would pass.
/// A test that expects a non-finite value must assert it explicitly.
fn within_tolerance(actual: f64, expected: f64) -> bool {
    actual.is_finite()
        && expected.is_finite()
        && (actual - expected).abs() <= ATOL + RTOL * expected.abs()
}

/// Print the worst absolute residual and the worst residual relative to
/// `max(|expected|, 1)` when `IMM_REFERENCE_RESIDUALS` is set.
fn report(kind: Residual, label: &str, actual: &[f64], expected: &[f64]) {
    if std::env::var_os("IMM_REFERENCE_RESIDUALS").is_none() {
        return;
    }
    let pairs = actual.iter().zip(expected);
    let abs = pairs
        .clone()
        .map(|(a, e)| (a - e).abs())
        .fold(0.0, f64::max);
    let rel = pairs
        .map(|(a, e)| (a - e).abs() / e.abs().max(1.0))
        .fold(0.0, f64::max);
    eprintln!("imm-residual {kind:?} abs {abs:.3e} rel {rel:.3e} {label}");
}

/// Assert `|actual − expected| ≤ 1e-10 + 1e-9 |expected|`.
pub fn assert_close(kind: Residual, label: &str, actual: f64, expected: f64) {
    report(kind, label, &[actual], &[expected]);
    assert!(
        within_tolerance(actual, expected),
        "{label}: {actual:e} vs expected {expected:e} (|Δ| = {:e})",
        (actual - expected).abs()
    );
}

/// Element-wise [`assert_close`] over a slice.
pub fn assert_slice_close(kind: Residual, label: &str, actual: &[f64], expected: &[f64]) {
    assert_eq!(actual.len(), expected.len(), "{label}: length mismatch");
    report(kind, label, actual, expected);
    for (i, (&a, &e)) in actual.iter().zip(expected).enumerate() {
        assert!(
            within_tolerance(a, e),
            "{label}[{i}]: {a:e} vs expected {e:e} (|Δ| = {:e})",
            (a - e).abs()
        );
    }
}

/// Element-wise [`assert_close`] over a vector.
pub fn assert_vector_close(
    kind: Residual,
    label: &str,
    actual: &DVector<f64>,
    expected: &DVector<f64>,
) {
    assert_slice_close(kind, label, actual.as_slice(), expected.as_slice());
}

/// Element-wise [`assert_close`] over a matrix.
pub fn assert_matrix_close(
    kind: Residual,
    label: &str,
    actual: &DMatrix<f64>,
    expected: &DMatrix<f64>,
) {
    assert_eq!(actual.shape(), expected.shape(), "{label}: shape mismatch");
    assert_slice_close(kind, label, actual.as_slice(), expected.as_slice());
}

/// Mean and covariance of two estimates agree element-wise.
pub fn assert_gaussian_close(kind: Residual, label: &str, actual: &Gaussian, expected: &Gaussian) {
    assert_vector_close(kind, &format!("{label} mean"), &actual.mean, &expected.mean);
    assert_matrix_close(kind, &format!("{label} cov"), &actual.cov, &expected.cov);
}

/// Mode probabilities sum to one within 1e-12 and are all finite.
pub fn assert_probabilities_sum_to_one(label: &str, mu: &DVector<f64>) {
    assert!(
        mu.iter().all(|p| p.is_finite()),
        "{label}: non-finite {mu:?}"
    );
    let sum = mu.sum();
    assert!(
        (sum - 1.0).abs() <= PROBABILITY_SUM_TOL,
        "{label}: sums to {sum:e}"
    );
}

/// Probability guard: every predicted probability exceeds 1e-12.
pub fn assert_predicted_probabilities_above_bound(label: &str, c: &DVector<f64>) {
    assert!(
        c.iter().all(|&cj| cj > oracle::SPECIFIED_PROBABILITY_BOUND),
        "{label}: predicted probabilities {c:?} leave the specified range (> 1e-12)"
    );
}

/// `p / q` as `f64`, for exact-rational expectations.
pub fn ratio(p: i64, q: i64) -> f64 {
    p as f64 / q as f64
}

/// Vector of exact rationals sharing one denominator.
pub fn rational_vector(numerators: &[i64], denominator: i64) -> DVector<f64> {
    DVector::from_iterator(
        numerators.len(),
        numerators.iter().map(|&p| ratio(p, denominator)),
    )
}
