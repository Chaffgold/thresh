//! CURRENT-POLICY CHARACTERISATION — not specified behaviour; changes
//! deliberately in `correct-native-imm-mode-policies`.
//!
//! These tests pin what the native IMM does today where the `imm-estimation`
//! capability leaves a mode policy unspecified. They are not evidence for any
//! `imm-estimation` requirement and must not be cited as such. Specified
//! behaviour is verified in `imm_reference.rs`, which references none of
//! these tests.
//!
//! Every test name starts with `current_policy_` and cites its defect ID:
//!
//! - **F1** — reconstruction of CA acceleration when a common estimate is
//!   expanded, after mixing and after the common-space update (mean 0,
//!   variance 100 (m/s²)², zero cross-covariance);
//! - **F3** — the default mode-probability floor 0.01 and its
//!   renormalisation;
//! - **F4** — the small-probability thresholds: mixing treats `c_j ≤ 1e-30`
//!   as zero (mixed state and covariance zero), and the mode update clamps
//!   `c_j` at 1e-300 before its logarithm;
//! - **F5** — mode probabilities left stale when a prediction is not
//!   followed by an update;
//! - **F6** — the predicted (pre-update) combined estimate weighted by the
//!   prior mode probabilities;
//! - **F7** — the transition matrix held at its configured value whatever
//!   the elapsed time.
//!
//! Each test emulates the declared current policy through the oracle's
//! current-policy layer (`CurrentPolicy::current`) and records the exact or
//! textbook value beside the pinned value. Characterisation setup: the
//! worked CV+CA bank is `ImmConfig::cv_ca(1, 1)` (σ_a = 1, σ_j = 1) with
//! dt = 1 s, `R = 4I` and `z = [11, 6, 100.5]`. Pinned decimals quoted to ten
//! places round below the 1e-10 absolute tolerance.

#[allow(dead_code)]
mod support;

use nalgebra::{DMatrix, DVector};
use thresh_filter::imm::{ImmConfig, ImmFilter, ImmStepResult};

use support::imm_fixtures::{
    ELAPSED_TIMES, END_TO_END_CYCLES, Residual, assert_close, assert_gaussian_close,
    assert_slice_close, assert_vector_close, measurement_sequence, mu_worked, native_cv_bank,
    native_cv_ca_bank, native_leaf, native_step, oracle_cv_ca_modes, pi_asymmetric, pi_identity,
    pi_symmetric, pi_zero_destination, position_measurement, rational_vector, run_cv_cv_end_to_end,
    set_native_leaves, worked_ca_leaf, worked_cv_ca_leaves, worked_cv_leaf, worked_measurement,
    zero_destination_exact_mean,
};
use support::imm_oracle::{
    CA_ACCEL_INDICES, CoastPolicy, CurrentPolicy, Gaussian, LinearMode, Measurement, PolicyCycle,
    PredictedOutputWeighting, ca_common_selector, joseph_update, mix_destination, mixing_weights,
    mode_posterior, policy_cycle, predicted_mode_probabilities, specified_cycle,
};

// ---------------------------------------------------------------------------
// Pinned values and their exact or textbook counterparts
// ---------------------------------------------------------------------------

/// F1: worked CA log-likelihood under the current reconstruction.
const LL_CA_PINNED: f64 = -8.519306179549595;
/// F1: worked posterior (asymmetric Π).
const MU_ASYMMETRIC_PINNED: [f64; 2] = [0.9274195958363679, 0.07258040416363172];
/// F1: worked combined mean (re-derivations agree within 1.4e-14).
const COMBINED_MEAN_PINNED: [f64; 6] = [
    10.84142489054062,
    10.234515136297619,
    5.5293027990598915,
    5.285603042308766,
    100.36744617154977,
    0.03519257163692508,
];
/// F1: worked combined `P[x, x]` and `P[x, vx]`.
const COMBINED_P_XX_PINNED: f64 = 2.7725529136;
const COMBINED_P_XVX_PINNED: f64 = 1.1895654128;
/// F1: worked posteriors with the identity and symmetric Π.
const MU_IDENTITY_PINNED: [f64; 2] = [0.9421643833, 0.0578356167];
const MU_SYMMETRIC_PINNED: [f64; 2] = [0.9334538576, 0.0665461424];
/// F1 textbook (Π = I): no interaction occurs, so a bank that keeps each
/// mode's own state predicts the CA leaf from its own 9-D estimate
/// (acceleration `[1.5, 0.5, 0]`).
const MU_IDENTITY_OWN_STATE: [f64; 2] = [0.866616032108139, 0.13338396789186124];
/// F1 textbook: a native-space update of the same 9-D CA prediction keeps
/// acceleration information instead of resetting it.
const ACCEL_NATIVE_SPACE_UPDATE: [f64; 3] = [-2.5836805752, 3.1901419706, -0.3650991591];
const ACCEL_VAR_NATIVE_SPACE_UPDATE: [f64; 3] = [44.5267402892, 40.0509283883, 39.4850421386];

/// F3: posterior, vz and var(vz) at the zero-probability destination with
/// the default floor.
const MU_DEFAULT_FLOOR_PINNED: [f64; 2] = [0.9900990099, 0.0099009901];
const VZ_DEFAULT_FLOOR_PINNED: f64 = 1.7555881216;
const VAR_VZ_DEFAULT_FLOOR_PINNED: f64 = 297.3863109839;
/// F3 exact (floor 0, zero-probability mode excluded): var(vz).
const VAR_VZ_EXACT: f64 = 2.6226006573;

/// F5: posterior after one coasted prediction, then an update.
const MU_COAST_PINNED: [f64; 2] = [0.8900140198, 0.1099859802];
/// F5 textbook: propagating `μ ← Πᵀμ` on the coast.
const MU_COAST_PROPAGATED: [f64; 2] = [0.8963774786, 0.1036225214];

/// F6: μ-weighted predicted x.
const PREDICTED_X_PINNED: f64 = 11.65704584040747;

/// F7: predicted probabilities `c = Πᵀμ` for the asymmetric Π at every dt.
const C_ASYMMETRIC_HELD: [f64; 2] = [0.62, 0.38];

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

/// One worked CV+CA cycle in the oracle under `policy`.
fn worked_policy_cycle(pi: &DMatrix<f64>, policy: &CurrentPolicy) -> PolicyCycle {
    policy_cycle(
        &oracle_cv_ca_modes(1.0),
        &worked_cv_ca_leaves(),
        &mu_worked(),
        pi,
        Some(&worked_measurement()),
        policy,
    )
}

/// One worked native CV+CA step.
fn native_worked_step(pi: &DMatrix<f64>, floor: f64) -> (ImmFilter, ImmStepResult) {
    let mut bank = native_cv_ca_bank(pi, &mu_worked(), floor);
    let result = native_step(&mut bank, 1.0, &worked_measurement());
    (bank, result)
}

fn combined_of(result: &ImmStepResult) -> Gaussian {
    Gaussian {
        mean: result.state.clone(),
        cov: result.covariance.clone(),
    }
}

/// Native posterior, every leaf (model space) and the combined moments equal
/// the oracle under the declared current policy.
fn assert_native_matches_policy(
    label: &str,
    bank: &ImmFilter,
    result: &ImmStepResult,
    oracle: &PolicyCycle,
) {
    let kind = Residual::NativeVsOracle;
    assert_vector_close(
        kind,
        &format!("{label} μ⁺"),
        &result.mode_probabilities,
        &oracle.mode_probabilities,
    );
    for (j, expected) in oracle.leaves.iter().enumerate() {
        assert_gaussian_close(
            kind,
            &format!("{label} leaf {j}"),
            &native_leaf(bank, j),
            expected,
        );
    }
    let expected = oracle.combined.as_ref().expect("cycle updated");
    assert_gaussian_close(
        kind,
        &format!("{label} combined"),
        &combined_of(result),
        expected,
    );
}

/// Both native and the oracle under the declared policy reproduce a pinned
/// value.
fn assert_pinned(label: &str, native: &[f64], oracle: &[f64], pinned: &[f64]) {
    assert_slice_close(Residual::NativeVsPinned, label, native, pinned);
    assert_slice_close(Residual::OracleVsPinned, label, oracle, pinned);
}

/// Pinned F1 reconstruction: every acceleration component has mean 0,
/// variance 100 and zero cross-covariance.
fn assert_acceleration_reset(label: &str, ca_leaf: &Gaussian) {
    for &k in &CA_ACCEL_INDICES {
        assert_close(
            Residual::NativeVsPinned,
            &format!("{label} accel mean {k}"),
            ca_leaf.mean[k],
            0.0,
        );
        assert_close(
            Residual::NativeVsPinned,
            &format!("{label} accel var {k}"),
            ca_leaf.cov[(k, k)],
            100.0,
        );
        for other in (0..9).filter(|&o| o != k) {
            assert_eq!(
                ca_leaf.cov[(k, other)],
                0.0,
                "{label} cross-cov ({k}, {other})"
            );
            assert_eq!(
                ca_leaf.cov[(other, k)],
                0.0,
                "{label} cross-cov ({other}, {k})"
            );
        }
    }
}

/// `c₁` under the rate-consistent `Π(dt) = Π^dt` of a 2 x 2 stochastic `Π`:
/// with `a = π₁₂`, `b = π₂₁`, `λ = 1 − a − b` and stationary `s₁ = b/(a + b)`,
/// `c₁(dt) = s₁ + λ^dt (μ₁ − s₁)`.
fn rate_consistent_c1(pi: &DMatrix<f64>, mu: &DVector<f64>, dt: f64) -> f64 {
    let (a, b) = (pi[(0, 1)], pi[(1, 0)]);
    let s1 = b / (a + b);
    s1 + (1.0 - a - b).powf(dt) * (mu[0] - s1)
}

// ---------------------------------------------------------------------------
// F1: CA reconstruction
// ---------------------------------------------------------------------------

#[test]
fn current_policy_f1_worked_cv_ca_cycle_asymmetric() {
    // Policy: CA reconstruction (mean 0, variance 100) at mixing and after
    // the common-space update; floor 0.
    // Textbook: none. The CA log-likelihood has no policy-free value; the
    // CV half (ℓ_CV) is specified and verified in imm_reference.rs.
    let policy = CurrentPolicy::current(0.0);
    let oracle = worked_policy_cycle(&pi_asymmetric(), &policy);
    let lls = oracle.log_likelihoods();
    assert_close(Residual::OracleVsPinned, "ℓ_CA", lls[1], LL_CA_PINNED);
    let (bank, result) = native_worked_step(&pi_asymmetric(), 0.0);
    assert_native_matches_policy("asymmetric", &bank, &result, &oracle);
    let combined = oracle.combined.as_ref().expect("cycle updated");
    let (mu, mu_oracle) = (&result.mode_probabilities, &oracle.mode_probabilities);
    assert_pinned(
        "μ⁺",
        mu.as_slice(),
        mu_oracle.as_slice(),
        &MU_ASYMMETRIC_PINNED,
    );
    let mean = result.state.as_slice();
    assert_pinned(
        "combined mean",
        mean,
        combined.mean.as_slice(),
        &COMBINED_MEAN_PINNED,
    );
    let (p, p_oracle) = (&result.covariance, &combined.cov);
    let xx = [COMBINED_P_XX_PINNED];
    assert_pinned("P[x,x]", &[p[(0, 0)]], &[p_oracle[(0, 0)]], &xx);
    let xvx = [COMBINED_P_XVX_PINNED];
    assert_pinned("P[x,vx]", &[p[(0, 1)]], &[p_oracle[(0, 1)]], &xvx);
    // Native's ℓ_CA implied by its posterior ratio.
    let implied = (mu[1] / mu[0]).ln() - (oracle.c[1] / oracle.c[0]).ln() + lls[0];
    assert_close(
        Residual::NativeVsPinned,
        "implied ℓ_CA",
        implied,
        LL_CA_PINNED,
    );
    // The pin depends on the reconstruction: variance 1 moves ℓ_CA.
    let other = CurrentPolicy {
        reconstruction_var: 1.0,
        ..policy
    };
    let moved = worked_policy_cycle(&pi_asymmetric(), &other).log_likelihoods()[1];
    assert!(
        (moved - LL_CA_PINNED).abs() > 1e-3,
        "ℓ_CA {moved} should depend on reconstruction"
    );
}

#[test]
fn current_policy_f1_worked_cv_ca_cycle_identity_posterior() {
    // Policy: CA reconstruction (mean 0, variance 100); floor 0.
    let oracle = worked_policy_cycle(&pi_identity(), &CurrentPolicy::current(0.0));
    let (bank, result) = native_worked_step(&pi_identity(), 0.0);
    assert_native_matches_policy("identity", &bank, &result, &oracle);
    let mu = result.mode_probabilities.as_slice();
    let mu_oracle = oracle.mode_probabilities.as_slice();
    assert_pinned("identity μ⁺", mu, mu_oracle, &MU_IDENTITY_PINNED);
    // Textbook (own state kept): predict each leaf from its own estimate.
    let m = worked_measurement();
    let (cv, ca) = (LinearMode::cv(1.0, 1.0), LinearMode::ca(1.0, 1.0));
    let ll_cv = joseph_update(&cv.predict(&worked_cv_leaf()), &m).log_likelihood;
    let ll_ca = joseph_update(&ca.marginal(&ca.predict(&worked_ca_leaf())), &m).log_likelihood;
    let own = mode_posterior(&mu_worked(), &[ll_cv, ll_ca]);
    assert_slice_close(
        Residual::OracleVsHand,
        "own-state μ⁺",
        own.as_slice(),
        &MU_IDENTITY_OWN_STATE,
    );
    assert!(
        (own[0] - mu[0]).abs() > 1e-2,
        "reconstruction moves the identity posterior"
    );
}

#[test]
fn current_policy_f1_worked_cv_ca_cycle_symmetric_posterior() {
    // Policy: CA reconstruction (mean 0, variance 100); floor 0.
    // Textbook: none — mixing with a non-identity Π needs a reconstruction.
    let oracle = worked_policy_cycle(&pi_symmetric(), &CurrentPolicy::current(0.0));
    let (bank, result) = native_worked_step(&pi_symmetric(), 0.0);
    assert_native_matches_policy("symmetric", &bank, &result, &oracle);
    let mu = result.mode_probabilities.as_slice();
    let mu_oracle = oracle.mode_probabilities.as_slice();
    assert_pinned("symmetric μ⁺", mu, mu_oracle, &MU_SYMMETRIC_PINNED);
}

#[test]
fn current_policy_f1_ca_acceleration_reset_after_update() {
    // Policy: the common-space update re-expands the CA leaf with
    // acceleration mean 0 and variance 100.
    let (bank, _) = native_worked_step(&pi_asymmetric(), 0.0);
    let ca_leaf = native_leaf(&bank, 1);
    assert_acceleration_reset("after update", &ca_leaf);
    // Textbook: update the same 9-D prediction in the CA state itself.
    let oracle = worked_policy_cycle(&pi_asymmetric(), &CurrentPolicy::current(0.0));
    let w = worked_measurement();
    let in_ca_state = Measurement {
        h: &w.h * ca_common_selector(),
        ..w
    };
    let textbook = joseph_update(&oracle.predicted_native[1], &in_ca_state).posterior;
    let accel: Vec<f64> = CA_ACCEL_INDICES.iter().map(|&k| textbook.mean[k]).collect();
    let var: Vec<f64> = CA_ACCEL_INDICES
        .iter()
        .map(|&k| textbook.cov[(k, k)])
        .collect();
    assert_slice_close(
        Residual::OracleVsHand,
        "textbook accel",
        &accel,
        &ACCEL_NATIVE_SPACE_UPDATE,
    );
    assert_slice_close(
        Residual::OracleVsHand,
        "textbook accel var",
        &var,
        &ACCEL_VAR_NATIVE_SPACE_UPDATE,
    );
    // The reset touches only acceleration: the common marginals agree.
    let ca = LinearMode::ca(1.0, 1.0);
    let (native_common, textbook_common) = (ca.marginal(&ca_leaf), ca.marginal(&textbook));
    assert_gaussian_close(
        Residual::NativeVsOracle,
        "CA common posterior",
        &native_common,
        &textbook_common,
    );
}

/// After `interaction_step()`, the CA leaf carries the reset acceleration and
/// its common marginal round-trips to `(x0_2, P0_2)`.
fn check_ca_round_trip(label: &str, pi: &DMatrix<f64>) {
    let mut bank = native_cv_ca_bank(pi, &mu_worked(), 0.0);
    bank.interaction_step();
    let leaf = native_leaf(&bank, 1);
    assert_acceleration_reset(label, &leaf);
    let policy = CurrentPolicy::current(0.0);
    let oracle = policy_cycle(
        &oracle_cv_ca_modes(1.0),
        &worked_cv_ca_leaves(),
        &mu_worked(),
        pi,
        None,
        &policy,
    );
    let kind = Residual::NativeVsOracle;
    assert_gaussian_close(
        kind,
        &format!("{label} CA leaf"),
        &leaf,
        &oracle.mixed_native[1],
    );
    let common = LinearMode::ca(1.0, 1.0).marginal(&leaf);
    assert_gaussian_close(
        kind,
        &format!("{label} CA common"),
        &common,
        &oracle.mixed_common[1],
    );
}

#[test]
fn current_policy_f1_ca_leaf_round_trips_after_interaction() {
    // Textbook: the leaf's own pre-mixing acceleration [1.5, 0.5, 0], with
    // variance 1 and a vx–ax covariance of 0.5, which a bank carrying
    // mode-specific components would keep.
    let own = worked_ca_leaf();
    let accel: Vec<f64> = CA_ACCEL_INDICES.iter().map(|&k| own.mean[k]).collect();
    assert_eq!(accel, [1.5, 0.5, 0.0]);
    assert_eq!(own.cov[(1, 2)], 0.5);
    check_ca_round_trip("identity", &pi_identity());
    check_ca_round_trip("symmetric", &pi_symmetric());
    check_ca_round_trip("asymmetric", &pi_asymmetric());
}

// ---------------------------------------------------------------------------
// F3 and F4: floor and small-probability thresholds
// ---------------------------------------------------------------------------

#[test]
fn current_policy_f3_default_floor_at_zero_probability_destination() {
    // Policy: default floor 0.01 with renormalisation, plus the F4
    // zero-destination assignment; Π = [[1, 0], [1, 0]].
    let default_bank = ImmFilter::new(
        ImmConfig::cv_ca(1.0, 1.0),
        &DVector::zeros(6),
        &DMatrix::identity(6, 6),
    );
    assert_eq!(
        default_bank.mode_probability_floor,
        CurrentPolicy::DEFAULT_FLOOR
    );
    let pi = pi_zero_destination();
    let floor = CurrentPolicy::DEFAULT_FLOOR;
    let oracle = worked_policy_cycle(&pi, &CurrentPolicy::current(floor));
    let (bank, result) = native_worked_step(&pi, floor);
    assert_native_matches_policy("default floor", &bank, &result, &oracle);
    let combined = oracle.combined.as_ref().expect("cycle updated");
    let (mu, mu_oracle) = (&result.mode_probabilities, &oracle.mode_probabilities);
    assert_pinned(
        "μ⁺",
        mu.as_slice(),
        mu_oracle.as_slice(),
        &MU_DEFAULT_FLOOR_PINNED,
    );
    let vz = [VZ_DEFAULT_FLOOR_PINNED];
    assert_pinned("vz", &[result.state[5]], &[combined.mean[5]], &vz);
    let var_vz = [VAR_VZ_DEFAULT_FLOOR_PINNED];
    let (p, p_oracle) = (&result.covariance, &combined.cov);
    assert_pinned("var vz", &[p[(5, 5)]], &[p_oracle[(5, 5)]], &var_vz);
    // Exact: floor 0 with the zero-probability mode excluded.
    let exact = specified_cycle(
        &oracle_cv_ca_modes(1.0),
        &worked_cv_ca_leaves(),
        &mu_worked(),
        &pi,
        &worked_measurement(),
    );
    let kind = Residual::OracleVsHand;
    assert_close(
        kind,
        "exact vz",
        exact.combined.mean[5],
        zero_destination_exact_mean()[5],
    );
    assert_close(
        kind,
        "exact var vz",
        exact.combined.cov[(5, 5)],
        VAR_VZ_EXACT,
    );
}

#[test]
fn current_policy_f4_mixing_threshold_zeroes_negligible_destination() {
    // Policy: mixing computes μ_i|j only for c_j > 1e-30; Π = I and
    // μ = [1, 1e-40] give c_2 = 1e-40, so mode 2's mixed moments are zero.
    let common = Gaussian::new(&[50.0, 1.0, 20.0, 2.0, 300.0, 0.0], DMatrix::identity(6, 6));
    let ca_state = [50.0, 1.0, 0.0, 20.0, 2.0, 0.0, 300.0, 0.0, 0.0];
    let leaves = vec![
        common.clone(),
        Gaussian::new(&ca_state, DMatrix::identity(9, 9)),
    ];
    let mu = DVector::from_column_slice(&[1.0, 1e-40]);
    let mut bank = native_cv_ca_bank(&pi_identity(), &mu_worked(), 0.0);
    set_native_leaves(&mut bank, &leaves);
    bank.mode_probabilities = mu.clone();
    bank.interaction_step();
    let mode_2 = LinearMode::ca(1.0, 1.0).marginal(&native_leaf(&bank, 1));
    let zero = Gaussian::new(&[0.0; 6], DMatrix::zeros(6, 6));
    assert_gaussian_close(Residual::NativeVsPinned, "mode 2 mixed", &mode_2, &zero);
    let policy = CurrentPolicy::current(0.0);
    let oracle = policy_cycle(
        &oracle_cv_ca_modes(1.0),
        &leaves,
        &mu,
        &pi_identity(),
        None,
        &policy,
    );
    assert_gaussian_close(
        Residual::OracleVsPinned,
        "oracle mode 2 mixed",
        &oracle.mixed_common[1],
        &zero,
    );
    assert_gaussian_close(
        Residual::NativeVsOracle,
        "mode 2 leaf",
        &native_leaf(&bank, 1),
        &oracle.mixed_native[1],
    );
    // Exact formula: μ_2|2 = 1, so x0_2 is mode 2's own common estimate.
    let c = predicted_mode_probabilities(&pi_identity(), &mu);
    let weights = mixing_weights(&pi_identity(), &mu, &c);
    let commons = [common.clone(), common.clone()];
    let formula = mix_destination(&commons, &weights, 1).estimate;
    assert_gaussian_close(
        Residual::OracleVsHand,
        "formula mode 2 mixed",
        &formula,
        &common,
    );
}

#[test]
fn current_policy_f4_far_measurement_zero_destination_takes_posterior() {
    // Policy: zero-destination assignment (mode 2's mixed moments zero) and
    // the 1e-300 clamp; floor 0. Mode 2's zeroed estimate sits at the
    // origin next to the measurement, so its log-likelihood exceeds mode 1's
    // by more than −ln(1e-300) ≈ 690.8 nats and it captures the posterior.
    let p = DMatrix::identity(6, 6);
    let ca_state = [10000.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1000.0, 0.0, 0.0];
    let leaves = vec![
        Gaussian::new(&[10000.0, 0.0, 0.0, 0.0, 1000.0, 0.0], p),
        Gaussian::new(&ca_state, DMatrix::identity(9, 9)),
    ];
    let (pi, m) = (
        pi_zero_destination(),
        position_measurement(&[0.0, 0.0, 0.0]),
    );
    let mut bank = native_cv_ca_bank(&pi, &mu_worked(), 0.0);
    set_native_leaves(&mut bank, &leaves);
    let result = native_step(&mut bank, 1.0, &m);
    let policy = CurrentPolicy::current(0.0);
    let oracle = policy_cycle(
        &oracle_cv_ca_modes(1.0),
        &leaves,
        &mu_worked(),
        &pi,
        Some(&m),
        &policy,
    );
    assert_native_matches_policy("far measurement", &bank, &result, &oracle);
    let mu = result.mode_probabilities.as_slice();
    assert_pinned(
        "far μ⁺",
        mu,
        oracle.mode_probabilities.as_slice(),
        &[0.0, 1.0],
    );
    let lls = oracle.log_likelihoods();
    assert!(
        lls[1] - lls[0] > -(1e-300_f64).ln(),
        "log-likelihood gap {}",
        lls[1] - lls[0]
    );
    // Exact: the zero-probability mode gets zero posterior weight.
    let exact = specified_cycle(&oracle_cv_ca_modes(1.0), &leaves, &mu_worked(), &pi, &m);
    assert_eq!(exact.posterior_mu.as_slice(), &[1.0, 0.0]);
}

// ---------------------------------------------------------------------------
// F5: coasting
// ---------------------------------------------------------------------------

/// Oracle: one coasted cycle, then an updated cycle, under `policy`.
fn coast_then_update(policy: &CurrentPolicy) -> PolicyCycle {
    let modes = oracle_cv_ca_modes(1.0);
    let pi = pi_asymmetric();
    let first = policy_cycle(
        &modes,
        &worked_cv_ca_leaves(),
        &mu_worked(),
        &pi,
        None,
        policy,
    );
    let m = worked_measurement();
    policy_cycle(
        &modes,
        &first.leaves,
        &first.mode_probabilities,
        &pi,
        Some(&m),
        policy,
    )
}

#[test]
fn current_policy_f5_coast_keeps_stale_mode_probabilities() {
    // Policy: a prediction without an update leaves μ unchanged; floor 0.
    let mut bank = native_cv_ca_bank(&pi_asymmetric(), &mu_worked(), 0.0);
    bank.predict(1.0);
    let stale = bank.mode_probabilities.clone();
    assert_vector_close(
        Residual::NativeVsPinned,
        "μ after coast",
        &stale,
        &mu_worked(),
    );
    bank.predict(1.0);
    let m = worked_measurement();
    let result = bank.update_with_measurement(&m.z, &m.h, &m.r);
    let oracle = coast_then_update(&CurrentPolicy::current(0.0));
    assert_native_matches_policy("coast", &bank, &result, &oracle);
    let mu = result.mode_probabilities.as_slice();
    let mu_oracle = oracle.mode_probabilities.as_slice();
    assert_pinned("coast μ⁺", mu, mu_oracle, &MU_COAST_PINNED);
    // Textbook: propagate μ ← Πᵀμ on the coast.
    let propagate = CurrentPolicy {
        coast: CoastPolicy::PropagateModeProbabilities,
        ..CurrentPolicy::current(0.0)
    };
    let textbook = coast_then_update(&propagate).mode_probabilities;
    assert_slice_close(
        Residual::OracleVsHand,
        "propagated μ⁺",
        textbook.as_slice(),
        &MU_COAST_PROPAGATED,
    );
}

// ---------------------------------------------------------------------------
// F6: predicted output
// ---------------------------------------------------------------------------

#[test]
fn current_policy_f6_predicted_output_uses_prior_mode_probabilities() {
    // Policy: `predict()` returns the μ-weighted predicted combination.
    let mut bank = native_cv_ca_bank(&pi_asymmetric(), &mu_worked(), 0.0);
    let (mean, cov) = bank.predict(1.0);
    let modes = oracle_cv_ca_modes(1.0);
    let policy = CurrentPolicy::current(0.0);
    let prior = policy_cycle(
        &modes,
        &worked_cv_ca_leaves(),
        &mu_worked(),
        &pi_asymmetric(),
        None,
        &policy,
    );
    let x = [PREDICTED_X_PINNED];
    assert_pinned(
        "predicted x",
        &[mean[0]],
        &[prior.predicted_output.mean[0]],
        &x,
    );
    let predicted = Gaussian { mean, cov };
    assert_gaussian_close(
        Residual::NativeVsOracle,
        "predicted output",
        &predicted,
        &prior.predicted_output,
    );
    // Textbook: c-weighted output, exactly [58/5, 54/5, 21/5, 23/5, 502/5, 0].
    let c_weighted = CurrentPolicy {
        predicted_output: PredictedOutputWeighting::PredictedModeProbabilities,
        ..policy
    };
    let textbook = policy_cycle(
        &modes,
        &worked_cv_ca_leaves(),
        &mu_worked(),
        &pi_asymmetric(),
        None,
        &c_weighted,
    );
    let exact = rational_vector(&[58, 54, 21, 23, 502, 0], 5);
    assert_vector_close(
        Residual::OracleVsHand,
        "c-weighted predicted mean",
        &textbook.predicted_output.mean,
        &exact,
    );
}

// ---------------------------------------------------------------------------
// F7: transition matrix and elapsed time
// ---------------------------------------------------------------------------

#[test]
fn current_policy_f7_predicted_probabilities_ignore_elapsed_time() {
    // Policy: Π is applied as configured for every elapsed time. Two
    // identical CV modes holding the same estimate have equal likelihoods,
    // so with floor 0 the posterior equals c and exposes it natively.
    let pi = pi_asymmetric();
    let leaves = [worked_cv_leaf(), worked_cv_leaf()];
    for dt in ELAPSED_TIMES {
        let mut bank = native_cv_bank(&[1.0, 1.0], &pi, &mu_worked(), &leaves, 0.0);
        let z = measurement_sequence(dt).remove(0);
        let mu = native_step(&mut bank, dt, &z).mode_probabilities;
        let label = format!("c at dt {dt}");
        assert_slice_close(
            Residual::NativeVsPinned,
            &label,
            mu.as_slice(),
            &C_ASYMMETRIC_HELD,
        );
        // Textbook: the rate-consistent Π(dt) = Π^dt gives c₁ = 2/3 − 0.7^dt/15,
        // which equals the held value only at dt = 1.
        let rate = rate_consistent_c1(&pi, &mu_worked(), dt);
        let agrees = (rate - C_ASYMMETRIC_HELD[0]).abs() < 1e-12;
        assert_eq!(
            agrees,
            (dt - 1.0).abs() < 1e-12,
            "{label}: rate-consistent c₁ {rate}"
        );
    }
}

/// CV+CV end-to-end runs at dt 0.1 and 2.3 match the oracle with Π held at
/// its configured value; the rate-consistent `c₁` of the first cycle differs.
fn check_off_unit_dt_runs(label: &str, pi: &DMatrix<f64>) {
    let held = predicted_mode_probabilities(pi, &mu_worked());
    for dt in [0.1, 2.3] {
        let run = format!("{label} dt {dt}");
        let records = run_cv_cv_end_to_end(&run, pi, dt, &measurement_sequence(dt));
        assert_eq!(records.len(), END_TO_END_CYCLES);
        assert_vector_close(
            Residual::OracleVsPinned,
            &format!("{run} held c"),
            &records[0].oracle.c,
            &held,
        );
        let rate = rate_consistent_c1(pi, &mu_worked(), dt);
        assert!(
            (rate - held[0]).abs() > 1e-3,
            "{run}: rate-consistent c₁ {rate} vs held {}",
            held[0]
        );
    }
}

#[test]
fn current_policy_f7_cv_cv_symmetric_runs_off_unit_dt() {
    check_off_unit_dt_runs("symmetric", &pi_symmetric());
}

#[test]
fn current_policy_f7_cv_cv_asymmetric_runs_off_unit_dt() {
    check_off_unit_dt_runs("asymmetric", &pi_asymmetric());
}
