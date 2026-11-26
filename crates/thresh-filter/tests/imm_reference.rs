//! Specified-behaviour verification of the native IMM (`imm-estimation`).
//!
//! Evidence, in order:
//! 1. the oracle's phase unit tests and its source-independence check;
//! 2. hand-derived exact rationals, reproduced by the oracle;
//! 3. posteriors, combinations and elapsed-time log-likelihoods from given
//!    inputs and closed forms;
//! 4. native conformance against the oracle and the hand-derived values;
//! 5. configuration validation (V).
//!
//! Scope: every native test here uses a configuration whose outputs depend on
//! no listed unspecified policy — floor 0; every predicted probability above
//! 1e-12 at every asserted cycle, except the declared zero-destination case;
//! no dependence on CA reconstruction; no coasting; and a non-identity
//! transition matrix only at dt = 1. Behaviour under the unspecified
//! policies is pinned in the separate current-policy characterisation target,
//! which this target never references.
//!
//! Tolerances: absolute 1e-10 plus relative 1e-9; probability sums within
//! 1e-12.

#[allow(dead_code)]
mod support;

use nalgebra::{DMatrix, DVector, SymmetricEigen};
use thresh_filter::imm::{ImmConfig, ImmFilter, ImmLeafKind};

use support::imm_fixtures::{
    ATOL, CycleRecord, ELAPSED_TIMES, END_TO_END_CYCLES, PROBABILITY_SUM_TOL, RTOL, Residual,
    SIGMA_A_MODE_1, SIGMA_A_MODE_2, assert_close, assert_gaussian_close, assert_matrix_close,
    assert_predicted_probabilities_above_bound, assert_probabilities_sum_to_one,
    assert_slice_close, assert_vector_close, measurement_sequence, mu_worked, native_cv_bank,
    native_cv_ca_bank, native_cv_cv_bank, native_leaf, native_step, oracle_cv_ca_modes,
    oracle_cv_cv_modes, pi_asymmetric, pi_identity, pi_symmetric, pi_zero_destination, ratio,
    rational_vector, run_cv_cv_end_to_end, underflow_sequence, worked_ca_common, worked_ca_leaf,
    worked_cv_ca_leaves, worked_cv_cv_leaves, worked_cv_leaf, worked_measurement,
    zero_destination_exact_mean,
};
use support::imm_oracle::{
    Gaussian, LinearMode, Measurement, MixedEstimate, ca_common_selector, ca_process_noise,
    ca_transition, combine, common_marginals, cv_process_noise, cv_transition,
    gaussian_log_likelihood, joseph_update, log_det_spd, mix_common, mix_destination,
    mixing_weights, mode_posterior, predict_common, predicted_mode_probabilities, specified_cycle,
};

/// Worked CV log-likelihood ℓ_CV (`imm-estimation`, worked constant-velocity mode).
const LL_CV_WORKED: f64 = -6.461143273565262;
/// Worked second-mode log-likelihood used by the given-likelihood posterior scenario.
const LL_MODE_2_WORKED: f64 = -8.519306179549595;
/// Worked posterior `μ⁺` from `c = [0.62, 0.38]` and the worked log-likelihoods.
const MU_POSTERIOR_WORKED: [f64; 2] = [0.9274195958363679, 0.07258040416363172];
/// Underflow log-likelihoods (`z_x = 2011`); both linear likelihoods underflow.
const LL_UNDERFLOW: [f64; 2] = [-163428.31075105123, -44784.431767829774];

// ---------------------------------------------------------------------------
// Shared expectations
// ---------------------------------------------------------------------------

/// A two-mode mixing case with exact-rational `c` and `μ_i|j`.
struct TwoModeMixing {
    pi: DMatrix<f64>,
    c: [f64; 2],
    /// `weights[i][j] = μ_i|j`.
    weights: [[f64; 2]; 2],
}

fn asymmetric_mixing() -> TwoModeMixing {
    TwoModeMixing {
        pi: pi_asymmetric(),
        c: [ratio(31, 50), ratio(19, 50)],
        weights: [[ratio(27, 31), ratio(3, 19)], [ratio(4, 31), ratio(16, 19)]],
    }
}

fn identity_mixing() -> TwoModeMixing {
    TwoModeMixing {
        pi: pi_identity(),
        c: [ratio(3, 5), ratio(2, 5)],
        weights: [[1.0, 0.0], [0.0, 1.0]],
    }
}

fn symmetric_mixing() -> TwoModeMixing {
    TwoModeMixing {
        pi: pi_symmetric(),
        c: [ratio(59, 100), ratio(41, 100)],
        weights: [[ratio(57, 59), ratio(3, 41)], [ratio(2, 59), ratio(38, 41)]],
    }
}

/// Closed-form two-mode mixing of the worked common estimates for
/// destination `j`: `x0 = w₁x₁ + w₂x₂`, spread `w₁w₂ΔΔᵀ` with `Δ = x₂ − x₁`,
/// and `P0 = w₁P₁ + w₂P₂ + spread`.
fn exact_mixed(case: &TwoModeMixing, j: usize) -> MixedEstimate {
    let (a, b) = (worked_cv_leaf(), worked_ca_common());
    let (w1, w2) = (case.weights[0][j], case.weights[1][j]);
    let delta = &b.mean - &a.mean;
    let spread = (&delta * delta.transpose()) * (w1 * w2);
    MixedEstimate {
        estimate: Gaussian {
            mean: &a.mean * w1 + &b.mean * w2,
            cov: &a.cov * w1 + &b.cov * w2 + &spread,
        },
        spread,
    }
}

fn assert_mixed_close(
    kind: Residual,
    label: &str,
    actual: &MixedEstimate,
    expected: &MixedEstimate,
) {
    assert_gaussian_close(kind, label, &actual.estimate, &expected.estimate);
    assert_matrix_close(
        kind,
        &format!("{label} spread"),
        &actual.spread,
        &expected.spread,
    );
}

/// Oracle `c`, `μ_i|j` and mixed moments for the worked CV+CA estimates
/// (the CA mode contributes its common marginal).
fn oracle_worked_mixing(pi: &DMatrix<f64>) -> (DVector<f64>, DMatrix<f64>, Vec<MixedEstimate>) {
    let commons = common_marginals(&oracle_cv_ca_modes(1.0), &worked_cv_ca_leaves());
    let c = predicted_mode_probabilities(pi, &mu_worked());
    let weights = mixing_weights(pi, &mu_worked(), &c);
    let mixed = (0..2)
        .map(|j| mix_destination(&commons, &weights, j))
        .collect();
    (c, weights, mixed)
}

fn check_oracle_two_mode_mixing(label: &str, case: &TwoModeMixing) {
    let kind = Residual::OracleVsHand;
    let (c, weights, mixed) = oracle_worked_mixing(&case.pi);
    assert_slice_close(kind, &format!("{label} c"), c.as_slice(), &case.c);
    let expected_weights = DMatrix::from_fn(2, 2, |i, j| case.weights[i][j]);
    assert_matrix_close(kind, &format!("{label} μ_i|j"), &weights, &expected_weights);
    for (j, m) in mixed.iter().enumerate() {
        assert_mixed_close(
            kind,
            &format!("{label} mixed {j}"),
            m,
            &exact_mixed(case, j),
        );
    }
}

/// Closed-form CV log-likelihood for an axis-separable estimate and diagonal
/// `R`, from the DWNA matrices: per axis
/// `S_a = P_pp + 2 dt P_pv + dt² P_vv + σ_a² dt⁴/4 + r_a` and
/// `y_a = z_a − (x_p + dt x_v)`, so `ℓ = −½ Σ_a (ln 2π + ln S_a + y_a²/S_a)`.
fn closed_form_cv_log_likelihood(est: &Gaussian, sigma_a: f64, dt: f64, m: &Measurement) -> f64 {
    let axis_of = |k: usize| k / 2;
    for (r, c) in (0..6).flat_map(|r| (0..6).map(move |c| (r, c))) {
        assert!(
            axis_of(r) == axis_of(c) || est.cov[(r, c)] == 0.0,
            "closed form needs an axis-separable estimate"
        );
    }
    (0..3)
        .map(|a| {
            let (p, v) = (2 * a, 2 * a + 1);
            let s = est.cov[(p, p)]
                + 2.0 * dt * est.cov[(p, v)]
                + dt * dt * est.cov[(v, v)]
                + sigma_a * sigma_a * dt.powi(4) / 4.0
                + m.r[(a, a)];
            let y = m.z[a] - (est.mean[p] + dt * est.mean[v]);
            -0.5 * ((2.0 * std::f64::consts::PI).ln() + s.ln() + y * y / s)
        })
        .sum()
}

/// Expected per-axis matrix: `block` on each axis, zero across axes.
fn per_axis_expected(block: &[f64], b: usize) -> DMatrix<f64> {
    DMatrix::from_fn(3 * b, 3 * b, |r, c| {
        if r / b == c / b {
            block[(r % b) * b + c % b]
        } else {
            0.0
        }
    })
}

/// The 1-D estimate `N(mean, var)`.
fn scalar(mean: f64, var: f64) -> Gaussian {
    Gaussian::new(&[mean], DMatrix::from_element(1, 1, var))
}

// ---------------------------------------------------------------------------
// 1. Oracle independence and phase unit tests
// ---------------------------------------------------------------------------

#[test]
fn oracle_source_imports_only_nalgebra_and_std() {
    let src = include_str!("support/imm_oracle.rs");
    for crate_path in ["thresh_", "thresh-", "thresh::"] {
        assert!(
            !src.contains(crate_path),
            "the oracle must not name a production crate ({crate_path})"
        );
    }
    assert!(!src.contains("crate::") && !src.contains("super::") && !src.contains("extern crate"));
    for line in src
        .lines()
        .map(str::trim_start)
        .filter(|l| l.starts_with("use "))
    {
        assert!(
            line.starts_with("use nalgebra::") || line.starts_with("use std::"),
            "oracle import outside nalgebra/std: {line}"
        );
    }
}

#[test]
fn oracle_phase_cv_model_matches_closed_form() {
    // dt = 2, σ_a = 3: F per axis [[1, 2], [0, 1]]; Q per axis
    // 9 · [[16/4, 8/2], [8/2, 4]] = [[36, 36], [36, 36]].
    let kind = Residual::OracleVsHand;
    let f = per_axis_expected(&[1.0, 2.0, 0.0, 1.0], 2);
    let q = per_axis_expected(&[36.0, 36.0, 36.0, 36.0], 2);
    assert_matrix_close(kind, "CV F", &cv_transition(2.0), &f);
    assert_matrix_close(kind, "CV Q", &cv_process_noise(2.0, 3.0), &q);
    assert!(LinearMode::cv(2.0, 3.0).is_common());
}

#[test]
fn oracle_phase_ca_model_matches_closed_form() {
    // dt = 2, σ_j = 1: F per axis [[1, 2, 2], [0, 1, 2], [0, 0, 1]]; Q per axis
    // [[32/20, 16/8, 8/6], [16/8, 8/3, 4/2], [8/6, 4/2, 2]].
    let kind = Residual::OracleVsHand;
    let f = per_axis_expected(&[1.0, 2.0, 2.0, 0.0, 1.0, 2.0, 0.0, 0.0, 1.0], 3);
    #[rustfmt::skip]
    let q = per_axis_expected(&[
        1.6, 2.0, ratio(4, 3),
        2.0, ratio(8, 3), 2.0,
        ratio(4, 3), 2.0, 2.0,
    ], 3);
    assert_matrix_close(kind, "CA F", &ca_transition(2.0), &f);
    assert_matrix_close(kind, "CA Q", &ca_process_noise(2.0, 1.0), &q);
    assert!(!LinearMode::ca(2.0, 1.0).is_common());
}

#[test]
fn oracle_phase_ca_marginal_selects_common_components() {
    // Selector [0, 1, 3, 4, 6, 7]: the acceleration entries and the vx–ax
    // covariance 0.5 are dropped; the x–vx covariance 1 survives.
    let selector = ca_common_selector();
    for (row, col) in [(0, 0), (1, 1), (2, 3), (3, 4), (4, 6), (5, 7)] {
        assert_eq!(selector[(row, col)], 1.0);
    }
    assert_eq!(selector.sum(), 6.0);
    let marginal = LinearMode::ca(1.0, 1.0).marginal(&worked_ca_leaf());
    assert_gaussian_close(
        Residual::OracleVsHand,
        "CA marginal",
        &marginal,
        &worked_ca_common(),
    );
}

#[test]
fn oracle_phase_joseph_update_matches_scalar_kalman() {
    // Prior N(0, 4), z = 2, R = 4: S = 8, K = 1/2, x⁺ = 1, P⁺ = 2.
    let kind = Residual::OracleVsHand;
    let m = Measurement {
        z: DVector::from_element(1, 2.0),
        h: DMatrix::from_element(1, 1, 1.0),
        r: DMatrix::from_element(1, 1, 4.0),
    };
    let u = joseph_update(&scalar(0.0, 4.0), &m);
    assert_close(kind, "scalar x⁺", u.posterior.mean[0], 1.0);
    assert_close(kind, "scalar P⁺", u.posterior.cov[(0, 0)], 2.0);
    assert_close(kind, "scalar S", u.innovation_cov[(0, 0)], 8.0);
    let ll = -0.5 * ((2.0 * std::f64::consts::PI).ln() + 8.0_f64.ln() + 0.5);
    assert_close(kind, "scalar ℓ", u.log_likelihood, ll);
}

#[test]
fn oracle_phase_joseph_update_matches_correlated_closed_form() {
    // Prior x = [0, 0], P = [[4, 1], [1, 2]], H = [1, 0], R = 1, z = 5:
    // S = 5, K = [4/5, 1/5], x⁺ = [4, 1], P⁺ = P − K S Kᵀ = [[4/5, 1/5], [1/5, 9/5]].
    let kind = Residual::OracleVsHand;
    let prior = Gaussian::new(
        &[0.0, 0.0],
        DMatrix::from_row_slice(2, 2, &[4.0, 1.0, 1.0, 2.0]),
    );
    let m = Measurement {
        z: DVector::from_element(1, 5.0),
        h: DMatrix::from_row_slice(1, 2, &[1.0, 0.0]),
        r: DMatrix::from_element(1, 1, 1.0),
    };
    let u = joseph_update(&prior, &m);
    let expected = Gaussian::new(
        &[4.0, 1.0],
        DMatrix::from_row_slice(2, 2, &[0.8, 0.2, 0.2, 1.8]),
    );
    assert_gaussian_close(kind, "correlated posterior", &u.posterior, &expected);
}

#[test]
fn oracle_phase_log_likelihood_matches_diagonal_closed_form() {
    // y = [1, 2, 3], S = diag(2, 4, 8): ℓ = −½ (3 ln 2π + ln 64 + 1/2 + 1 + 9/8).
    let y = DVector::from_column_slice(&[1.0, 2.0, 3.0]);
    let s = DMatrix::from_diagonal(&DVector::from_column_slice(&[2.0, 4.0, 8.0]));
    let expected = -0.5 * (3.0 * (2.0 * std::f64::consts::PI).ln() + 64.0_f64.ln() + 2.625);
    let kind = Residual::OracleVsHand;
    assert_close(
        kind,
        "diagonal ℓ",
        gaussian_log_likelihood(&y, &s),
        expected,
    );
    assert_close(kind, "diagonal ln det", log_det_spd(&s), 64.0_f64.ln());
}

#[test]
fn oracle_phase_posterior_gives_zero_probability_mode_exact_zero() {
    // The exact posterior ignores a zero-probability mode's likelihood,
    // however large, with no clamp.
    let c = DVector::from_column_slice(&[1.0, 0.0]);
    let mu = mode_posterior(&c, &[-10.0, 1000.0]);
    assert_eq!(mu.as_slice(), &[1.0, 0.0]);
}

// ---------------------------------------------------------------------------
// 2. Hand-derived exact rationals (oracle)
// ---------------------------------------------------------------------------

#[test]
fn oracle_asymmetric_mixing_matches_exact_rationals() {
    let case = asymmetric_mixing();
    check_oracle_two_mode_mixing("asymmetric", &case);
    // The spec's explicit fractions.
    let kind = Residual::OracleVsHand;
    let (_, _, mixed) = oracle_worked_mixing(&case.pi);
    let x01 = rational_vector(&[8, 318, -4, 151, 3104, 0], 31);
    let x02 = rational_vector(&[32, 222, -16, 79, 1916, 0], 19);
    assert_vector_close(kind, "x0_1", &mixed[0].estimate.mean, &x01);
    assert_vector_close(kind, "x0_2", &mixed[1].estimate.mean, &x02);
    let delta = DVector::from_column_slice(&[2.0, 2.0, -1.0, -1.0, 1.0, 0.0]);
    let ddt = &delta * delta.transpose();
    assert_matrix_close(
        kind,
        "spread 1",
        &mixed[0].spread,
        &(&ddt * ratio(108, 961)),
    );
    assert_matrix_close(kind, "spread 2", &mixed[1].spread, &(&ddt * ratio(48, 361)));
    let (p01, p02) = (&mixed[0].estimate.cov, &mixed[1].estimate.cov);
    assert_close(kind, "P0_1[x,x]", p01[(0, 0)], ratio(4896, 961));
    assert_close(kind, "P0_1[x,vx]", p01[(0, 1)], ratio(556, 961));
    assert_close(kind, "P0_2[x,x]", p02[(0, 0)], ratio(3156, 361));
    assert_close(kind, "P0_2[x,vx]", p02[(0, 1)], ratio(496, 361));
}

#[test]
fn oracle_identity_mixing_matches_exact_rationals() {
    let case = identity_mixing();
    check_oracle_two_mode_mixing("identity", &case);
    // Each mixed estimate is the mode's own estimate, with zero spread.
    let (_, _, mixed) = oracle_worked_mixing(&case.pi);
    let kind = Residual::OracleVsHand;
    assert_gaussian_close(
        kind,
        "identity own 1",
        &mixed[0].estimate,
        &worked_cv_leaf(),
    );
    assert_gaussian_close(
        kind,
        "identity own 2",
        &mixed[1].estimate,
        &worked_ca_common(),
    );
    assert_eq!(mixed[0].spread.abs().max(), 0.0);
    assert_eq!(mixed[1].spread.abs().max(), 0.0);
}

#[test]
fn oracle_symmetric_mixing_matches_exact_rationals() {
    check_oracle_two_mode_mixing("symmetric", &symmetric_mixing());
}

#[test]
fn oracle_worked_cv_prediction_matches_exact_rationals() {
    let kind = Residual::OracleVsHand;
    let (_, _, mixed) = oracle_worked_mixing(&pi_asymmetric());
    let predicted = predict_common(&LinearMode::cv(1.0, SIGMA_A_MODE_1), &mixed[0].estimate);
    let expected_mean = rational_vector(&[326, 318, 147, 151, 3104, 0], 31);
    assert_vector_close(kind, "predicted mean", &predicted.mean, &expected_mean);
    let u = joseph_update(&predicted, &worked_measurement());
    let innovation = DVector::from_column_slice(&[ratio(15, 31), ratio(39, 31), ratio(23, 62)]);
    assert_vector_close(kind, "innovation", &u.innovation, &innovation);
    let det_s = log_det_spd(&u.innovation_cov).exp();
    assert_close(kind, "det S", det_s, ratio(80701246125, 59105344));
    assert_close(kind, "ℓ_CV", u.log_likelihood, LL_CV_WORKED);
}

#[test]
fn oracle_zero_destination_combined_mean_matches_exact_rationals() {
    let kind = Residual::OracleVsHand;
    let cycle = specified_cycle(
        &oracle_cv_ca_modes(1.0),
        &worked_cv_ca_leaves(),
        &mu_worked(),
        &pi_zero_destination(),
        &worked_measurement(),
    );
    assert_eq!(cycle.c.as_slice(), &[1.0, 0.0]);
    assert!(
        cycle.mixed[1].is_none(),
        "zero-probability mode is excluded"
    );
    assert_eq!(cycle.posterior_mu.as_slice(), &[1.0, 0.0]);
    let expected = zero_destination_exact_mean();
    assert_vector_close(
        kind,
        "zero-destination combined mean",
        &cycle.combined.mean,
        &expected,
    );
    let mode_1 = &cycle.updates[0].as_ref().expect("mode 1 active").posterior;
    assert_gaussian_close(kind, "combined = mode 1 posterior", &cycle.combined, mode_1);
}

// ---------------------------------------------------------------------------
// 3. Given log-likelihoods, given moments and elapsed time (oracle)
// ---------------------------------------------------------------------------

#[test]
fn oracle_worked_posterior_matches_given_log_likelihoods() {
    let c = DVector::from_column_slice(&[0.62, 0.38]);
    let mu = mode_posterior(&c, &[LL_CV_WORKED, LL_MODE_2_WORKED]);
    assert_probabilities_sum_to_one("worked μ⁺", &mu);
    assert_slice_close(
        Residual::OracleVsHand,
        "worked μ⁺",
        mu.as_slice(),
        &MU_POSTERIOR_WORKED,
    );
}

#[test]
fn oracle_posterior_is_exact_for_extreme_log_likelihoods() {
    // Float64 spacing near 1e16 is 2, so adding ln c_j directly to ℓ_j would
    // round the priors away. The exact posterior is ∝ [0.6, 0.4 e^-2].
    let c = DVector::from_column_slice(&[0.6, 0.4]);
    let mu = mode_posterior(&c, &[-1e16, -1e16 - 2.0]);
    let w2 = 0.4 * (-2.0_f64).exp();
    let expected = [0.6 / (0.6 + w2), w2 / (0.6 + w2)];
    assert_probabilities_sum_to_one("extreme-ℓ μ⁺", &mu);
    assert_slice_close(
        Residual::OracleVsHand,
        "extreme-ℓ μ⁺",
        mu.as_slice(),
        &expected,
    );
}

#[test]
fn oracle_log_space_underflow_gives_zero_and_one() {
    assert!(
        LL_UNDERFLOW.iter().all(|ll| ll.exp() == 0.0),
        "both linear likelihoods must underflow"
    );
    let c = DVector::from_column_slice(&[0.62, 0.38]);
    let mu = mode_posterior(&c, &LL_UNDERFLOW);
    assert_eq!(
        mu.as_slice(),
        &[0.0, 1.0],
        "no NaN, no uniform, no reversion to c"
    );
}

#[test]
fn oracle_worked_combination_matches_given_moments() {
    let a = scalar(10.812892420810806, 2.6933736334545353);
    let b = scalar(11.206007753504345, 3.6409686692927807);
    let combined = combine(&[&a, &b], &MU_POSTERIOR_WORKED);
    let kind = Residual::OracleVsHand;
    assert_close(kind, "combined x", combined.mean[0], 10.84142489054062);
    assert_close(
        kind,
        "combined var x",
        combined.cov[(0, 0)],
        2.7725529136406477,
    );
}

#[test]
fn oracle_elapsed_time_log_likelihoods_match_closed_form() {
    // One fixed mixed estimate (mode 2's identity-mixed common estimate),
    // predicted by each CV mode over 0.1 s, 1 s and 2.3 s.
    let fixed = worked_ca_common();
    let m = worked_measurement();
    for dt in ELAPSED_TIMES {
        for sigma in [SIGMA_A_MODE_1, SIGMA_A_MODE_2] {
            let predicted = predict_common(&LinearMode::cv(dt, sigma), &fixed);
            let ll = joseph_update(&predicted, &m).log_likelihood;
            let expected = closed_form_cv_log_likelihood(&fixed, sigma, dt, &m);
            assert_close(
                Residual::OracleVsHand,
                &format!("ℓ dt {dt} σ {sigma}"),
                ll,
                expected,
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Mixing invariants (shared by the oracle and native checks)
// ---------------------------------------------------------------------------

/// A three-mode CV bank. Row 2 of `Π` sums to 1 + 4e-7, inside the 1e-6
/// validation tolerance, so the general form of the mean invariant is used.
struct ThreeModeCase {
    sigmas: [f64; 3],
    pi: DMatrix<f64>,
    mu: DVector<f64>,
    leaves: Vec<Gaussian>,
}

fn three_mode_case() -> ThreeModeCase {
    let mut p3 =
        DMatrix::from_diagonal(&DVector::from_column_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]));
    p3[(0, 2)] = 0.5;
    p3[(2, 0)] = 0.5;
    #[rustfmt::skip]
    let pi = DMatrix::from_row_slice(3, 3, &[
        0.8, 0.15, 0.05,
        0.1, 0.7, 0.2000004,
        0.25, 0.25, 0.5,
    ]);
    ThreeModeCase {
        sigmas: [1.0, 2.0, 3.0],
        pi,
        mu: DVector::from_column_slice(&[0.5, 0.3, 0.2]),
        leaves: vec![
            worked_cv_leaf(),
            worked_ca_common(),
            Gaussian::new(&[-1.0, 9.0, 2.0, 6.0, 99.0, 1.0], p3),
        ],
    }
}

/// Each column of `μ_i|j` sums to one within 1e-12.
fn assert_mixing_columns_sum_to_one(label: &str, weights: &DMatrix<f64>) {
    for (j, col) in weights.column_iter().enumerate() {
        let sum = col.sum();
        assert!(
            (sum - 1.0).abs() <= PROBABILITY_SUM_TOL,
            "{label}: column {j} sums to {sum:e}"
        );
    }
}

/// `Σ_j c_j x0_j = Σ_i (Σ_j π_ij) μ_i x_i`, componentwise within
/// `ATOL + RTOL · Σ_i (Σ_j π_ij) μ_i |x_i|`. The relative term scales with
/// the magnitude of the summed terms, not with the result, so cancellation
/// cannot shrink the bound (spec scenario "Mixing invariants").
fn assert_mean_invariant(
    label: &str,
    pi: &DMatrix<f64>,
    mu: &DVector<f64>,
    commons: &[Gaussian],
    mixed_means: &[DVector<f64>],
) {
    let c = predicted_mode_probabilities(pi, mu);
    let lhs = mixed_means
        .iter()
        .enumerate()
        .fold(DVector::zeros(6), |acc, (j, x)| acc + x * c[j]);
    let (rhs, magnitude) = commons.iter().enumerate().fold(
        (DVector::zeros(6), DVector::zeros(6)),
        |(sum, mag), (i, g)| {
            let w = pi.row(i).sum() * mu[i];
            (sum + &g.mean * w, mag + g.mean.abs() * w)
        },
    );
    for k in 0..6 {
        let diff = (lhs[k] - rhs[k]).abs();
        assert!(
            diff.is_finite() && diff <= ATOL + RTOL * magnitude[k],
            "{label} Σ c x0 component {k}: |Δ| = {diff:e}, scale {:e}",
            magnitude[k]
        );
    }
}

/// The spread `P0_j − Σ_i μ_i|j P_i` is positive semidefinite within tolerance.
fn assert_spread_psd(
    label: &str,
    mixed_cov: &DMatrix<f64>,
    commons: &[Gaussian],
    weights: &DMatrix<f64>,
    j: usize,
) {
    let own = commons
        .iter()
        .enumerate()
        .fold(DMatrix::zeros(6, 6), |acc, (i, g)| {
            acc + &g.cov * weights[(i, j)]
        });
    let spread = mixed_cov - own;
    let eig = SymmetricEigen::new(&spread + spread.transpose()).eigenvalues / 2.0;
    let scale = eig.abs().max();
    assert!(
        eig.min() >= -(ATOL + RTOL * scale),
        "{label}: spread of mode {j} has eigenvalue {:e}",
        eig.min()
    );
}

/// All three mixing invariants for one set of mixed estimates.
fn assert_mixing_invariants(
    label: &str,
    pi: &DMatrix<f64>,
    mu: &DVector<f64>,
    commons: &[Gaussian],
    mixed: &[Gaussian],
) {
    let c = predicted_mode_probabilities(pi, mu);
    assert_predicted_probabilities_above_bound(label, &c);
    let weights = mixing_weights(pi, mu, &c);
    assert_mixing_columns_sum_to_one(label, &weights);
    let means: Vec<DVector<f64>> = mixed.iter().map(|g| g.mean.clone()).collect();
    assert_mean_invariant(label, pi, mu, commons, &means);
    for (j, g) in mixed.iter().enumerate() {
        assert_spread_psd(label, &g.cov, commons, &weights, j);
    }
}

fn oracle_mixed(pi: &DMatrix<f64>, mu: &DVector<f64>, commons: &[Gaussian]) -> Vec<Gaussian> {
    let c = predicted_mode_probabilities(pi, mu);
    let weights = mixing_weights(pi, mu, &c);
    mix_common(commons, &weights, &c)
        .into_iter()
        .map(|m| m.expect("every c_j > 0").estimate)
        .collect()
}

#[test]
fn oracle_mixing_invariants_hold() {
    let commons = worked_cv_cv_leaves();
    for (label, pi) in [
        ("identity", pi_identity()),
        ("symmetric", pi_symmetric()),
        ("asymmetric", pi_asymmetric()),
    ] {
        let mixed = oracle_mixed(&pi, &mu_worked(), &commons);
        assert_mixing_invariants(label, &pi, &mu_worked(), &commons, &mixed);
    }
    let case = three_mode_case();
    let mixed = oracle_mixed(&case.pi, &case.mu, &case.leaves);
    assert_mixing_invariants("three-mode", &case.pi, &case.mu, &case.leaves, &mixed);
}

// ---------------------------------------------------------------------------
// 4. Native conformance
// ---------------------------------------------------------------------------

/// `interaction_step()` on a CV+CV bank: both leaves equal the oracle's and
/// the closed-form mixed moments.
fn check_native_cv_cv_interaction(label: &str, case: &TwoModeMixing) {
    let mut bank = native_cv_cv_bank(&case.pi, &mu_worked(), 0.0);
    bank.interaction_step();
    let (_, _, mixed) = oracle_worked_mixing(&case.pi);
    for (j, m) in mixed.iter().enumerate() {
        let leaf = native_leaf(&bank, j);
        let what = format!("{label} CV+CV leaf {j}");
        assert_gaussian_close(Residual::NativeVsOracle, &what, &leaf, &m.estimate);
        assert_gaussian_close(
            Residual::NativeVsHand,
            &what,
            &leaf,
            &exact_mixed(case, j).estimate,
        );
    }
}

/// `interaction_step()` on a CV+CA bank: the CV leaf equals `(x0_1, P0_1)`.
fn check_native_cv_ca_interaction(label: &str, case: &TwoModeMixing) {
    let mut bank = native_cv_ca_bank(&case.pi, &mu_worked(), 0.0);
    bank.interaction_step();
    let (_, _, mixed) = oracle_worked_mixing(&case.pi);
    let leaf = native_leaf(&bank, 0);
    let what = format!("{label} CV+CA CV leaf");
    assert_gaussian_close(Residual::NativeVsOracle, &what, &leaf, &mixed[0].estimate);
    assert_gaussian_close(
        Residual::NativeVsHand,
        &what,
        &leaf,
        &exact_mixed(case, 0).estimate,
    );
}

#[test]
fn native_interaction_cv_cv_identity_matches_mixed_moments() {
    check_native_cv_cv_interaction("identity", &identity_mixing());
}

#[test]
fn native_interaction_cv_cv_symmetric_matches_mixed_moments() {
    check_native_cv_cv_interaction("symmetric", &symmetric_mixing());
}

#[test]
fn native_interaction_cv_cv_asymmetric_matches_mixed_moments() {
    check_native_cv_cv_interaction("asymmetric", &asymmetric_mixing());
}

#[test]
fn native_interaction_cv_ca_identity_cv_leaf_matches_mixed_moments() {
    check_native_cv_ca_interaction("identity", &identity_mixing());
}

#[test]
fn native_interaction_cv_ca_symmetric_cv_leaf_matches_mixed_moments() {
    check_native_cv_ca_interaction("symmetric", &symmetric_mixing());
}

#[test]
fn native_interaction_cv_ca_asymmetric_cv_leaf_matches_mixed_moments() {
    check_native_cv_ca_interaction("asymmetric", &asymmetric_mixing());
}

/// With `Π = I`, `ln(μ⁺₁/μ⁺₂) − ln(μ₁/μ₂) = ℓ₁ − ℓ₂`: the native posterior
/// ratio checks each mode's elapsed-time log-likelihood difference against
/// the closed form.
fn assert_identity_posterior_ratio(label: &str, first: &CycleRecord, dt: f64) {
    let z = measurement_sequence(dt).remove(0);
    let ll_1 = closed_form_cv_log_likelihood(&worked_cv_leaf(), SIGMA_A_MODE_1, dt, &z);
    let ll_2 = closed_form_cv_log_likelihood(&worked_ca_common(), SIGMA_A_MODE_2, dt, &z);
    let native = (first.native_mu[0] / first.native_mu[1]).ln()
        - (first.mu_prior[0] / first.mu_prior[1]).ln();
    assert_close(
        Residual::NativeVsHand,
        &format!("{label} ℓ₁ − ℓ₂"),
        native,
        ll_1 - ll_2,
    );
}

fn run_identity(label: &str, dt: f64) {
    let records = run_cv_cv_end_to_end(label, &pi_identity(), dt, &measurement_sequence(dt));
    assert_eq!(records.len(), END_TO_END_CYCLES);
    assert_identity_posterior_ratio(label, &records[0], dt);
}

#[test]
fn native_cv_cv_identity_run_dt_1_matches_oracle() {
    run_identity("identity dt 1", 1.0);
}

#[test]
fn native_cv_cv_identity_run_dt_0p1_matches_oracle() {
    run_identity("identity dt 0.1", 0.1);
}

#[test]
fn native_cv_cv_identity_run_dt_2p3_matches_oracle() {
    run_identity("identity dt 2.3", 2.3);
}

#[test]
fn native_cv_cv_symmetric_run_dt_1_matches_oracle() {
    let records = run_cv_cv_end_to_end(
        "symmetric dt 1",
        &pi_symmetric(),
        1.0,
        &measurement_sequence(1.0),
    );
    assert_eq!(records.len(), END_TO_END_CYCLES);
}

#[test]
fn native_cv_cv_asymmetric_run_dt_1_matches_oracle() {
    let records = run_cv_cv_end_to_end(
        "asymmetric dt 1",
        &pi_asymmetric(),
        1.0,
        &measurement_sequence(1.0),
    );
    assert_eq!(records.len(), END_TO_END_CYCLES);
}

#[test]
fn native_cv_cv_underflow_run_matches_oracle() {
    let records = run_cv_cv_end_to_end("underflow", &pi_asymmetric(), 1.0, &underflow_sequence());
    assert_eq!(records.len(), 2);
    let first = &records[0];
    assert!(
        first
            .oracle
            .log_likelihoods()
            .iter()
            .all(|ll| ll.exp() == 0.0),
        "both linear likelihoods must underflow in the first cycle"
    );
    assert_slice_close(
        Residual::NativeVsHand,
        "underflow μ⁺",
        first.native_mu.as_slice(),
        &[0.0, 1.0],
    );
}

#[test]
fn native_cv_ca_zero_destination_floor_zero_gives_exact_posterior() {
    // Declared zero-destination case: c = [1, 0]. Under the current
    // zero-destination assignment mode 2's log-likelihood (−184.355…) is below
    // mode 1's (−6.859…), so the spec's zero-probability clause applies; the
    // test claims nothing about a larger mode-2 log-likelihood.
    let (pi, m) = (pi_zero_destination(), worked_measurement());
    let mut bank = native_cv_ca_bank(&pi, &mu_worked(), 0.0);
    let result = native_step(&mut bank, 1.0, &m);
    let oracle = specified_cycle(
        &oracle_cv_ca_modes(1.0),
        &worked_cv_ca_leaves(),
        &mu_worked(),
        &pi,
        &m,
    );
    assert_probabilities_sum_to_one("zero-destination μ⁺", &result.mode_probabilities);
    let mu = result.mode_probabilities.as_slice();
    assert_slice_close(
        Residual::NativeVsHand,
        "zero-destination μ⁺",
        mu,
        &[1.0, 0.0],
    );
    assert_vector_close(
        Residual::NativeVsOracle,
        "zero-destination μ⁺",
        &result.mode_probabilities,
        &oracle.posterior_mu,
    );
    let combined = Gaussian {
        mean: result.state,
        cov: result.covariance,
    };
    assert_gaussian_close(
        Residual::NativeVsOracle,
        "zero-destination combined",
        &combined,
        &oracle.combined,
    );
    let cv_post = &oracle.updates[0].as_ref().expect("mode 1 active").posterior;
    assert_gaussian_close(
        Residual::NativeVsOracle,
        "zero-destination CV leaf",
        &native_leaf(&bank, 0),
        cv_post,
    );
    assert_vector_close(
        Residual::NativeVsHand,
        "zero-destination combined mean",
        &combined.mean,
        &zero_destination_exact_mean(),
    );
}

/// Overwrite the public mode probabilities, call `combine()`, and compare
/// with the oracle's moment match of the leaves' common estimates.
fn check_supplied_combination(
    label: &str,
    bank: &mut ImmFilter,
    modes: &[LinearMode],
    supplied: &[f64],
) {
    bank.mode_probabilities = DVector::from_column_slice(supplied);
    let (x, p) = bank.combine();
    let leaves: Vec<Gaussian> = (0..modes.len()).map(|j| native_leaf(bank, j)).collect();
    let commons = common_marginals(modes, &leaves);
    let refs: Vec<&Gaussian> = commons.iter().collect();
    let expected = combine(&refs, supplied);
    let actual = Gaussian { mean: x, cov: p };
    assert_gaussian_close(Residual::NativeVsOracle, label, &actual, &expected);
}

#[test]
fn native_combine_with_supplied_probabilities_matches_moment_match() {
    // CV+CA worked leaves: the CA leaf contributes its common marginal.
    let mut bank = native_cv_ca_bank(&pi_asymmetric(), &mu_worked(), 0.0);
    check_supplied_combination(
        "CV+CA supplied",
        &mut bank,
        &oracle_cv_ca_modes(1.0),
        &[0.25, 0.75],
    );
    // CV+CV posterior leaves after one specified step.
    let mut bank = native_cv_cv_bank(&pi_asymmetric(), &mu_worked(), 0.0);
    native_step(&mut bank, 1.0, &worked_measurement());
    let modes = oracle_cv_cv_modes(1.0);
    check_supplied_combination("CV+CV supplied", &mut bank, &modes, &[0.7, 0.3]);
    check_supplied_combination("CV+CV one-hot", &mut bank, &modes, &[0.0, 1.0]);
}

#[test]
fn native_combine_reproduces_worked_combination() {
    let mut a = worked_cv_leaf();
    a.mean[0] = 10.812892420810806;
    a.cov[(0, 0)] = 2.6933736334545353;
    let mut b = worked_ca_common();
    b.mean[0] = 11.206007753504345;
    b.cov[(0, 0)] = 3.6409686692927807;
    let mut bank = native_cv_bank(
        &[SIGMA_A_MODE_1, SIGMA_A_MODE_2],
        &pi_asymmetric(),
        &mu_worked(),
        &[a, b],
        0.0,
    );
    bank.mode_probabilities = DVector::from_column_slice(&MU_POSTERIOR_WORKED);
    let (x, p) = bank.combine();
    assert_close(
        Residual::NativeVsHand,
        "worked combined x",
        x[0],
        10.84142489054062,
    );
    assert_close(
        Residual::NativeVsHand,
        "worked combined var x",
        p[(0, 0)],
        2.7725529136406477,
    );
}

#[test]
fn native_mixing_and_posterior_invariants_hold() {
    let case = three_mode_case();
    let mut bank = native_cv_bank(&case.sigmas, &case.pi, &case.mu, &case.leaves, 0.0);
    bank.interaction_step();
    let mixed: Vec<Gaussian> = (0..3).map(|j| native_leaf(&bank, j)).collect();
    let expected = oracle_mixed(&case.pi, &case.mu, &case.leaves);
    for (j, (n, o)) in mixed.iter().zip(&expected).enumerate() {
        assert_gaussian_close(
            Residual::NativeVsOracle,
            &format!("three-mode leaf {j}"),
            n,
            o,
        );
    }
    assert_mixing_invariants(
        "three-mode native",
        &case.pi,
        &case.mu,
        &case.leaves,
        &mixed,
    );
    // Posterior: a fresh bank through one step.
    let mut bank = native_cv_bank(&case.sigmas, &case.pi, &case.mu, &case.leaves, 0.0);
    let m = worked_measurement();
    let result = native_step(&mut bank, 1.0, &m);
    let modes: Vec<LinearMode> = case
        .sigmas
        .iter()
        .map(|&s| LinearMode::cv(1.0, s))
        .collect();
    let oracle = specified_cycle(&modes, &case.leaves, &case.mu, &case.pi, &m);
    assert_probabilities_sum_to_one("three-mode native μ⁺", &result.mode_probabilities);
    assert_vector_close(
        Residual::NativeVsOracle,
        "three-mode μ⁺",
        &result.mode_probabilities,
        &oracle.posterior_mu,
    );
}

// ---------------------------------------------------------------------------
// 5. Configuration validation (V)
// ---------------------------------------------------------------------------

/// `ImmConfig::cv_ca(1, 1)` with the given transition matrix (row-major) and
/// initial mode probabilities.
fn cv_ca_config_with(pi: [f64; 4], mu: [f64; 2]) -> ImmConfig {
    let mut config = ImmConfig::cv_ca(1.0, 1.0);
    config.transition_matrix = DMatrix::from_row_slice(2, 2, &pi);
    config.initial_mode_probabilities = DVector::from_column_slice(&mu);
    config
}

fn assert_rejected(config: &ImmConfig, needle: &str) {
    match config.validate() {
        Ok(()) => panic!("configuration should be rejected ({needle})"),
        Err(e) => assert!(e.contains(needle), "error {e:?} should mention {needle:?}"),
    }
}

#[test]
fn validate_rejects_nan_transition_entry() {
    assert_rejected(
        &cv_ca_config_with([0.9, f64::NAN, 0.2, 0.8], [0.6, 0.4]),
        "transition matrix entry (0, 1)",
    );
}

#[test]
fn validate_rejects_positive_infinite_transition_entry() {
    assert_rejected(
        &cv_ca_config_with([0.9, 0.1, f64::INFINITY, 0.8], [0.6, 0.4]),
        "transition matrix entry (1, 0)",
    );
}

#[test]
fn validate_rejects_negative_infinite_transition_entry() {
    assert_rejected(
        &cv_ca_config_with([0.9, 0.1, 0.2, f64::NEG_INFINITY], [0.6, 0.4]),
        "transition matrix entry (1, 1)",
    );
}

#[test]
fn validate_rejects_negative_transition_entry_in_row_summing_to_one() {
    assert_rejected(
        &cv_ca_config_with([1.2, -0.2, 0.05, 0.95], [0.6, 0.4]),
        "transition matrix entry (0, 1)",
    );
}

#[test]
fn validate_rejects_nan_initial_probability() {
    assert_rejected(
        &cv_ca_config_with([0.95, 0.05, 0.05, 0.95], [f64::NAN, 0.4]),
        "initial mode probability 0",
    );
}

#[test]
fn validate_rejects_infinite_initial_probability() {
    assert_rejected(
        &cv_ca_config_with([0.95, 0.05, 0.05, 0.95], [0.6, f64::INFINITY]),
        "initial mode probability 1",
    );
}

#[test]
fn validate_rejects_negative_initial_probability_summing_to_one() {
    assert_rejected(
        &cv_ca_config_with([0.95, 0.05, 0.05, 0.95], [1.5, -0.5]),
        "initial mode probability 1",
    );
}

#[test]
fn validate_accepts_library_presets() {
    let presets = [
        ImmConfig::cv_ca(1.0, 1.0),
        ImmConfig::cv_ca(5.0, 1.0),
        ImmConfig::cv_ctrv(1.0, 5.0, 0.5),
        ImmConfig::cv_ca_ctrv_ct(5.0, 1.0, 2.0, 0.1),
    ];
    for (k, preset) in presets.iter().enumerate() {
        assert_eq!(preset.validate(), Ok(()), "preset {k}");
    }
}

#[test]
#[should_panic(expected = "transition matrix entry (0, 1)")]
fn imm_filter_new_panics_on_nan_transition_entry() {
    let config = cv_ca_config_with([0.9, f64::NAN, 0.2, 0.8], [0.6, 0.4]);
    let _ = ImmFilter::new(config, &DVector::zeros(6), &DMatrix::identity(6, 6));
}

#[test]
#[should_panic(expected = "initial mode probability 1")]
fn imm_filter_new_panics_on_negative_initial_probability() {
    let config = cv_ca_config_with([0.95, 0.05, 0.05, 0.95], [1.5, -0.5]);
    let _ = ImmFilter::new(config, &DVector::zeros(6), &DMatrix::identity(6, 6));
}

#[test]
#[should_panic(expected = "transition matrix entry (0, 1)")]
fn imm_filter_with_leaf_kind_panics_on_negative_transition_entry() {
    let config = cv_ca_config_with([1.2, -0.2, 0.05, 0.95], [0.6, 0.4]);
    let _ = ImmFilter::with_leaf_kind(
        config,
        ImmLeafKind::Ckf,
        &DVector::zeros(6),
        &DMatrix::identity(6, 6),
    );
}
