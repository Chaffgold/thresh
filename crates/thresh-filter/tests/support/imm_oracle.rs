//! Independent IMM reference oracle (test support; no `#[test]` here).
//!
//! Written from the textbook interacting-multiple-model formulation (Blom and
//! Bar-Shalom 1988; Bar-Shalom, Li and Kirubarajan 2001, ch. 11) and the
//! closed-form fixture models declared below. It shares no code with the
//! production filter bank, its motion models, state mappings or leaf filters:
//! it imports only `nalgebra` and `std`, and a source check in
//! `imm_reference.rs` enforces that.
//!
//! Common state: `[x, vx, y, vy, z, vz]` in metres and metres per second.
//! The constant-acceleration (CA) fixture layout is
//! `[x, vx, ax, y, vy, ay, z, vz, az]`; its common-state marginal uses the
//! selector `[0, 1, 3, 4, 6, 7]`.
//!
//! The oracle has two layers:
//!
//! - **Specified path** — the policy-independent mathematics of the
//!   `imm-estimation` capability, as the phase sequence
//!   `predicted_mode_probabilities → mixing_weights → mix_common →
//!   predict_common → joseph_update → mode_posterior → combine`, driven by
//!   [`specified_cycle`]. A mode whose predicted probability is exactly zero
//!   is excluded and gets posterior weight exactly zero, with no clamp.
//! - **Current-policy layer** — [`CurrentPolicy`] and [`policy_cycle`]. Every
//!   unspecified policy is an explicit parameter there, so a characterisation
//!   test states which policy it emulates. The specified path never calls
//!   this layer; the layer reuses the specified phase helpers.

use nalgebra::{Cholesky, DMatrix, DVector, Dyn};

/// Common state dimension, `[x, vx, y, vy, z, vz]`.
pub const COMMON_DIM: usize = 6;

/// Indices of the common components inside the CA layout.
pub const CA_COMMON_INDICES: [usize; COMMON_DIM] = [0, 1, 3, 4, 6, 7];

/// Indices of the CA acceleration components (outside the common state).
pub const CA_ACCEL_INDICES: [usize; 3] = [2, 5, 8];

/// Predicted-probability bound above which the capability specifies mixing
/// and the posterior.
pub const SPECIFIED_PROBABILITY_BOUND: f64 = 1e-12;

/// A Gaussian estimate: mean and covariance.
#[derive(Clone, Debug)]
pub struct Gaussian {
    /// Mean vector.
    pub mean: DVector<f64>,
    /// Covariance matrix.
    pub cov: DMatrix<f64>,
}

impl Gaussian {
    /// Build a Gaussian from a mean slice and a covariance.
    pub fn new(mean: &[f64], cov: DMatrix<f64>) -> Self {
        Self {
            mean: DVector::from_column_slice(mean),
            cov,
        }
    }
}

/// A linear-Gaussian fixture mode `x' = F x + w`, `w ~ N(0, Q)`, with a
/// selector `S` whose product `S x` is the mode's common-state marginal.
#[derive(Clone, Debug)]
pub struct LinearMode {
    /// State transition matrix for the cycle's elapsed time.
    pub f: DMatrix<f64>,
    /// Process-noise covariance for the cycle's elapsed time.
    pub q: DMatrix<f64>,
    /// Common-state selector (`COMMON_DIM` rows, one column per state).
    pub selector: DMatrix<f64>,
}

/// Measurement `z` with linear common-state observation `H` and noise `R`.
#[derive(Clone, Debug)]
pub struct Measurement {
    /// Measurement vector.
    pub z: DVector<f64>,
    /// Observation matrix acting on the common state.
    pub h: DMatrix<f64>,
    /// Measurement noise covariance.
    pub r: DMatrix<f64>,
}

// ---------------------------------------------------------------------------
// Closed-form fixture models
// ---------------------------------------------------------------------------

/// Place one per-axis block on the diagonal for each of the three axes.
fn per_axis(block: &DMatrix<f64>) -> DMatrix<f64> {
    let b = block.nrows();
    let mut out = DMatrix::zeros(3 * b, 3 * b);
    for axis in 0..3 {
        out.view_mut((axis * b, axis * b), (b, b)).copy_from(block);
    }
    out
}

/// CV transition: per axis `[[1, dt], [0, 1]]`.
pub fn cv_transition(dt: f64) -> DMatrix<f64> {
    per_axis(&DMatrix::from_row_slice(2, 2, &[1.0, dt, 0.0, 1.0]))
}

/// CV discrete white-noise acceleration:
/// per axis `σ_a² [[dt⁴/4, dt³/2], [dt³/2, dt²]]`.
pub fn cv_process_noise(dt: f64, sigma_a: f64) -> DMatrix<f64> {
    let block = DMatrix::from_row_slice(
        2,
        2,
        &[
            dt.powi(4) / 4.0,
            dt.powi(3) / 2.0,
            dt.powi(3) / 2.0,
            dt.powi(2),
        ],
    );
    per_axis(&(block * (sigma_a * sigma_a)))
}

/// CA transition: per axis `[[1, dt, dt²/2], [0, 1, dt], [0, 0, 1]]`.
pub fn ca_transition(dt: f64) -> DMatrix<f64> {
    let block =
        DMatrix::from_row_slice(3, 3, &[1.0, dt, dt * dt / 2.0, 0.0, 1.0, dt, 0.0, 0.0, 1.0]);
    per_axis(&block)
}

/// CA continuous white-noise jerk:
/// per axis `σ_j² [[dt⁵/20, dt⁴/8, dt³/6], [dt⁴/8, dt³/3, dt²/2], [dt³/6, dt²/2, dt]]`.
pub fn ca_process_noise(dt: f64, sigma_j: f64) -> DMatrix<f64> {
    let block = DMatrix::from_row_slice(
        3,
        3,
        &[
            dt.powi(5) / 20.0,
            dt.powi(4) / 8.0,
            dt.powi(3) / 6.0,
            dt.powi(4) / 8.0,
            dt.powi(3) / 3.0,
            dt.powi(2) / 2.0,
            dt.powi(3) / 6.0,
            dt.powi(2) / 2.0,
            dt,
        ],
    );
    per_axis(&(block * (sigma_j * sigma_j)))
}

/// Selector `S` (6 x 9) with `common = S · ca_state`.
pub fn ca_common_selector() -> DMatrix<f64> {
    let mut s = DMatrix::zeros(COMMON_DIM, 9);
    for (row, &col) in CA_COMMON_INDICES.iter().enumerate() {
        s[(row, col)] = 1.0;
    }
    s
}

impl LinearMode {
    /// CV mode over `dt` seconds with acceleration noise `sigma_a`.
    pub fn cv(dt: f64, sigma_a: f64) -> Self {
        Self {
            f: cv_transition(dt),
            q: cv_process_noise(dt, sigma_a),
            selector: DMatrix::identity(COMMON_DIM, COMMON_DIM),
        }
    }

    /// CA mode over `dt` seconds with jerk noise `sigma_j`.
    pub fn ca(dt: f64, sigma_j: f64) -> Self {
        Self {
            f: ca_transition(dt),
            q: ca_process_noise(dt, sigma_j),
            selector: ca_common_selector(),
        }
    }

    /// Dimension of the mode's own state.
    pub fn state_dim(&self) -> usize {
        self.selector.ncols()
    }

    /// Whether the mode's state is the common state itself (no projection).
    pub fn is_common(&self) -> bool {
        self.selector == DMatrix::identity(COMMON_DIM, COMMON_DIM)
    }

    /// Common-state marginal `(S x, S P Sᵀ)` of an estimate in the mode's state.
    pub fn marginal(&self, g: &Gaussian) -> Gaussian {
        Gaussian {
            mean: &self.selector * &g.mean,
            cov: &self.selector * &g.cov * self.selector.transpose(),
        }
    }

    /// Prediction `(F x, F P Fᵀ + Q)` in the mode's own state.
    pub fn predict(&self, g: &Gaussian) -> Gaussian {
        Gaussian {
            mean: &self.f * &g.mean,
            cov: &self.f * &g.cov * self.f.transpose() + &self.q,
        }
    }
}

// ---------------------------------------------------------------------------
// Specified path: phase helpers
// ---------------------------------------------------------------------------

/// Common-state marginals of each mode's estimate.
pub fn common_marginals(modes: &[LinearMode], leaves: &[Gaussian]) -> Vec<Gaussian> {
    modes
        .iter()
        .zip(leaves)
        .map(|(m, g)| m.marginal(g))
        .collect()
}

/// Predicted mode probabilities `c_j = Σ_i π_ij μ_i`.
pub fn predicted_mode_probabilities(pi: &DMatrix<f64>, mu: &DVector<f64>) -> DVector<f64> {
    pi.transpose() * mu
}

/// Conditional mixing weights `μ_i|j = π_ij μ_i / c_j` (rows `i`, columns
/// `j`) for every destination with `c_j > threshold`; other columns are zero.
pub fn mixing_weights_above(
    pi: &DMatrix<f64>,
    mu: &DVector<f64>,
    c: &DVector<f64>,
    threshold: f64,
) -> DMatrix<f64> {
    let n = mu.len();
    DMatrix::from_fn(n, n, |i, j| {
        if c[j] > threshold {
            pi[(i, j)] * mu[i] / c[j]
        } else {
            0.0
        }
    })
}

/// Specified mixing weights: exact, excluding only an exactly-zero `c_j`.
pub fn mixing_weights(pi: &DMatrix<f64>, mu: &DVector<f64>, c: &DVector<f64>) -> DMatrix<f64> {
    mixing_weights_above(pi, mu, c, 0.0)
}

/// Mixed common moments of one destination, with its between-mode spread.
#[derive(Clone, Debug)]
pub struct MixedEstimate {
    /// `(x0_j, P0_j)`, where `P0_j` includes the spread.
    pub estimate: Gaussian,
    /// Spread `Σ_i μ_i|j (x_i − x0_j)(x_i − x0_j)ᵀ`.
    pub spread: DMatrix<f64>,
}

/// Mix destination `j`: `x0_j = Σ_i μ_i|j x_i` and
/// `P0_j = Σ_i μ_i|j [P_i + (x_i − x0_j)(x_i − x0_j)ᵀ]`. A zero weight
/// column gives zero moments.
pub fn mix_destination(commons: &[Gaussian], weights: &DMatrix<f64>, j: usize) -> MixedEstimate {
    let mut mean = DVector::zeros(COMMON_DIM);
    for (i, g) in commons.iter().enumerate() {
        mean += &g.mean * weights[(i, j)];
    }
    let mut own = DMatrix::zeros(COMMON_DIM, COMMON_DIM);
    let mut spread = DMatrix::zeros(COMMON_DIM, COMMON_DIM);
    for (i, g) in commons.iter().enumerate() {
        let d = &g.mean - &mean;
        own += &g.cov * weights[(i, j)];
        spread += (&d * d.transpose()) * weights[(i, j)];
    }
    MixedEstimate {
        estimate: Gaussian {
            mean,
            cov: own + &spread,
        },
        spread,
    }
}

/// Specified mixing for every destination: `Some` for `c_j > 0`, `None` for
/// an exactly-zero `c_j` (that mode is excluded from the cycle).
pub fn mix_common(
    commons: &[Gaussian],
    weights: &DMatrix<f64>,
    c: &DVector<f64>,
) -> Vec<Option<MixedEstimate>> {
    (0..c.len())
        .map(|j| (c[j] > 0.0).then(|| mix_destination(commons, weights, j)))
        .collect()
}

/// Specified prediction of a mixed common estimate. Only a mode whose state
/// is the common state can be predicted without a reconstruction policy.
pub fn predict_common(mode: &LinearMode, mixed: &Gaussian) -> Gaussian {
    assert!(
        mode.is_common(),
        "the specified path predicts only common-state modes; components outside \
         the common state need a reconstruction policy (current-policy layer)"
    );
    mode.predict(mixed)
}

/// Kalman update output in the observed state.
#[derive(Clone, Debug)]
pub struct UpdateOutput {
    /// Posterior estimate (Joseph-form covariance).
    pub posterior: Gaussian,
    /// Innovation `y = z − H x̂`.
    pub innovation: DVector<f64>,
    /// Innovation covariance `S = H P Hᵀ + R`.
    pub innovation_cov: DMatrix<f64>,
    /// Model-conditioned log-likelihood of `z`.
    pub log_likelihood: f64,
}

/// Cholesky factor of a positive-definite innovation covariance.
fn cholesky(s: &DMatrix<f64>) -> Cholesky<f64, Dyn> {
    Cholesky::new(s.clone()).expect("innovation covariance must be positive definite")
}

/// `ln det S = 2 Σ ln L_ii` from the Cholesky factor `S = L Lᵀ`.
pub fn log_det_spd(s: &DMatrix<f64>) -> f64 {
    let l = cholesky(s).l();
    2.0 * (0..l.nrows()).map(|i| l[(i, i)].ln()).sum::<f64>()
}

/// Gaussian log-likelihood `−½ (m ln 2π + ln det S + yᵀ S⁻¹ y)`, evaluated in
/// log space through the Cholesky factor.
pub fn gaussian_log_likelihood(y: &DVector<f64>, s: &DMatrix<f64>) -> f64 {
    let maha = y.dot(&cholesky(s).solve(y));
    let m = y.len() as f64;
    -0.5 * (m * (2.0 * std::f64::consts::PI).ln() + log_det_spd(s) + maha)
}

/// Kalman update with gain `K = P Hᵀ S⁻¹` and the Joseph-form covariance
/// `(I − K H) P (I − K H)ᵀ + K R Kᵀ`.
pub fn joseph_update(prior: &Gaussian, m: &Measurement) -> UpdateOutput {
    let y = &m.z - &m.h * &prior.mean;
    let s = &m.h * &prior.cov * m.h.transpose() + &m.r;
    let k = cholesky(&s).solve(&(&m.h * &prior.cov)).transpose();
    let n = prior.mean.len();
    let i_kh = DMatrix::identity(n, n) - &k * &m.h;
    let cov = &i_kh * &prior.cov * i_kh.transpose() + &k * &m.r * k.transpose();
    UpdateOutput {
        posterior: Gaussian {
            mean: &prior.mean + &k * &y,
            cov,
        },
        log_likelihood: gaussian_log_likelihood(&y, &s),
        innovation: y,
        innovation_cov: s,
    }
}

/// Normalise log weights in shifted coordinates:
/// `w_j = exp(lw_j − max) / Σ_k exp(lw_k − max)`. Normalising after the
/// shift, rather than adding the log normaliser back to `max`, keeps the
/// result summing to one when `|max|` is large. Entries of `−∞` map to
/// exactly zero.
pub fn normalise_log_weights(log_weights: &[f64]) -> DVector<f64> {
    let max = log_weights
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);
    assert!(
        max.is_finite(),
        "at least one finite log weight is required"
    );
    let shifted = DVector::from_iterator(
        log_weights.len(),
        log_weights.iter().map(|lw| (lw - max).exp()),
    );
    let sum = shifted.sum();
    shifted / sum
}

/// Exact Bayes posterior `μ⁺_j ∝ c_j exp(ℓ_j)` in log space. The
/// log-likelihoods are shifted by the largest one among modes with
/// `c_j > 0` before the log priors are added, so the priors survive even
/// when `|ℓ|` is large; the shift cancels in the normalisation. A mode with
/// `c_j = 0` gets exactly zero, whatever its log-likelihood.
pub fn mode_posterior(c: &DVector<f64>, log_likelihoods: &[f64]) -> DVector<f64> {
    let max_active = c
        .iter()
        .zip(log_likelihoods)
        .filter(|(cj, _)| **cj > 0.0)
        .map(|(_, &lj)| lj)
        .fold(f64::NEG_INFINITY, f64::max);
    let log_weights: Vec<f64> = c
        .iter()
        .zip(log_likelihoods)
        .map(|(&cj, &lj)| {
            if cj > 0.0 {
                cj.ln() + (lj - max_active)
            } else {
                f64::NEG_INFINITY
            }
        })
        .collect();
    normalise_log_weights(&log_weights)
}

/// Moment match `x = Σ μ_j x_j`, `P = Σ μ_j [P_j + (x_j − x)(x_j − x)ᵀ]`.
pub fn combine(estimates: &[&Gaussian], mu: &[f64]) -> Gaussian {
    let dim = estimates[0].mean.len();
    let mut mean = DVector::zeros(dim);
    for (g, &w) in estimates.iter().zip(mu) {
        mean += &g.mean * w;
    }
    let mut cov = DMatrix::zeros(dim, dim);
    for (g, &w) in estimates.iter().zip(mu) {
        let d = &g.mean - &mean;
        cov += (&g.cov + &d * d.transpose()) * w;
    }
    Gaussian { mean, cov }
}

// ---------------------------------------------------------------------------
// Specified path: one cycle
// ---------------------------------------------------------------------------

/// Every intermediate of one specified IMM cycle. `None` marks a mode
/// excluded because its predicted probability is exactly zero.
#[derive(Clone, Debug)]
pub struct SpecifiedCycle {
    /// Predicted mode probabilities `c`.
    pub c: DVector<f64>,
    /// Mixing weights `μ_i|j`.
    pub weights: DMatrix<f64>,
    /// Mixed common moments per destination.
    pub mixed: Vec<Option<MixedEstimate>>,
    /// Predicted estimates per mode.
    pub predicted: Vec<Option<Gaussian>>,
    /// Measurement updates per mode.
    pub updates: Vec<Option<UpdateOutput>>,
    /// Posterior mode probabilities `μ⁺`.
    pub posterior_mu: DVector<f64>,
    /// Combined posterior estimate.
    pub combined: Gaussian,
}

impl SpecifiedCycle {
    /// Posterior estimate of every mode, for the next cycle.
    ///
    /// # Panics
    /// Panics if a mode was excluded from this cycle.
    pub fn posterior_leaves(&self) -> Vec<Gaussian> {
        self.updates
            .iter()
            .map(|u| {
                u.as_ref()
                    .expect("mode excluded from cycle")
                    .posterior
                    .clone()
            })
            .collect()
    }

    /// Log-likelihood of every mode (`−∞` for an excluded mode).
    pub fn log_likelihoods(&self) -> Vec<f64> {
        active_log_likelihoods(&self.updates)
    }
}

fn predict_active(modes: &[LinearMode], mixed: &[Option<MixedEstimate>]) -> Vec<Option<Gaussian>> {
    modes
        .iter()
        .zip(mixed)
        .map(|(mode, m)| m.as_ref().map(|m| predict_common(mode, &m.estimate)))
        .collect()
}

fn update_active(predicted: &[Option<Gaussian>], m: &Measurement) -> Vec<Option<UpdateOutput>> {
    predicted
        .iter()
        .map(|p| p.as_ref().map(|p| joseph_update(p, m)))
        .collect()
}

fn active_log_likelihoods(updates: &[Option<UpdateOutput>]) -> Vec<f64> {
    updates
        .iter()
        .map(|u| u.as_ref().map_or(f64::NEG_INFINITY, |u| u.log_likelihood))
        .collect()
}

fn combine_active(updates: &[Option<UpdateOutput>], mu: &DVector<f64>) -> Gaussian {
    let (estimates, weights): (Vec<&Gaussian>, Vec<f64>) = updates
        .iter()
        .zip(mu.iter())
        .filter_map(|(u, &w)| u.as_ref().map(|u| (&u.posterior, w)))
        .unzip();
    combine(&estimates, &weights)
}

/// One specified cycle: interaction, prediction, update, posterior and
/// combination, with no unspecified policy. Every active destination must be
/// a common-state mode; a non-common mode may only appear as a mixing source
/// (through its marginal) or as an excluded destination.
pub fn specified_cycle(
    modes: &[LinearMode],
    leaves: &[Gaussian],
    mu: &DVector<f64>,
    pi: &DMatrix<f64>,
    m: &Measurement,
) -> SpecifiedCycle {
    let commons = common_marginals(modes, leaves);
    let c = predicted_mode_probabilities(pi, mu);
    let weights = mixing_weights(pi, mu, &c);
    let mixed = mix_common(&commons, &weights, &c);
    let predicted = predict_active(modes, &mixed);
    let updates = update_active(&predicted, m);
    let posterior_mu = mode_posterior(&c, &active_log_likelihoods(&updates));
    let combined = combine_active(&updates, &posterior_mu);
    SpecifiedCycle {
        c,
        weights,
        mixed,
        predicted,
        updates,
        posterior_mu,
        combined,
    }
}

// ---------------------------------------------------------------------------
// Current-policy layer (characterisation only; never called by the
// specified path)
// ---------------------------------------------------------------------------

/// Mode-probability propagation when a prediction is not followed by an
/// update.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CoastPolicy {
    /// Keep the pre-prediction probabilities (stale `μ`).
    KeepModeProbabilities,
    /// Propagate `μ ← Πᵀ μ`.
    PropagateModeProbabilities,
}

/// Weights of the predicted (pre-update) combined estimate.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PredictedOutputWeighting {
    /// Weight by the prior mode probabilities `μ`.
    PriorModeProbabilities,
    /// Weight by the predicted mode probabilities `c`.
    PredictedModeProbabilities,
}

/// Explicit values for every unspecified policy the oracle can emulate.
#[derive(Clone, Copy, Debug)]
pub struct CurrentPolicy {
    /// Mean re-created for each state component outside the common state
    /// (CA acceleration) when a common estimate is expanded.
    pub reconstruction_mean: f64,
    /// Variance re-created for each such component, with zero
    /// cross-covariance.
    pub reconstruction_var: f64,
    /// Minimum posterior mode probability, followed by renormalisation.
    pub floor: f64,
    /// Mixing computes `μ_i|j` only when `c_j` exceeds this; otherwise the
    /// mixed state and covariance are zero.
    pub mixing_threshold: f64,
    /// The mode update uses `ln max(c_j, clamp)`.
    pub mode_update_clamp: f64,
    /// Mode probabilities after a prediction without an update.
    pub coast: CoastPolicy,
    /// Weights of the predicted combined estimate.
    pub predicted_output: PredictedOutputWeighting,
}

impl CurrentPolicy {
    /// The declared current production floor.
    pub const DEFAULT_FLOOR: f64 = 0.01;

    /// The declared current production policy, with the given floor:
    /// reconstruction mean 0 and variance 100 (m/s²)², mixing threshold
    /// 1e-30, mode-update clamp 1e-300, stale coasting and prior-weighted
    /// predicted output.
    pub fn current(floor: f64) -> Self {
        Self {
            reconstruction_mean: 0.0,
            reconstruction_var: 100.0,
            floor,
            mixing_threshold: 1e-30,
            mode_update_clamp: 1e-300,
            coast: CoastPolicy::KeepModeProbabilities,
            predicted_output: PredictedOutputWeighting::PriorModeProbabilities,
        }
    }
}

/// Expand a common estimate into the mode's own state: `(Sᵀ x, Sᵀ P S)`, then
/// re-create every component outside the common state from the policy.
pub fn reconstruct(mode: &LinearMode, common: &Gaussian, policy: &CurrentPolicy) -> Gaussian {
    let st = mode.selector.transpose();
    let mut g = Gaussian {
        mean: &st * &common.mean,
        cov: &st * &common.cov * &mode.selector,
    };
    for k in (0..mode.state_dim()).filter(|&k| mode.selector.column(k).iter().all(|&v| v == 0.0)) {
        g.mean[k] = policy.reconstruction_mean;
        g.cov[(k, k)] = policy.reconstruction_var;
    }
    g
}

/// Current mode update: `ln max(c_j, clamp) + ℓ_j`, log-sum-exp, then the
/// floor and renormalisation.
pub fn policy_mode_update(
    c: &DVector<f64>,
    log_likelihoods: &[f64],
    policy: &CurrentPolicy,
) -> DVector<f64> {
    let log_weights: Vec<f64> = c
        .iter()
        .zip(log_likelihoods)
        .map(|(&cj, &lj)| cj.max(policy.mode_update_clamp).ln() + lj)
        .collect();
    let floored = normalise_log_weights(&log_weights).map(|p| p.max(policy.floor));
    let sum = floored.sum();
    floored / sum
}

/// Every intermediate of one current-policy cycle.
#[derive(Clone, Debug)]
pub struct PolicyCycle {
    /// Predicted mode probabilities `c`.
    pub c: DVector<f64>,
    /// Mixing weights after the mixing threshold.
    pub weights: DMatrix<f64>,
    /// Mixed common moments (zero below the mixing threshold).
    pub mixed_common: Vec<Gaussian>,
    /// Mixed estimates expanded into each mode's own state.
    pub mixed_native: Vec<Gaussian>,
    /// Predictions in each mode's own state.
    pub predicted_native: Vec<Gaussian>,
    /// Predicted combined common estimate under the output weighting.
    pub predicted_output: Gaussian,
    /// Common-space updates per mode (`None` when coasting).
    pub updates: Option<Vec<UpdateOutput>>,
    /// Each mode's estimate after the cycle, in its own state.
    pub leaves: Vec<Gaussian>,
    /// Mode probabilities after the cycle.
    pub mode_probabilities: DVector<f64>,
    /// Combined posterior common estimate (`None` when coasting).
    pub combined: Option<Gaussian>,
}

impl PolicyCycle {
    /// Log-likelihood of every mode (panics when coasting).
    pub fn log_likelihoods(&self) -> Vec<f64> {
        self.updates
            .as_ref()
            .expect("cycle coasted")
            .iter()
            .map(|u| u.log_likelihood)
            .collect()
    }
}

fn policy_predicted_output(
    modes: &[LinearMode],
    predicted: &[Gaussian],
    mu: &DVector<f64>,
    c: &DVector<f64>,
    policy: &CurrentPolicy,
) -> Gaussian {
    let weights = match policy.predicted_output {
        PredictedOutputWeighting::PriorModeProbabilities => mu,
        PredictedOutputWeighting::PredictedModeProbabilities => c,
    };
    let marginals = common_marginals(modes, predicted);
    let refs: Vec<&Gaussian> = marginals.iter().collect();
    combine(&refs, weights.as_slice())
}

/// Common-space update of every predicted mode, re-expanded by the policy.
fn policy_updates(
    modes: &[LinearMode],
    predicted: &[Gaussian],
    m: &Measurement,
    policy: &CurrentPolicy,
) -> (Vec<UpdateOutput>, Vec<Gaussian>) {
    let updates: Vec<UpdateOutput> = common_marginals(modes, predicted)
        .iter()
        .map(|g| joseph_update(g, m))
        .collect();
    let leaves = modes
        .iter()
        .zip(&updates)
        .map(|(mode, u)| reconstruct(mode, &u.posterior, policy))
        .collect();
    (updates, leaves)
}

fn coast_mode_probabilities(
    mu: &DVector<f64>,
    c: &DVector<f64>,
    policy: &CurrentPolicy,
) -> DVector<f64> {
    match policy.coast {
        CoastPolicy::KeepModeProbabilities => mu.clone(),
        CoastPolicy::PropagateModeProbabilities => c.clone(),
    }
}

/// The interaction and prediction phases of a current-policy cycle.
struct PolicyPrediction {
    c: DVector<f64>,
    weights: DMatrix<f64>,
    mixed_common: Vec<Gaussian>,
    mixed_native: Vec<Gaussian>,
    predicted_native: Vec<Gaussian>,
    predicted_output: Gaussian,
}

fn policy_predict(
    modes: &[LinearMode],
    leaves: &[Gaussian],
    mu: &DVector<f64>,
    pi: &DMatrix<f64>,
    policy: &CurrentPolicy,
) -> PolicyPrediction {
    let commons = common_marginals(modes, leaves);
    let c = predicted_mode_probabilities(pi, mu);
    let weights = mixing_weights_above(pi, mu, &c, policy.mixing_threshold);
    let mixed_common: Vec<Gaussian> = (0..modes.len())
        .map(|j| mix_destination(&commons, &weights, j).estimate)
        .collect();
    let mixed_native: Vec<Gaussian> = modes
        .iter()
        .zip(&mixed_common)
        .map(|(mode, g)| reconstruct(mode, g, policy))
        .collect();
    let predicted_native: Vec<Gaussian> = modes
        .iter()
        .zip(&mixed_native)
        .map(|(mode, g)| mode.predict(g))
        .collect();
    let predicted_output = policy_predicted_output(modes, &predicted_native, mu, &c, policy);
    PolicyPrediction {
        c,
        weights,
        mixed_common,
        mixed_native,
        predicted_native,
        predicted_output,
    }
}

/// One cycle under an explicit current policy. With `measurement = None` the
/// cycle coasts: leaves are the predictions and the mode probabilities follow
/// the coast policy.
pub fn policy_cycle(
    modes: &[LinearMode],
    leaves: &[Gaussian],
    mu: &DVector<f64>,
    pi: &DMatrix<f64>,
    measurement: Option<&Measurement>,
    policy: &CurrentPolicy,
) -> PolicyCycle {
    let p = policy_predict(modes, leaves, mu, pi, policy);
    let (updates, leaves, mode_probabilities, combined) = match measurement {
        None => (
            None,
            p.predicted_native.clone(),
            coast_mode_probabilities(mu, &p.c, policy),
            None,
        ),
        Some(m) => {
            let (updates, leaves) = policy_updates(modes, &p.predicted_native, m, policy);
            let lls: Vec<f64> = updates.iter().map(|u| u.log_likelihood).collect();
            let posterior = policy_mode_update(&p.c, &lls, policy);
            let marginals = common_marginals(modes, &leaves);
            let refs: Vec<&Gaussian> = marginals.iter().collect();
            let combined = combine(&refs, posterior.as_slice());
            (Some(updates), leaves, posterior, Some(combined))
        }
    };
    PolicyCycle {
        c: p.c,
        weights: p.weights,
        mixed_common: p.mixed_common,
        mixed_native: p.mixed_native,
        predicted_native: p.predicted_native,
        predicted_output: p.predicted_output,
        updates,
        leaves,
        mode_probabilities,
        combined,
    }
}
