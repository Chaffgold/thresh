# thresh-filter

Kalman filter family with configurable motion models for the thresh tracking
stack.

## Filters

| Filter | Type | Use it for |
|---|---|---|
| `kf` — Kalman Filter | Linear | Linear-Gaussian models; cheapest. |
| `ekf` — Extended KF | Jacobian linearization | Mildly nonlinear models; default IMM leaf. |
| `ukf` — Unscented KF | `2n + 1` weighted sigma points | Strongly nonlinear models; tunable via `UkfParams` (`alpha`, `beta`, `kappa`). |
| `ckf` — Cubature KF | `2n` equally-weighted cubature points | Strongly nonlinear models where robustness matters more than tuning flexibility. Parameter-free. |

All filters share the same surface: `new(x, p)` (UKF also takes `UkfParams`),
`predict(&dyn MotionModel, dt)`, `update_linear(z, h, r)` / closure `update`,
and public `x` / `p`.

Motion models (`models/`): CV, CA, CTRV, CT. The IMM filter (`imm`) blends a
bank of model-conditioned filters by Bayesian mode probability.

## CKF vs UKF — when to choose which

Both are third-order-accurate deterministic-sample nonlinear filters and are
empirically interchangeable on linear and mildly nonlinear problems. Reach for
the **UKF** when you need the extra spread/prior flexibility its `(alpha,
beta, kappa)` knobs give and you are willing to tune them. Reach for the
**CKF** when you want a *parameter-free* filter that cannot mis-tune: its `2n`
points are equally and positively weighted, so the predicted covariance stays
positive-definite under the standard rule — useful for thresh's corner cases
(ballistic re-entry's wide dynamic range, hard-maneuvering UAVs, OTHR
ionospheric refraction) where a poorly-chosen UKF `kappa` can drive the
covariance indefinite. When in doubt on aerospace targets, prefer CKF.

CKF reference: I. Arasaratnam and S. Haykin, "Cubature Kalman Filters", *IEEE
Transactions on Automatic Control*, vol. 54, no. 6, pp. 1254–1269, June 2009.

## IMM leaf filter

The IMM bank's model-conditioned leaf is selectable — EKF, UKF, or CKF — via
the `ImmLeafKind` chosen at construction:

```rust
use thresh_filter::imm::{ImmConfig, ImmFilter, ImmLeafKind};

// Default leaf is EKF — behaviour identical to the pre-pluggable bank.
let ekf_bank = ImmFilter::new(ImmConfig::cv_ca(5.0, 1.0), &x0, &p0);

// Or pick the leaf kind explicitly (every mode uses the same kind).
let ckf_bank =
    ImmFilter::with_leaf_kind(ImmConfig::cv_ca(5.0, 1.0), ImmLeafKind::Ckf, &x0, &p0);
```

The kind is owned by the `ImmFilter` instance and also drives the
common-space measurement-update leaf. `ImmFilter::new` defaults to
`ImmLeafKind::Ekf`, so existing callers and the `thresh-tracker` IMM path
are unaffected.

## IMM mathematics: specified, unspecified, verified

The `imm-estimation` capability specifies only the policy-independent IMM
mathematics. In the common state `[x, vx, y, vy, z, vz]` (m, m/s):

- **Mixing.** `c_j = Σ_i π_ij μ_i`; for every mode with `c_j > 1e-12`,
  `μ_i|j = π_ij μ_i / c_j`, `x0_j = Σ_i μ_i|j x_i` and
  `P0_j = Σ_i μ_i|j [P_i + (x_i − x0_j)(x_i − x0_j)ᵀ]`, including the
  between-mode spread. A mode whose state contains the common components
  (such as CA) contributes their marginal. For a 1 s cycle, `Π` is the
  configured transition matrix.
- **Log-likelihood.** `ℓ = −½ (m ln 2π + ln det S + yᵀ S⁻¹ y)` with
  `y = z − H x̂` and `S = H P Hᵀ + R`, evaluated in log space.
- **Posterior.** With floor 0, every `c_j > 1e-12` and every `|ℓ_j| ≤ 1e6`,
  the mode update is the exact Bayes posterior `μ⁺_j ∝ c_j exp(ℓ_j)`,
  normalised by log-sum-exp, so underflowing likelihoods give finite
  probabilities (for example `[0, 1]`), never NaN, and never a fallback to a
  uniform distribution or to `c` in place of the posterior. With floor 0, a mode with `c_j = 0` gets zero when every other
  `c_k > 1e-12` and its log-likelihood (from whatever estimate it is
  assigned) does not exceed the largest among the others.
- **Combination.** The output is the moment match
  `x = Σ_j μ⁺_j x_j`, `P = Σ_j μ⁺_j [P_j + (x_j − x)(x_j − x)ᵀ]` under
  whichever mode probabilities are in effect, including probabilities
  supplied by the learned classifier.

**Not specified.** These are current implementation choices, not contract.
The follow-up OpenSpec change `correct-native-imm-mode-policies` changes
them deliberately:

- the default mode-probability floor (0.01) and the renormalisation after
  flooring;
- the treatment of a mode whose predicted probability is at most 1e-12,
  including zero: its mixed estimate, its likelihood, its weight in the mode
  update, and the thresholds that treat a small probability as zero (1e-30
  in mixing, 1e-300 in the mode update), beyond the zero-probability case
  above;
- the weighting of the predicted (pre-update) combined estimate used for
  gating (currently the prior `μ`);
- mode-probability propagation when a prediction is not followed by an
  update (currently left unchanged);
- the reconstruction of a mode's own state from common-state moments,
  including components outside the common state such as acceleration
  (currently mean 0, variance 100 (m/s²)²) and turn rate;
- the common-state representation of speed, heading and turn-rate modes
  (CTRV, CT), including their vertical channel (currently `z = vz = 0` with
  variance 100);
- how the transition matrix depends on the elapsed time, other than for a
  1 s cycle (currently applied as configured for every `dt`);
- behaviour when an innovation covariance is not positive definite or is
  numerically singular.

**Verification.** `tests/imm_reference.rs` checks the specified mathematics
against hand-derived exact rationals and an independent reference
(`tests/support/imm_oracle.rs`, which uses only `nalgebra` and `std` and
shares no code with this crate), at an absolute tolerance of 1e-10 plus a
relative tolerance of 1e-9, with probability sums within 1e-12. Its native
tests use only configurations that depend on none of the policies above.
`tests/imm_current_policy.rs` pins today's behaviour under those policies;
its tests are prefixed `current_policy_` and are characterisation, not
evidence for the specification.

```sh
cargo test -p thresh-filter --test imm_reference --test imm_current_policy
# Print every compared residual:
IMM_REFERENCE_RESIDUALS=1 cargo test -p thresh-filter --test imm_reference -- --nocapture
```

**Validation.** `ImmConfig::validate` rejects a configuration with no models;
with mismatched counts or dimensions; with any transition-matrix entry that
is non-finite (NaN, ±∞) or negative; with a transition-matrix row that does
not sum to one within 1e-6; with any initial mode probability that is
non-finite or negative; or with initial probabilities that do not sum to one
within 1e-6. The entry checks are new: a NaN entry, or a row such as
`[1.2, −0.2]` that sums to one, used to pass. The error names the offending
entry, for example `transition matrix entry (0, 1)` or
`initial mode probability 1`. Every library preset still validates, and
every configuration that still validates behaves exactly as before.

`ImmFilter::new`, `ImmFilter::with_leaf_kind` and
`thresh_tracker::tracker::MultiObjectTracker::new_imm_position` panic on a
rejected configuration (see their `# Panics` sections), while
`MultiObjectTracker::new_imm_position_learned` returns the error. Call
`validate()` first to handle an invalid configuration without panicking.

## `learned-imm` feature (Track B)

The optional `learned-imm` feature (off by default) adds `imm_adapter`, which
loads an ONNX mode classifier via `thresh-inference` and lets the IMM use
*learned* mode probabilities instead of the analytic Markov update:

```sh
cargo build -p thresh-filter --features learned-imm   # pulls the ort/ONNX stack
```

`ImmModeAdapter::from_onnx(path)` loads the checkpoint; `LearnedImmFilter` wraps
an `ImmFilter`, feeding it the classifier's probabilities once a full 10-step
window of filter-state projections (`project_filter_state`, 13-dim: state,
covariance diagonal, and elapsed seconds) exists, and
falling back to the analytic update on any error. The core `ImmFilter` is
untouched, so the analytic test suite passes unchanged with the feature on. The
training/export pipeline lives under `python/` (see `TRAINING.md`); the feature
is feature-gated because the `ort`/ONNX-Runtime stack is heavy. Ships off by
default until the trained checkpoint meets its exit criterion.

History contains post-birth measurement updates (including tentative tracks),
not births or prediction-only states. The first elapsed value is zero; later
values include all prediction intervals since the previous update. The adapter
rejects legacy 12-wide ONNX models at load time with a re-export instruction.
