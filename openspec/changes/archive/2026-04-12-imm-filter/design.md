## Context

thresh-filter implements four motion models (CV, CA, CTRV, CT) and three filter types (KF, EKF, UKF), but each track uses a single fixed model for its entire lifetime. When a target changes behavior mid-track -- straight flight to coordinated turn, cruise to acceleration -- the fixed model's assumptions diverge from reality. A CV filter lags behind a turning target; a CT filter hallucinates maneuvers during straight flight. The standard solution in defense and aerospace tracking is the Interacting Multiple Model (IMM) estimator, which maintains a bank of model-conditioned filters and blends their outputs using Bayesian mode probabilities updated at every timestep.

The four motion models already implement the `MotionModel` trait (and `LinearModel` for CV/CA), providing `predict`, `jacobian`, `process_noise`, and `transition_matrix`. The IMM can wrap these existing implementations without modifying them.

A significant design challenge is that the models have different state dimensions: CV is 6D `[x, vx, y, vy, z, vz]`, CA is 9D `[x, vx, ax, y, vy, ay, z, vz, az]`, CTRV is 5D `[x, y, theta, v, omega]`, and CT is 5D `[x, vx, y, vy, omega]`. The IMM interaction step must map states between these representations when mixing.

## Goals / Non-Goals

**Goals:**
- Implement the full IMM cycle (interaction, predict, update, mode probability update, combination) in `thresh-filter`
- Support arbitrary combinations of existing motion models, including mixed linear/nonlinear
- Handle heterogeneous state dimensions via explicit state mapping between models
- Provide default configurations for common model pairings (CV+CA, CV+CTRV, CV+CA+CTRV+CT)
- Integrate with `MultiObjectTracker` in `thresh-tracker` so IMM-based tracks work end-to-end
- Report the dominant motion mode per track for situational awareness

**Non-Goals:**
- Variable-structure IMM (adding/removing models at runtime)
- Online learning or adaptation of the transition probability matrix
- GPU acceleration of the IMM cycle
- Particle filter or other non-Gaussian model-mixing approaches
- Automatic selection of the model set based on target class

## Decisions

### 1. `ImmFilter` lives in `thresh-filter` as a peer to `KalmanFilter` and `ExtendedKalmanFilter`

**Decision:** Add `crates/thresh-filter/src/imm.rs` and re-export from `lib.rs` as `pub mod imm`. The `ImmFilter` struct is not a motion model -- it is a composite filter that owns model-conditioned filter instances.

**Rationale:** The IMM is a filter-level construct, not a motion model. It wraps KF/EKF instances paired with motion models. Placing it in `thresh-filter` keeps the dependency direction clean -- `thresh-tracker` depends on `thresh-filter`, not vice versa.

### 2. Dynamic dispatch for heterogeneous model sets

**Decision:** `ImmFilter` stores models as `Vec<Box<dyn MotionModel>>` and filter instances as `Vec<ModelConditionedFilter>`, where `ModelConditionedFilter` holds an EKF (which subsumes KF via `update_linear`). No const generic `N` parameter despite the proposal.

**Rationale:** The proposal suggested `ImmFilter<const N: usize>`, but the models have different state dimensions (CV=6D, CA=9D, CTRV=5D, CT=5D). Const generics would require all models to share the same state dimension, which defeats the purpose. Dynamic dispatch via `dyn MotionModel` accommodates heterogeneous state sizes with negligible performance cost (the IMM cycle is O(N^2) in the number of models, where N is 2-4).

### 3. Explicit state mapping between model representations

**Decision:** Define a `StateMapping` trait with methods `to_common(&self, model_state) -> CommonState` and `from_common(&self, common_state) -> model_state`. Each model pairing in `ImmConfig` includes its mapping. The common state representation is `[x, vx, y, vy, z, vz]` (the CV state), since position and velocity are the universally shared quantities.

```
pub trait StateMapping: Send + Sync {
    fn to_common(&self, state: &DVector<f64>, cov: &DMatrix<f64>) -> (DVector<f64>, DMatrix<f64>);
    fn from_common(&self, state: &DVector<f64>, cov: &DMatrix<f64>) -> (DVector<f64>, DMatrix<f64>);
    fn common_dim(&self) -> usize;
    fn model_dim(&self) -> usize;
}
```

**Rationale:** The IMM interaction step must mix states and covariances across models. When CV (6D) and CA (9D) interact, we cannot directly average their state vectors. The mapping layer projects into a common space for mixing, then back-projects to each model's native representation. This is the standard approach in heterogeneous-state IMM implementations (Bar-Shalom & Li, Chapter 11).

Provide built-in mappings: `CvMapping` (identity), `CaMapping` (drop/add acceleration), `CtrvMapping` (Cartesian to heading/speed conversion), `CtMapping` (reorder to common).

### 4. `ImmConfig` bundles model set, TPM, initial probabilities, and mappings

**Decision:** A configuration struct that fully specifies an IMM instance:

```
pub struct ImmConfig {
    pub models: Vec<Box<dyn MotionModel>>,
    pub mappings: Vec<Box<dyn StateMapping>>,
    pub transition_matrix: DMatrix<f64>,  // N x N Markov TPM
    pub initial_mode_probabilities: DVector<f64>,  // N-vector, sums to 1
}
```

**Rationale:** Separating configuration from state lets the tracker create new IMM instances from a shared config template when birthing tracks. The TPM encodes prior knowledge about mode-switching frequency (e.g., high self-transition probability = modes are sticky).

### 5. Per-step cycle: interaction -> predict -> update -> mode update -> combine

**Decision:** The IMM cycle follows the standard five-step algorithm (Blom & Bar-Shalom, 1988):

1. **Interaction:** Compute mixing probabilities `mu_{i|j} = (pi_{ij} * mu_i) / c_j` where `c_j = sum_i(pi_{ij} * mu_i)`. Mix states and covariances in common space: `x_j^0 = sum_i(mu_{i|j} * T_i(x_i))`, `P_j^0 = sum_i(mu_{i|j} * (P_i + delta_i * delta_i'))` where `delta_i = T_i(x_i) - x_j^0` and `T_i` is the state mapping for model i.
2. **Model-conditioned prediction:** Each EKF predicts with its own motion model.
3. **Model-conditioned update:** Each EKF updates with the measurement. Compute the innovation likelihood `Lambda_j = N(y_j; 0, S_j)` for mode probability update.
4. **Mode probability update:** `mu_j = c_j * Lambda_j / sum_k(c_k * Lambda_k)`.
5. **State/covariance combination:** `x = sum_j(mu_j * T_j(x_j))`, `P = sum_j(mu_j * (P_j + delta_j * delta_j'))`, both in common space.

The combined output is in the common 6D representation for use by the tracker's association and lifecycle logic.

### 6. Innovation likelihood computation

**Decision:** Compute the multivariate Gaussian likelihood from each model's innovation `y_j` and innovation covariance `S_j`: `Lambda_j = (2*pi)^{-m/2} * |S_j|^{-1/2} * exp(-0.5 * y_j' * S_j^{-1} * y_j)`. Use log-likelihood internally and exponentiate at the normalization step to avoid underflow.

**Rationale:** Direct likelihood computation underflows for high-dimensional measurements or large innovations. Log-space arithmetic with the log-sum-exp trick is numerically stable.

### 7. Integration with `MultiObjectTracker`

**Decision:** Add `new_imm_position(config, measurement_noise_sigma, gate_threshold)` constructor on `MultiObjectTracker`. The tracker stores an `ImmConfig` and uses it to birth IMM-based tracks. The `Track` struct gains an optional `dominant_mode: Option<usize>` field set during each update.

The `step()` method is refactored so that predict/update logic is dispatched based on whether the tracker was constructed with a single model or an IMM config. The observation matrix `H` operates on the common state representation.

**Rationale:** This mirrors the existing `new_cv_position` pattern. The IMM's combined state/covariance (in common 6D space) is stored in `Track::state` and `Track::covariance`, so association (Mahalanobis distance) and lifecycle management work unchanged.

### 8. Default configurations with literature-standard TPMs

**Decision:** Provide three factory methods on `ImmConfig`:

- `ImmConfig::cv_ca(sigma_a, sigma_j)` -- 2-model, TPM `[[0.95, 0.05], [0.05, 0.95]]`
- `ImmConfig::cv_ctrv(sigma_a, sigma_v, sigma_omega)` -- 2-model, TPM `[[0.95, 0.05], [0.05, 0.95]]`
- `ImmConfig::cv_ca_ctrv_ct(...)` -- 4-model, TPM with 0.90 self-transition, 0.033 cross-transition

**Rationale:** These cover the most common aerospace tracking scenarios. The 0.95 self-transition probability means mode switches are expected roughly every 20 timesteps on average, which matches typical maneuvering target behavior. Users can override the TPM for their specific scenario.

## Risks / Trade-offs

**[Risk] State mapping introduces approximation error.** Mapping between CV (6D) and CTRV (5D) requires converting Cartesian velocities to heading/speed and back. This is exact for the state but approximate for the covariance (the Jacobian of the mapping is used). Mitigation: document the approximation; for most tracking scenarios the error is small relative to process noise.

**[Risk] Computational cost scales quadratically with model count.** The interaction step is O(N^2) in the number of models, and each model runs a full predict/update. For N=4, this is 4x the cost of a single-model filter. Mitigation: N is small (2-4) in practice. Profile and optimize if needed.

**[Trade-off] Dynamic dispatch vs static dispatch.** Using `dyn MotionModel` instead of generics adds vtable overhead per predict/update call. However, the matrix math dominates runtime, and the flexibility to mix linear and nonlinear models in one IMM outweighs the nanosecond-level dispatch cost.

**[Trade-off] Common state dimension fixed at 6D.** Choosing CV's `[x, vx, y, vy, z, vz]` as the common representation means the combined output always has 6 dimensions. Models with richer state (CA's acceleration, CT's turn rate) lose those extra components in the output. This is standard practice -- the combined state represents the best position/velocity estimate, and mode-specific quantities are available from the individual model-conditioned filters if needed.

**[Risk] Numerically degenerate mode probabilities.** If one model fits much better than others over many timesteps, its mode probability approaches 1.0 and the others approach 0.0. This starves the minority models of information, so they cannot recover quickly when the target switches mode. Mitigation: clamp mode probabilities to a minimum floor (e.g., 0.01) to ensure all models remain viable.

## Open Questions

- Should the mode probability floor be configurable, or hardcoded at a sensible default (e.g., 0.01)?
- Should `ImmFilter` expose per-model state/covariance for diagnostics, or only the combined output?
- For 2D-only tracking scenarios (no z-axis), should the common state be 4D `[x, vx, y, vy]`? Or always 6D with z zeroed?
- Should the `Track` struct store the full mode probability vector, or just the dominant mode index?
- Is there value in supporting UKF as a model-conditioned filter (in addition to KF/EKF), or is EKF sufficient for all current motion models?
