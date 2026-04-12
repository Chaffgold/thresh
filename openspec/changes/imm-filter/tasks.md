## 1. State Mapping (thresh-filter)

- [x] 1.1 Define `StateMapping` trait in `crates/thresh-filter/src/imm.rs` with `to_common`, `from_common`, `common_dim`, `model_dim` methods. Common representation is 6D `[x, vx, y, vy, z, vz]`.
- [x] 1.2 Implement `CvMapping` (identity, 6D <-> 6D).
- [x] 1.3 Implement `CaMapping` (9D <-> 6D, drops/initializes acceleration components).
- [x] 1.4 Implement `CtrvMapping` (5D <-> 6D, converts heading/speed to Cartesian velocity and back, with Jacobian-based covariance transformation).
- [x] 1.5 Implement `CtMapping` (5D <-> 6D, reorders `[x, vx, y, vy, omega]` to common and adds zero z-axis, with covariance remapping).

## 2. IMM Configuration (thresh-filter)

- [x] 2.1 Define `ImmConfig` struct: `models: Vec<Box<dyn MotionModel>>`, `mappings: Vec<Box<dyn StateMapping>>`, `transition_matrix: DMatrix<f64>`, `initial_mode_probabilities: DVector<f64>`. Add validation (TPM rows sum to 1, probabilities sum to 1, lengths consistent).
- [x] 2.2 Implement `ImmConfig::cv_ca(sigma_a, sigma_j)` factory -- 2-model with TPM `[[0.95, 0.05], [0.05, 0.95]]`, uniform initial mode probabilities.
- [x] 2.3 Implement `ImmConfig::cv_ctrv(sigma_a, sigma_v, sigma_omega)` factory -- 2-model with TPM `[[0.95, 0.05], [0.05, 0.95]]`.
- [x] 2.4 Implement `ImmConfig::cv_ca_ctrv_ct(...)` factory -- 4-model with TPM 0.90 self-transition, 0.033 cross-transition.

## 3. IMM Filter Core (thresh-filter)

- [x] 3.1 Define `ModelConditionedFilter` struct holding an `ExtendedKalmanFilter`, a `Box<dyn MotionModel>`, and a `Box<dyn StateMapping>`. Linear models (CV, CA) use `EKF::update_linear`.
- [x] 3.2 Define `ImmFilter` struct: `filters: Vec<ModelConditionedFilter>`, `mode_probabilities: DVector<f64>`, `transition_matrix: DMatrix<f64>`, `mode_probability_floor: f64`.
- [x] 3.3 Implement `ImmFilter::new(config, initial_state, initial_covariance)` -- initializes each model-conditioned EKF by mapping the common initial state/covariance into each model's native representation.

## 4. IMM Step Cycle (thresh-filter)

- [x] 4.1 Implement interaction step: compute mixing probabilities `mu_{i|j}` from TPM and current mode probabilities. Mix states and covariances in common space, then back-project to each model's native state. Update each model-conditioned filter with mixed state/covariance.
- [x] 4.2 Implement model-conditioned prediction: call `ekf.predict(model, dt)` for each model-conditioned filter.
- [x] 4.3 Implement model-conditioned update: call `ekf.update_linear` or `ekf.update` for each filter with the measurement. Store the innovation `y_j` and innovation covariance `S_j` for likelihood computation.
- [x] 4.4 Implement innovation likelihood computation in log-space: `log(Lambda_j) = -0.5 * (m * ln(2*pi) + ln(|S_j|) + y_j' * S_j^{-1} * y_j)`. Use log-sum-exp for normalization.
- [x] 4.5 Implement mode probability update: `mu_j = c_j * Lambda_j / sum(c_k * Lambda_k)`. Clamp to floor value to prevent mode starvation.
- [x] 4.6 Implement state/covariance combination: compute weighted sum of model-conditioned estimates in common space. Return combined 6D state and covariance.
- [x] 4.7 Implement `ImmFilter::step(dt, z, h, r)` that orchestrates the full cycle (interaction -> predict -> update -> mode update -> combine) and returns the combined state, covariance, and dominant mode index.

## 5. Tracker Integration (thresh-tracker)

- [x] 5.1 Add `dominant_mode: Option<usize>` and `mode_probabilities: Option<DVector<f64>>` fields to `Track` struct.
- [x] 5.2 Add `new_imm_position(config, measurement_noise_sigma, gate_threshold)` constructor on `MultiObjectTracker`. Stores the `ImmConfig` for birthing new tracks.
- [x] 5.3 Refactor `MultiObjectTracker::step()` to dispatch predict/update through an internal enum (`TrackerMode::SingleModel` vs `TrackerMode::Imm`) so the association and lifecycle logic remains shared.
- [x] 5.4 Implement IMM-aware `birth_track`: create an `ImmFilter` from the stored config, initialize from the detection, store combined state/covariance in the new `Track`.

## 6. Tests

- [x] 6.1 Unit test: state mapping round-trip for each mapping type (CV, CA, CTRV, CT). Verify `from_common(to_common(x, P))` preserves shared state components.
- [x] 6.2 Unit test: `ImmConfig` validation rejects mismatched dimensions, non-stochastic TPM, probabilities not summing to 1.
- [x] 6.3 Unit test: interaction step with uniform mode probabilities and identical states produces unchanged states.
- [x] 6.4 Unit test: mode probability update shifts weight toward the model with smaller innovation.
- [x] 6.5 Unit test: 2-model CV+CA IMM on a straight-line trajectory converges to CV-dominant mode probability (mu_CV > 0.8 after 20 steps).
- [x] 6.6 Integration test: CV+CTRV IMM on a trajectory that switches from straight flight to coordinated turn. Verify mode probabilities flip and position RMSE is lower than either single-model filter alone.
- [x] 6.7 Integration test: 4-model IMM through `MultiObjectTracker::new_imm_position` -- birth, confirm, and maintain a maneuvering track over 50 steps. Verify `dominant_mode` changes when the target maneuvers.
- [x] 6.8 Integration test: IMM covariance stays positive semi-definite over 1000 steps with mode switching (eigenvalue check, same pattern as `kf_covariance_stays_psd`).
