## Capability: Learned Tracker Components

### Overview

A family of small ONNX models that augment the existing classical Bayesian tracker. Three candidate components are scoped: an IMM mode-probability classifier (implemented in this change), a learned association cost network (designed but not implemented here), and a short-horizon trajectory predictor (designed but not implemented here). Each component is opt-in via a Cargo feature gate; the classical pipeline remains the default.

**Training-data architecture.** ADS-B reports are **system-level truth**, not sensor measurements. A model that runs inside a filter at deployment time must be trained on inputs that match the filter's deployment-time inputs — not on raw ADS-B states, whose noise / smoothing / rate characteristics differ. Therefore the training pipeline for every component in this capability runs the full **ADS-B truth → `thresh-synth` measurements → classical tracker → filter state** chain and trains on the filter-state outputs. The classifier never sees a raw ADS-B state vector as an input.

## ADDED Requirements

### Requirement: IMM mode classifier — model contract

The IMM mode classifier ONNX model MUST conform to the following contract. The input is a **filter-state history**, not a raw trajectory; it carries both the state vector and a projection of the covariance produced by the classical Kalman/IMM filter.

| Tensor | Shape | dtype | Semantics |
|---|---|---|---|
| `filter_state_history` (input) | `(batch, 10, filter_state_dim)` | float32 | 10 consecutive filter-state snapshots. Default `filter_state_dim` = 18: state `[x, y, z, vx, vy, vz, ax, ay, az]` concatenated with the 9 diagonal entries of the state covariance. All values in sensor-ENU frame and SI units. |
| `mode_probs` (output) | `(batch, 4)` | float32 | Softmaxed probabilities over `[CV, CA, CTRV, coord_turn]` |

#### Scenario: Contract verification

**WHEN** the `onnx-tests` workflow runs against `test-data/models/imm_mode_classifier.onnx`

**THEN** the workflow asserts the model's input shape is `(batch, 10, 18)` (or whatever `filter_state_dim` the trained model uses, documented in the model card), output shape is `(batch, 4)`, and output values sum to 1.0 per batch row within 1e-5 tolerance

**SHALL** fail the build on any shape, name, or normalisation violation.

### Requirement: Training data must come from filter outputs, not raw ADS-B

The IMM mode classifier MUST be trained on inputs produced by running the classical tracker over synthesised measurements from `thresh-synth`. The training dataset MUST NOT contain raw ADS-B state vectors as classifier inputs. Analytic mode labels may be derived from ADS-B trajectory kinematics (truth labels), but the input features MUST be filter outputs.

#### Scenario: Dataset construction check

**WHEN** the training dataset for the IMM classifier is constructed

**THEN** every input feature window in the dataset is traceable to a `thresh-tracker` run over `thresh-synth`-produced measurements; no input window is sourced directly from an ADS-B state-vector record

**SHALL** be enforced by a unit test in `python/training/test_imm_dataset.py` that fails if any dataset row's provenance metadata says `source = "adsbx"` or `source = "opensky"` in the input-features column (truth labels may still cite those sources for the label column).

#### Scenario: Adapter rejects raw measurements at runtime

**WHEN** the `learned-imm` adapter is fed an input that does not match the `filter_state_dim` (e.g. a 9-dim raw state without covariance, or a measurement-level vector)

**THEN** the adapter returns an error and falls back to the analytic transition matrix; it does NOT silently produce mode probabilities from a malformed input

**SHALL** log the rejection at WARN level with the observed input shape.

### Requirement: IMM mode label derivation

The training pipeline MUST derive analytic mode labels for each trajectory state using kinematic thresholds, against which the learned classifier is trained and evaluated.

#### Scenario: Labeling a CV segment

**WHEN** a trajectory state has speed change rate < 1 m/s² and turn rate < 1°/s over a 5-second window

**THEN** the label for that state is `CV` (constant velocity)

**SHALL** be applied consistently across the entire trajectory window.

#### Scenario: Labeling a coordinated-turn segment

**WHEN** a trajectory state has turn rate > 3°/s sustained over a 5-second window with bank-angle inference > 15°

**THEN** the label for that state is `coord_turn`

**SHALL** take precedence over `CTRV` when both criteria are met.

### Requirement: `learned-imm` feature gate in `thresh-filter`

`thresh-filter` MUST expose a Cargo feature `learned-imm` that, when enabled, provides an adapter loading the ONNX mode classifier and using it as the IMM's mode-transition input.

#### Scenario: Feature off — classical IMM unchanged

**WHEN** the `learned-imm` feature is disabled

**THEN** the existing IMM filter implementation behaves identically to the pre-change version

**SHALL** pass the existing IMM test suite without modification.

#### Scenario: Feature on — learned mode probabilities feed IMM

**WHEN** the `learned-imm` feature is enabled and a trained classifier ONNX checkpoint is loaded

**THEN** the IMM uses the classifier's `mode_probs` output as its per-step mode-transition input, replacing the analytic mode-transition matrix

**SHALL** fall back to the analytic transition matrix and log a warning if the ONNX inference fails at runtime.

### Requirement: Exit criteria for IMM classifier

The IMM mode classifier ONNX checkpoint MUST be checked into `test-data/models/imm_mode_classifier.onnx` only when all of the following hold:

- Classifier accuracy ≥ 0.70 against analytic labels on a held-out trajectory split.
- With `learned-imm` enabled, the existing `thresh-filter` IMM test suite still passes.
- With `learned-imm` enabled, the downstream tracker MOTA on `thresh-eval`'s ADS-B scenario is no worse than with analytic mode probabilities.

#### Scenario: Shipping with no MOTA regression

**WHEN** the three exit criteria are met

**THEN** the trained classifier is committed at `test-data/models/imm_mode_classifier.onnx`, the model card is updated, and the `learned-imm` feature is documented as ready in `crates/thresh-filter/README.md`

**SHALL** include benchmark numbers and the training-data provenance in the model card.

#### Scenario: MOTA regression — ship with feature off by default

**WHEN** the first two exit criteria are met but downstream MOTA regresses

**THEN** the classifier is still committed at `test-data/models/imm_mode_classifier.onnx`, but the `learned-imm` feature is documented as experimental, defaults to off, and the regression is documented in `design.md`'s Open Questions section

**SHALL NOT** make `learned-imm` a default feature until the regression is resolved.

### Requirement: Future learned components — designed but deferred

The acquisition and toolchain infrastructure built in this change MUST be reusable for two additional learned-tracker components, scoped but not implemented here. The capability spec documents the integration shape for each:

- **Learned association cost network.** Inputs: a track's predicted state covariance and a detection's box + class + score. Output: a scalar cost in `[0, 1]` that augments or replaces Mahalanobis distance in `thresh-association`. Feature gate: `learned-association` in `thresh-association`.
- **Short-horizon trajectory predictor.** Inputs: a track's 10-step state history. Output: predicted state at `t + 2s`. Feature gate: `learned-predictor` in `thresh-tracker`. Use case: improved gating before measurement-to-track association under occlusion.

#### Scenario: Future contributor implementing learned-association

**WHEN** a future change implements `learned-association`

**THEN** it reuses the trajectory schema, the Python training toolchain, and the ONNX export utilities defined by this change

**SHALL** add its own capability spec but not need to revise the flight-data-acquisition spec.

### Requirement: Determinism

All learned tracker components MUST produce identical outputs when given identical inputs across runs. The ONNX export MUST disable any non-deterministic operators in the source PyTorch model.

#### Scenario: Reproducing inference output

**WHEN** the same `state_history` batch is fed to the mode classifier twice in the same process and across two separate processes

**THEN** the `mode_probs` output is byte-identical in both cases

**SHALL** be verified by a unit test in `python/eval/` that runs the model twice and asserts hash equality of the output tensor.
