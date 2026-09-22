# Capability: Learned Tracker Components

## Purpose

A family of small ONNX models that augment the existing classical Bayesian tracker. Three candidate components are scoped: an IMM mode-probability classifier (implemented in this change), a learned association cost network (designed but not implemented here), and a short-horizon trajectory predictor (designed but not implemented here). Each component is opt-in via a Cargo feature gate; the classical pipeline remains the default.

**Training-data architecture.** ADS-B reports are **system-level truth**, not sensor measurements. A model that runs inside a filter at deployment time must be trained on inputs that match the filter's deployment-time inputs — not on raw ADS-B states, whose noise / smoothing / rate characteristics differ. Therefore the training pipeline for every component in this capability runs the full **ADS-B truth → `thresh-synth` measurements → classical tracker → filter state** chain and trains on the filter-state outputs. The classifier never sees a raw ADS-B state vector as an input.

## ADDED Requirements

### Requirement: IMM mode classifier — model contract

The IMM mode classifier ONNX model MUST conform to the following contract. The input is a **filter-state history**, not a raw trajectory; it carries both the state vector and a projection of the covariance produced by the classical Kalman/IMM filter.

| Tensor | Shape | dtype | Semantics |
|---|---|---|---|
| `features` (input) | `(batch, 10, 13)` | float32 | 10 measurement-updated filter-state snapshots: common-space state `[x, vx, y, vy, z, vz]`, 6 covariance-diagonal variances, then elapsed seconds since the preceding recorded update. State is sensor-ENU in SI units; the first post-birth update has elapsed time zero (Decision 30). Produced by `thresh_filter::imm::project_filter_state` (`CLASSIFIER_FEATURE_DIM = 13`). |
| `mode_probs` (output) | `(batch, 4)` | float32 | Softmaxed probabilities over `[CV, CA, CTRV, coord_turn]` |

#### Scenario: Contract verification

**WHEN** the `onnx-tests` workflow runs against `test-data/models/imm_mode_classifier.onnx`

**THEN** the workflow asserts the model's input shape is exactly `(batch, 10, 13)` — the trailing dimension MUST equal `thresh_filter::imm::CLASSIFIER_FEATURE_DIM` (13) so a model whose feature width disagrees with the runtime projection cannot pass — output shape is `(batch, 4)`, and output values sum to 1.0 per batch row within 1e-5 tolerance

**SHALL** fail the build on any shape, name, or normalisation violation.

### Requirement: Shared elapsed-time and observation-history contract

Training and deployment MUST record post-birth measurement updates, including
tentative tracks, and MUST NOT append birth initialization or prediction-only
states. The first recorded row MUST have elapsed time zero. Each later row
MUST contain all elapsed seconds since the preceding recorded update; sliding
windows MUST preserve that value rather than resetting their first row.
Runtime prediction intervals MUST be finite and nonnegative, and invalid or
overflowing intervals MUST be rejected before mutating the filter. Legacy
12-wide datasets and models MUST fail explicitly with a regeneration or
re-export instruction rather than assuming a sample rate.

#### Scenario: Different sample rates remain distinguishable

**WHEN** identical filter-state sequences are sampled at 1 Hz and 10 Hz

**THEN** the corresponding elapsed features after the initial zero are 1.0
and 0.1 seconds respectively in both training and runtime windows.

#### Scenario: Missed measurements accumulate time

**WHEN** three predictions of 0.1 seconds occur between two measurement updates

**THEN** only the two updated states enter history, and the second carries
0.3 seconds regardless of the missing intermediate measurements.

#### Scenario: Training timestamps contradict stored intervals

**WHEN** a dataset contains nonfinite timestamps, or elapsed
features that disagree with consecutive timestamps for its track

**THEN** Python rejects the dataset rather than producing training windows.

#### Scenario: Multiple updates at the same timestamp

**WHEN** consecutive updates share a finite timestamp

**THEN** both training and deployment retain their update order and record
zero elapsed seconds for each update after the first at that timestamp.

### Requirement: Training data must come from filter outputs, not raw ADS-B

The IMM mode classifier MUST be trained on inputs produced by running the classical tracker over synthesised measurements from `thresh-synth`. The training dataset MUST NOT contain raw ADS-B state vectors as classifier inputs. Analytic mode labels may be derived from ADS-B trajectory kinematics (truth labels), but the input features MUST be filter outputs.

#### Scenario: Dataset construction check

**WHEN** the training dataset for the IMM classifier is constructed

**THEN** every input feature window in the dataset is traceable to a `thresh-tracker` run over `thresh-synth`-produced measurements; no input window is sourced directly from an ADS-B state-vector record

**SHALL** be enforced by a unit test in `python/training/test_imm_dataset.py` that fails if any dataset row's provenance metadata says `source = "adsbx"` or `source = "opensky"` in the input-features column (truth labels may still cite those sources for the label column).

#### Scenario: Adapter rejects raw measurements at runtime

**WHEN** the `learned-imm` adapter is fed an input that does not match the `filter_state_dim` (e.g. a 9-dim raw state without covariance, or a measurement-level vector)

**THEN** the adapter returns an error and the learned filter wrapper retains the analytic update; it does NOT silently produce mode probabilities from a malformed input

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

`thresh-filter` MUST expose a Cargo feature `learned-imm` that, when enabled, provides an adapter loading the ONNX mode classifier and a wrapper using its output to replace the IMM's posterior mode-probability estimate. The analytic transition matrix and interaction-step mixing MUST remain unchanged, as defined in design.md Decision 19.

#### Scenario: Feature off — classical IMM unchanged

**WHEN** the `learned-imm` feature is disabled

**THEN** the existing IMM filter implementation behaves identically to the pre-change version

**SHALL** pass the existing IMM test suite without modification.

#### Scenario: Feature on — learned mode probabilities feed IMM

**WHEN** the `learned-imm` feature is enabled, a trained classifier ONNX checkpoint is loaded, and a full 10-step window of filter-state projections is available

**THEN** after the analytic measurement update, the learned filter wrapper replaces the posterior `mode_probabilities` with the classifier's `mode_probs` output and re-runs `combine()` to re-blend the bank's estimates, without replacing the analytic transition matrix or interaction-step mixing

**SHALL** retain the analytic update during the warm-up window and fall back to the analytic update with a warning if ONNX inference fails at runtime.

### Requirement: Exit criteria for IMM classifier

The IMM mode classifier ONNX checkpoint MUST be checked into `test-data/models/imm_mode_classifier.onnx` only when all of the following hold:

- Classifier accuracy ≥ 0.70 against analytic labels on a held-out trajectory split.
- With `learned-imm` enabled, the existing `thresh-filter` IMM test suite still passes.
- With `learned-imm` enabled, the downstream tracker MOTA on `thresh-eval`'s ADS-B scenario is no worse than with analytic mode probabilities.

Distribution MUST also satisfy the documented rights requirement in `LICENSING.md`.

#### Scenario: Shipping with no MOTA regression

**WHEN** the three exit criteria are met and the intended training and checkpoint distribution rights are documented

**THEN** the trained classifier is committed at `test-data/models/imm_mode_classifier.onnx`, the model card is updated, and the `learned-imm` feature is documented as ready in `crates/thresh-filter/README.md`

**SHALL** include benchmark numbers and the training-data provenance in the model card.
**SHALL** record in the model card the checkpoint's data lineage, release terms, and the basis for its intended training and distribution rights as defined under "Trained models" in `LICENSING.md`.

#### Scenario: Distribution rights remain unverified

**WHEN** the metric thresholds are met but the intended training or checkpoint distribution rights remain unverified

**THEN** the random stub remains in place and the trained checkpoint is not committed or published until the applicable rights are documented

#### Scenario: MOTA regression blocks checkpoint release

**WHEN** the first two exit criteria are met but downstream MOTA regresses, even if the intended training and checkpoint distribution rights are documented

**THEN** the random stub remains at `test-data/models/imm_mode_classifier.onnx`, the trained checkpoint is not committed or published, and the regression is documented in `design.md` for further training and evaluation

**SHALL NOT** waive the no-regression gate by labeling the trained model experimental or leaving `learned-imm` disabled by default.

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
