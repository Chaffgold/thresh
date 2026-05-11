## Capability: Learned Point-Cloud Detector

### Overview

A PyTorch-trained DETR-family detector that consumes 3D point clouds and emits oriented 3D bounding boxes, plus a per-detection class index and confidence score. Training data is produced by driving `thresh-synth`'s radar simulator with real ADS-B trajectories from the acquisition layer. The trained model is exported to ONNX and replaces the random-weights stub at `test-data/models/test_detector.onnx`.

## ADDED Requirements

### Requirement: Trajectory-driven synth pairing

`thresh-synth` MUST provide an API that consumes a canonical trajectory and emits paired `(point_cloud, gt_boxes_3D, gt_classes)` snapshots at the configured sensor sample rate. The snapshots SHALL match the existing ONNX detector input/output shape contract exactly.

#### Scenario: Generating a paired snapshot from a straight-line trajectory

**WHEN** the synth pairing API is called with a constant-velocity trajectory at altitude 3000 m heading north at 250 kt, a sensor pose at origin in ENU frame, and a sample rate of 1 Hz

**THEN** each emitted snapshot contains a 1000-point point cloud with at least one cluster near the predicted target position, a single ground-truth box `[x, y, z, L, W, H, yaw]` matching the trajectory state, and a class index drawn from the trajectory's mapped class

**SHALL** produce point cloud values within physical units (positions in metres, intensity in `[0, 1]`) and box yaw in radians.

#### Scenario: Padding multiple targets to the fixed output size

**WHEN** a trajectory contains 7 simultaneous aircraft at one snapshot

**THEN** the snapshot's box tensor is padded to shape `(100, 7)` with a parallel validity mask of length 100 indicating which entries are real

**SHALL** order the real entries first in the tensor and zero the padded entries.

### Requirement: ONNX detector input/output contract

The exported ONNX detector MUST conform to the following input/output contract:

| Tensor | Shape | dtype | Semantics |
|---|---|---|---|
| `point_cloud` (input) | `(1, 1000, 4)` | float32 | `[x, y, z, intensity]` per point in sensor-ENU frame |
| `boxes` (output) | `(1, 100, 7)` | float32 | `[x, y, z, L, W, H, yaw]` per detection in sensor-ENU frame |
| `scores` (output) | `(1, 100, 1)` | float32 | Confidence in `[0, 1]` |
| `classes` (output) | `(1, 100, 1)` | int64 | Class index in `[0, 5)` per the thresh class enum |

#### Scenario: Contract verification in CI

**WHEN** the `onnx-tests` workflow runs against `test-data/models/test_detector.onnx`

**THEN** the workflow asserts the model's named inputs and outputs match the table above

**SHALL** fail the build on any shape, dtype, or name mismatch.

#### Scenario: Backward compatibility with a missing classes tensor

**WHEN** the Rust-side `thresh-inference` parser receives an ONNX model with no `classes` output

**THEN** the parser defaults the class index to 0 for every detection without erroring

**SHALL** be removed once Track A's trained checkpoint lands (the trained model emits `classes` natively).

### Requirement: PyTorch training script

The training script MUST be a single entry point under `python/training/train_detector.py` that reproduces the trained checkpoint from canonical trajectories using a fixed random seed and a pinned environment.

#### Scenario: One-command reproduction

**WHEN** a developer runs `uv sync && uv run python python/training/train_detector.py --config python/training/configs/detector_default.yaml`

**THEN** the script trains for the configured number of epochs, writes a PyTorch checkpoint and a TensorBoard log, and writes a summary JSON with final training and held-out metrics

**SHALL** be deterministic given the same `pyproject.toml`, `uv.lock`, config, and trajectory data.

### Requirement: ONNX export with parity verification

The training pipeline MUST include an export step that converts the best PyTorch checkpoint to ONNX and verifies output parity with the source PyTorch model.

#### Scenario: Export and parity check

**WHEN** `python/export/export_detector.py` is run on a trained checkpoint

**THEN** it writes the ONNX file and runs both PyTorch and ONNX Runtime on a fixture batch of 16 point clouds, asserting per-tensor max absolute difference < 1e-4

**SHALL** fail the export if parity is not met and SHALL print the per-output max-diff in the failure message.

### Requirement: Class taxonomy

The detector's class index output MUST map to the canonical five-bucket class enum: `light-fixed-wing`, `heavy-fixed-wing`, `rotorcraft`, `glider-or-balloon-or-uav`, `other`. The enum SHALL be defined in `python/training/classes.py` and mirrored in the Rust-side `thresh-core` as a `DetectionClass` enum.

#### Scenario: Round-tripping a class through ONNX

**WHEN** the detector outputs `classes[0, k] = 2` for some detection `k`

**THEN** both the Python decoder in `python/eval/` and the Rust decoder in `thresh-inference` interpret the value as `rotorcraft`

**SHALL** never emit a class index outside `[0, 5)`.

### Requirement: Exit criteria for replacing the random stub

The trained detector ONNX checkpoint MUST replace `test-data/models/test_detector.onnx` only when the held-out evaluation meets all of the following:

- mAP at IoU 0.5 (3D) ≥ 0.30 on a geographic-holdout region.
- Per-detection class accuracy ≥ 0.50.
- Downstream tracker MOTA strictly better than the current random-stub baseline on `thresh-eval`'s ADS-B scenario.

#### Scenario: Decision to replace the stub

**WHEN** a developer runs the evaluation harness and the three exit-criterion metrics meet the thresholds above

**THEN** the trained checkpoint is committed at `test-data/models/test_detector.onnx`, the corresponding model card at `test-data/models/MODEL_CARD.md` is updated with the metric values, and the change's `design.md` Open Questions section is annotated with any decisions made during training

**SHALL** include the OpenSky attribution string in the model card.

#### Scenario: Failure to meet exit criteria

**WHEN** a developer runs the evaluation harness and any exit-criterion metric falls below threshold

**THEN** the random-stub model remains in place, the failed metrics are documented in `design.md`, and either (a) the developer iterates on training configuration and re-runs, or (b) Track A is documented as abandoned in `design.md`'s Open Questions section without blocking Track B

**SHALL** not silently land a failing checkpoint.

### Requirement: Sensor frame convention

The detector MUST operate entirely in a local East-North-Up frame centered at the sensor location. Frame conversion to/from ECEF or WGS84 is the responsibility of downstream tracker code.

#### Scenario: Sensor-relative box outputs

**WHEN** the detector is invoked with a point cloud generated by a sensor at `(lat, lon) = (47.45°N, 122.31°W)` (KSEA)

**THEN** the output box positions are in metres east, north, and up from KSEA

**SHALL** match the convention already used by `thresh-synth`'s radar generator.
