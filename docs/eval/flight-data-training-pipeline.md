# Evaluation — flight-data-training-pipeline

MOT evaluation of the thresh tracker on synthetic radar scenarios, for the
`flight-data-training-pipeline` change (Phase 9, task 9.4).

## Harness

The evaluation engine is **Rust-native** (`thresh::eval_harness`, design
Decision 24): each scenario's target trajectories are turned into synthetic
radar measurements (`thresh-synth`), run through the analytic 4-model IMM
tracker (`thresh-tracker`, `cv_ca_ctrv_ct`), and the confirmed tracks are scored
against ground truth with `thresh-eval` (MOTA / MOTP / IDF1). Ground truth is
built independently from the trajectory waypoints (stable per-target IDs,
sensor-frame, range-gated) — a missed detection is a true false negative.

Reproduce:

```sh
cargo run -p thresh --bin eval-tracker            # human-readable
cargo run -p thresh --bin eval-tracker -- --json  # machine-readable
# or via the Python wrapper:
uv run -C python python -m eval.run_tracker
```

Settings: 10 Hz, seed 42, matching gate 50 m, `TrajectoryRadarConfig::default()`
radar (≈10 m range / 1 mrad angular noise, P_d ≈ 0.9).

## Analytic baseline (current)

| Scenario | MOTA | MOTP (m) | IDF1 | ID switches | frames |
|---|---|---|---|---|---|
| single-cruise (1 CV target) | 0.904 | 20.6 | 0.949 | 0 | 301 |
| two-separated (2 CV targets) | 0.894 | 20.7 | 0.944 | 0 | 301 |
| maneuvering (CV → coordinated turn) | 0.890 | 36.7 | 0.942 | 0 | 301 |

These are the **analytic-IMM baseline** the learned components must beat (or at
least not regress). MOTA ≈ 0.9 — the ~10% gap is dominated by the pre-confirmation
warm-up (the first few frames have ground truth but no confirmed track) plus
occasional missed detections; **0 ID switches** and IDF1 ≈ 0.94 indicate stable
identity. MOTP rises during the turn (≈37 m vs ≈21 m), as expected for an IMM
tracking a maneuver.

## Exit criteria — status

| Track | Criterion | Status |
|---|---|---|
| **B** (IMM mode classifier) | held-out accuracy ≥ 0.70; passing IMM tests; MOTA no worse than analytic (task 6.8) | **Unmet** — historical maneuvering runs regressed; runtime integration is available |
| **A** (detector) | micro distance-AP strictly above the classical same-point-cloud baseline; matched class accuracy ≥ 0.50; MOTA strictly above the random detector stub (task 7.9, Decision 26) | **Unmet** — historical learned distance-AP was below classical; detector-to-tracker evaluation is now available |

Both tracks additionally require documented training and model-distribution
rights. These synthetic implementation checks do not satisfy representative
trained-model acceptance, and the checked-in models remain random-weight stubs.

## Learned-mode evaluation (2026-09-22)

The IMM path accepts `--learned-imm --imm-model PATH` (`--model` remains an
IMM-only alias). The detector path accepts `--learned-detector --detector-model
PATH`; it runs `from_trajectory` point clouds through `OnnxDetector::detect`
and feeds its output into `MultiObjectTracker::step_detections`. The two modes
can be combined with separate model paths. Missing features, missing model
arguments, and nonexistent/unloadable models fail instead of reporting an
analytic fallback, including in JSON mode.

`--duration-seconds` accepts a positive finite duration (default 30 seconds)
and preserves the maneuvering scenario's 1:2 cruise/turn phase ratio. CI uses
1.5-second sequences as bounded smoke checks only, not acceptance evidence.

From the repository root:

```sh
# Learned IMM using the committed random stub (integration smoke only):
cargo run -p thresh --features learned-imm --bin eval-tracker -- \
  --json --learned-imm --imm-model test-data/models/imm_mode_classifier.onnx

# Candidate versus current random detector, sharing exact point clouds:
cargo run -p thresh --features onnx --bin eval-tracker -- \
  --json --learned-detector --detector-model /path/to/candidate.onnx \
  --detector-baseline-model test-data/models/test_detector.onnx

# Python selects Cargo features automatically; paths are caller-relative:
cd python
uv run python -m eval.run_tracker \
  --learned-detector --detector-model ../test-data/models/test_detector.onnx \
  --learned-imm --imm-model ../test-data/models/imm_mode_classifier.onnx
```

`--detector-baseline-model` materializes one immutable
`DetectorEvalSequence` per scenario and reuses it for both detector runs.
Each run starts a fresh tracker. The generic `run_detector_eval` API accepts
any `DetectionPipeline`, allowing alternative detectors to use the same
observations without exposing truth to inference. Truth comes from the
original trajectories, not snapshot GT-box slots; sensor misses remain false
negatives and target IDs remain stable as targets enter/leave.

JSON includes `pipeline`, distinguishing `radar-measurements/analytic-imm`,
`radar-measurements/learned-imm`, `detector-candidate/analytic-imm`, and
`detector-baseline/analytic-imm` (or `/learned-imm` in combined mode).
The radar-measurement baseline above is **not** a same-input detector baseline:
radar plots and point clouds are different observations. Compare detector
models using the shared sequence, not against those historic radar numbers.
The Python distance-AP bracket continues to handle the classical DBSCAN
comparison; this tracker harness does not assert that distance-AP gate passed.

Regression coverage includes valid-position versus empty detectors (MOTA
above 0.95 versus 0), detector invocation on every frame, byte-equivalent
materialized observations, independent truth despite sensor misses,
sensor-offset/arrival identity, explicit CLI rejection paths, and exact Python
model/feature forwarding. The ONNX CI job also executes the real committed
detector stub and same-model candidate/baseline parity. Combined learned-mode
smoke uses `eval_single_detection.onnx`: a wholly synthetic constant-output
fixture generated by `scripts/generate_eval_detector_fixture.py`, avoiding
dozens of random tracks and their per-track classifier sessions. This fixture
contains no training or external data and must never be an accuracy baseline.
