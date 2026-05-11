# Tasks — Flight Data Training Pipeline

> Phases are ordered by dependency, not strict serial execution. Phase 2 can start before Phase 1 is fully done. Phases 3 and 4 are largely independent and may proceed in parallel.

## 1. Python toolchain and project layout

- [ ] 1.1 Add `pyproject.toml` at repo root (or `python/`) declaring the training-side dependencies: `torch`, `onnx`, `onnxruntime`, `numpy`, `pandas`, `pyarrow`, `requests`, `httpx`, `pydantic`.
- [ ] 1.2 Generate `uv.lock` and document the bootstrap command (`uv sync`) in `TRAINING.md`.
- [ ] 1.3 Create the `python/` source tree: `python/acquisition/`, `python/training/`, `python/export/`, `python/eval/`, with `__init__.py` stubs.
- [ ] 1.4 Add a pre-commit hook running `ruff` and `pyright` on the new Python tree.
- [ ] 1.5 Wire `python/` into the existing CI workflow so its tests run alongside the Rust ones.

## 2. Acquisition layer — OpenSky

- [ ] 2.1 Implement `python/acquisition/opensky.py` with `fetch_state_vectors(bbox, time_range)` over the Impala REST endpoint.
- [ ] 2.2 Implement Zenodo trajectory-dump loader (`load_zenodo_dump(path)`).
- [ ] 2.3 Define the canonical trajectory schema in `python/acquisition/schema.py` using Pydantic + PyArrow: `icao24`, `timestamp`, `lat`, `lon`, `alt_geom`, `alt_baro`, `vel_ground`, `track`, `vrate`, `category`, `callsign`, `quality_nic`, `quality_nac_p`, `source` (one of `opensky` / `adsbx`).
- [ ] 2.4 Implement track stitching: `stitch_tracks(state_vectors)` returns one row per `(icao24, contact)` with the full state-vector sequence as a nested list, splitting on gaps > 60 s.
- [ ] 2.5 Implement Parquet writer with one file per day, partitioned by source and airport region.
- [ ] 2.6 Add a unit test that round-trips a tiny synthetic state-vector stream through the schema and stitching, asserting trajectories are non-empty and timestamps are monotonic.
- [ ] 2.7 Check in a small (< 5 MB) OpenSky-derived sample under `test-data/trajectories/opensky-sample.parquet` for CI dry-runs.

## 3. Acquisition layer — ADS-B Exchange v2

- [ ] 3.1 Implement `python/acquisition/adsbx.py` with `fetch_airport(icao, api_key)` hitting `/api/aircraft/v2/airport/{icao}`.
- [ ] 3.2 Translate the readsb-style ADSBx response into the canonical trajectory schema; map `hex` → `icao24`, `flight` → `callsign`, `alt_baro` / `alt_geom` directly, `gs` → `vel_ground`, `track` directly, `baro_rate` → `vrate`, `category` directly.
- [ ] 3.3 Implement a polling scheduler `python/acquisition/adsbx_poller.py` with a rate-limit budget (default: 1 req/sec, configurable).
- [ ] 3.4 Add per-snapshot append-to-parquet logic with deduplication on `(icao24, timestamp)`.
- [ ] 3.5 Unit test the schema translation against a recorded sample ADSBx response fixture.
- [ ] 3.6 Document the API-key bootstrap in `TRAINING.md` and `LICENSING.md`.

## 4. Track B first experiment — IMM mode classifier

- [ ] 4.1 Implement `python/training/imm_mode_labels.py`: from a canonical trajectory, label each state with one of `{CV, CA, CTRV, coord_turn}` using kinematic thresholds (acceleration > 1 m/s² for CA; turn-rate > 3°/s for CTRV; sustained turn-rate for coord_turn; else CV).
- [ ] 4.2 Implement `python/training/imm_dataset.py` returning sliding windows of fixed length (default 10 steps at 1 Hz) of `(state_history, mode_label_at_end)`.
- [ ] 4.3 Implement a small sequence model in `python/training/imm_model.py`: a 2-layer GRU + linear head outputting 4 logits.
- [ ] 4.4 Implement `python/training/train_imm.py` with a fixed-seed train/test split and standard cross-entropy loss.
- [ ] 4.5 Implement `python/export/export_imm.py`: load best checkpoint, `torch.onnx.export` with input `(batch, 10, state_dim)` and output `(batch, 4)`.
- [ ] 4.6 Verify the export with `onnxruntime` running on a fixture batch; assert outputs are valid probabilities (softmax-applied).
- [ ] 4.7 Add a `learned-imm` Cargo feature in `crates/thresh-filter/Cargo.toml`; behind this feature, expose an `ImmModeAdapter` that loads an ONNX checkpoint via `thresh-inference` and predicts mode probabilities given a state history.
- [ ] 4.8 Add an integration test in `crates/thresh-filter/tests/` that runs the existing IMM test suite with `learned-imm` enabled, using the trained checkpoint.
- [ ] 4.9 Track B exit criterion: on a held-out trajectory split, classifier accuracy ≥ 0.70 against analytic labels AND existing IMM tests still pass.

## 5. Trajectory-driven thresh-synth pairing (Track A foundation)

- [ ] 5.1 Add a new `thresh_synth::radar::from_trajectory(trajectory, sensor_pose, frame=ENU)` API that consumes a canonical trajectory (read via the Rust-side Parquet reader or passed in directly as a slice of states) and emits paired `(point_cloud, gt_boxes_3D)` snapshots at the configured sensor sample rate.
- [ ] 5.2 Reuse existing RCS, clutter, and detection-probability models; the only new piece is the trajectory ingestion.
- [ ] 5.3 Snapshot output should match the ONNX input shape exactly: 1000 points × `[x, y, z, intensity]` plus a variable number of ground-truth boxes (padded to 100, with a validity mask) per `[x, y, z, L, W, H, yaw]`.
- [ ] 5.4 Unit test: feed in a straight-line constant-velocity trajectory, assert the synthesised PCL has a cluster of returns near the predicted target position and the GT box matches the trajectory state.
- [ ] 5.5 Add a Python binding (via `pyo3` or just CSV/parquet I/O) so the training script can call this from PyTorch.
- [ ] 5.6 Spike check: visualise a generated batch and verify the PCL "looks reasonable" before committing to training in Phase 6.

## 6. Track A detector training

- [ ] 6.1 Pick the pretrained backbone (RT-DETR / 3DETR / Group-Free-3D); document the decision in `design.md` under Open Questions.
- [ ] 6.2 Implement `python/training/detector_dataset.py` that materialises `(point_cloud, gt_boxes_3D, gt_classes)` pairs by calling the synth pairing from Phase 5.
- [ ] 6.3 Implement `python/training/train_detector.py` with set-prediction loss (Hungarian matching + L1 + GIoU-3D + cross-entropy on class).
- [ ] 6.4 Define class taxonomy in `python/training/classes.py` per Decision 7 in design.md.
- [ ] 6.5 Run a first training pass on a single airport region (KSEA suggested) for 24–48 GPU-hours.
- [ ] 6.6 Implement `python/export/export_detector.py`: `torch.onnx.export` matching the contract `(1,1000,4) → (1,100,7) + (1,100,1) + (1,100,1)`.
- [ ] 6.7 Add an updated `scripts/generate_test_model.py` that emits a stub ONNX with the new three-output contract (still random weights) so the CI shape contract passes before the real model lands.
- [ ] 6.8 Update Rust-side ONNX parser in `crates/thresh-inference` to read the new `classes` output tensor (default to 0 if absent for backward compatibility).
- [ ] 6.9 Track A exit criterion: mAP@0.5 ≥ 0.30 on the holdout region AND downstream MOTA improvement on `thresh-eval`'s ADS-B scenario.

## 7. ONNX export and verification

- [ ] 7.1 Update `onnx-tests` workflow to assert the new three-output contract (boxes, scores, classes) for `test_detector.onnx`.
- [ ] 7.2 Add a similar contract test for `imm_mode_classifier.onnx`: input `(batch, 10, state_dim)`, output `(batch, 4)`.
- [ ] 7.3 Add a `python/eval/onnx_parity.py` smoke test that runs both Python (`onnxruntime`) and Rust (via `thresh-inference`) on the same fixture batch and asserts outputs match within 1e-5.
- [ ] 7.4 Replace `test-data/models/test_detector.onnx` with the trained Track A checkpoint (only when exit criterion 6.9 is met).
- [ ] 7.5 Drop `test-data/models/imm_mode_classifier.onnx` (the trained Track B checkpoint) into the repository.

## 8. Evaluation harness

- [ ] 8.1 Add `python/eval/holdout_split.py` that splits trajectories by geographic region into train and held-out sets.
- [ ] 8.2 Add `python/eval/run_tracker.py` that drives the full thresh tracker (via `thresh-py`) on a held-out trajectory set and reports MOTA / MOTP / IDF1 from `thresh-eval`.
- [ ] 8.3 Add `python/eval/run_tracker.py --learned-imm` and `--learned-detector` flags so we can A/B test the classical and learned pipelines.
- [ ] 8.4 Generate an evaluation report committed to `docs/eval/flight-data-training-pipeline.md` summarising both tracks' exit criteria against the holdout numbers.

## 9. Documentation and reproducibility

- [ ] 9.1 Write `TRAINING.md` at repo root: end-to-end reproduction recipe from `uv sync` through `python/eval/run_tracker.py`.
- [ ] 9.2 Write `LICENSING.md` documenting OpenSky and ADSBx attribution / redistribution posture.
- [ ] 9.3 Write `test-data/models/MODEL_CARD.md` documenting both trained checkpoints' provenance, training data, and exit-criteria results.
- [ ] 9.4 Update `CLAUDE.md` to mention the new Python tree under `python/` and the `TRAINING.md` entry point.
- [ ] 9.5 Update each affected crate's README where the learned feature gates land.

## 10. Wrap-up

- [ ] 10.1 Final integration test: full `cargo test --workspace --features thresh-filter/learned-imm` passes.
- [ ] 10.2 Final lint pass: `ruff check`, `pyright`, `cargo clippy --workspace -- -D warnings`, `cargo fmt --all -- --check`.
- [ ] 10.3 Final OpenSpec validation: `openspec validate --all --strict --no-interactive` (or equivalent).
- [ ] 10.4 Update this change's `proposal.md` and `design.md` with any decisions made during implementation that diverged from the original plan.
- [ ] 10.5 Open the PR against `develop` once both exit criteria are met (or document any abandoned track in `design.md`'s Open Questions section).
