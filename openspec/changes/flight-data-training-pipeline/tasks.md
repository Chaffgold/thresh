# Tasks — Flight Data Training Pipeline

> Phases are ordered by dependency, not strict serial execution. Phase 2 can start before Phase 1 is fully done. Phases 3 and 4 are largely independent and may proceed in parallel.

## 1. Python toolchain and project layout

- [x] 1.1 Add `pyproject.toml` at repo root (or `python/`) declaring the training-side dependencies: `torch`, `onnx`, `onnxruntime`, `numpy`, `pandas`, `pyarrow`, `requests`, `httpx`, `pydantic`. _Landed at `python/pyproject.toml`; heavy deps (`torch`, `onnx`, `onnxruntime`) split into the `training` optional-extra to keep CI smoke fast._
- [x] 1.2 Generate `uv.lock` and document the bootstrap command (`uv sync`) in `TRAINING.md`.
- [x] 1.3 Create the `python/` source tree: `python/acquisition/`, `python/training/`, `python/export/`, `python/eval/`, with `__init__.py` stubs.
- [x] 1.4 Add a pre-commit hook running `ruff` and `pyright` on the new Python tree.
- [x] 1.5 Wire `python/` into the existing CI workflow so its tests run alongside the Rust ones. _New `python-training-tree` job in `.github/workflows/ci.yml` runs ruff + pyright + pytest via `uv`._

## 2. Acquisition layer — OpenSky

- [x] 2.1 Implement `python/acquisition/opensky.py` with `fetch_state_vectors(bbox, time_range)` over the Impala REST endpoint. _Uses the public `/api/states/all` endpoint with optional credentials; polls across the time range with exponential-backoff retry. Exercised against `httpx.MockTransport` (no live calls in unit tests)._
- [x] 2.2 Implement Zenodo trajectory-dump loader (`load_zenodo_dump(path)`). _Loads Parquet files written in the canonical schema; validates SHA-256 when a checksum is supplied._
- [x] 2.3 Define the canonical trajectory schema in `python/acquisition/schema.py` using Pydantic + PyArrow. _Adds an optional `provenance` map for ADSBx MLAT/TIS-B flags; a parity test asserts Pydantic fields and PyArrow columns match._
- [x] 2.4 Implement track stitching. _`stitch_tracks` yields `Track` records with stable IDs of the form `{icao24}-{first_timestamp_iso}`._
- [x] 2.5 Implement Parquet writer with one file per day, partitioned by source and airport region. _`storage.write_partition` writes at `<root>/source=<src>/date=YYYY-MM-DD/trajectories.parquet` and enforces per-icao monotonicity. Airport-region sub-partitioning is deferred to Phase 3 (ADSBx poller introduces per-airport directories)._
- [x] 2.6 Add a unit test that round-trips a tiny synthetic state-vector stream through the schema and stitching. _46 unit tests across schema / stitching / storage / opensky; full round-trip in `test_storage.TestRoundTrip`._
- [ ] 2.7 Check in a small (< 5 MB) OpenSky-derived sample under `test-data/trajectories/opensky-sample.parquet` for CI dry-runs. _Deferred: requires a live OpenSky fetch, which is not available in the implementation sandbox. To be filled in by a contributor with network access — `uv run python -m acquisition.opensky_cli --bbox … --time …` (small CLI to be added)._

## 3. Acquisition layer — ADS-B Exchange v2

- [ ] 3.1 Implement `python/acquisition/adsbx.py` with `fetch_airport(icao, api_key)` hitting `/api/aircraft/v2/airport/{icao}`.
- [ ] 3.2 Translate the readsb-style ADSBx response into the canonical trajectory schema; map `hex` → `icao24`, `flight` → `callsign`, `alt_baro` / `alt_geom` directly, `gs` → `vel_ground`, `track` directly, `baro_rate` → `vrate`, `category` directly.
- [ ] 3.3 Implement a polling scheduler `python/acquisition/adsbx_poller.py` with a rate-limit budget (default: 1 req/sec, configurable).
- [ ] 3.4 Add per-snapshot append-to-parquet logic with deduplication on `(icao24, timestamp)`.
- [ ] 3.5 Unit test the schema translation against a recorded sample ADSBx response fixture.
- [ ] 3.6 Document the API-key bootstrap in `TRAINING.md` and `LICENSING.md`.

## 4. Trajectory-driven `thresh-synth` pairing (shared foundation for Tracks A and B)

> Both tracks depend on running `thresh-synth` over a real trajectory: Track A consumes paired point-cloud / box snapshots; Track B consumes the synthesised measurement stream as input to the classical tracker. This phase delivers the shared API once.

- [x] 4.1 Add a new `thresh_synth::radar::from_trajectory(...)` API. _Landed in `crates/thresh-synth/src/radar_trajectory.rs` as `from_trajectory(&[TargetTrack], &TrajectoryRadarConfig, &mut Rng) -> Vec<RadarSnapshot>`. Takes a slice of waypoints in memory (no Parquet read on the Rust side per the Phase 4 design memo); Python-side conversion is the canonical entry point._
- [x] 4.2 Add a parallel `thresh_synth::radar::measurements_from_trajectory(...)` API. _Implemented; wraps `measurement_gen::generate_radar_with_rcs` per tick, returns `Vec<Vec<Measurement>>`._
- [x] 4.3 Reuse existing RCS, clutter, and detection-probability models. _Per-class RCS table (`CLASS_RCS_M2`) is passed through to `measurement_gen::detection_probability`; clutter is sampled uniformly inside a configurable cube around the sensor; the radar equation is reused via the underlying `RadarConfig`._
- [x] 4.4 Snapshot output matches the ONNX input shape exactly. _`RadarSnapshot` carries a flat `Vec<f32>` of length `POINT_CLOUD_SIZE * POINT_DIM` (1000 × 4) plus boxes of length `MAX_GT_BOXES * BOX_DIM` (100 × 7) and a parallel `gt_valid` mask + `gt_classes` index._
- [x] 4.5 Unit tests: CV trajectory → cluster + box correctness, measurement-stream noise sanity, out-of-range drop, validation paths. _10 new lib tests in `radar_trajectory::tests` (138 total `cargo test -p thresh-synth --lib`); covers (a) interpolation linearity and clamping, (b) snapshot count = `floor(T/dt)+1`, (c) PCL cluster near GT, (d) shape contract, (e) out-of-range target dropped, (f) measurement-stream per-tick rate, plus three error-path checks._
- [ ] 4.6 Add a Python binding (via `pyo3` or just CSV/parquet I/O) so the training scripts can call both APIs from PyTorch. _Deferred to a follow-up: the existing `jsbsim.rs` / `radar_scene.rs` PyO3 bridges are the model; this would add another `pyo3`-gated feature flag. Out of scope for the initial Phase 4 spike._
- [ ] 4.7 Spike check: visualise a generated PCL batch and a measurement-stream trace, verify both "look reasonable" before committing to training in Phases 5 and 7. _Deferred to a follow-up; the unit tests in 4.5 cover the "looks reasonable" criterion algorithmically (cluster proximity to GT, snapshot rate, missed-detection behaviour). A separate notebook-style visualisation would be useful before Track A training but not required to land 4.1–4.5._

## 5. Track B foundation — measurement synthesis + classical-tracker loop

> Per design.md Decision 4: ADS-B is system-level truth, not a measurement source. The IMM classifier must train on **filter outputs** (state estimates from the classical tracker) so its training distribution matches deployment. This phase wires truth → synth-measurements → classical tracker → filter-state-history into a training-ready dataset.

- [ ] 5.1 Drive `thresh_synth::radar::measurements_from_trajectory` (Phase 4.2) over each canonical trajectory in the acquisition output, producing a stream of synthesised measurements per trajectory.
- [ ] 5.2 Run the classical tracker (`thresh-tracker` configured with the analytic IMM) over the synthesised measurement stream. Snapshot the per-step filter state and covariance.
- [ ] 5.3 Define a `filter_state_projection` that maps the full Kalman/IMM state and covariance into a fixed-size feature vector suitable as classifier input. Default: state `[x, y, z, vx, vy, vz, ax, ay, az]` concatenated with the diagonal of the covariance (dim = 18). Document the choice in `design.md` Open Questions and revisit if the classifier struggles.
- [ ] 5.4 Derive analytic mode labels per snapshot from the underlying trajectory kinematics (acceleration > 1 m/s² → CA; turn rate > 3°/s → CTRV; sustained turn rate > 3°/s with bank > 15° → coord_turn; else CV). Labels MUST come from the trajectory, not from the filter output.
- [ ] 5.5 Write the resulting `(filter_state_history, analytic_mode_label_at_end)` examples to Parquet under `test-data/training/imm-classifier/` (sample only) and an external bucket (full).
- [ ] 5.6 Unit test: a constant-velocity trajectory yields labels that are 100% `CV`; a known coordinated-turn trajectory yields labels that are at least 80% `coord_turn` during the turn segment.
- [ ] 5.7 Smoke test: assert the filter-state input to the classifier dataset is NEVER drawn directly from an ADS-B state vector (only ever from a tracker filter output). This guard fails the build if the dataset pipeline is wired incorrectly.

## 6. Track B first experiment — IMM mode classifier training

- [ ] 6.1 Implement `python/training/imm_dataset.py` returning sliding windows of fixed length (default 10 steps) of `(filter_state_history, mode_label_at_end)` from the Parquet produced in Phase 5.5.
- [ ] 6.2 Implement a small sequence model in `python/training/imm_model.py`: a 2-layer GRU + linear head outputting 4 logits.
- [ ] 6.3 Implement `python/training/train_imm.py` with a fixed-seed train/test split (by region) and standard cross-entropy loss.
- [ ] 6.4 Implement `python/export/export_imm.py`: load best checkpoint, `torch.onnx.export` with input `(batch, 10, filter_state_dim)` and output `(batch, 4)`.
- [ ] 6.5 Verify the export with `onnxruntime` running on a fixture batch; assert outputs are valid probabilities (softmax-applied).
- [ ] 6.6 Add a `learned-imm` Cargo feature in `crates/thresh-filter/Cargo.toml`; behind this feature, expose an `ImmModeAdapter` that loads an ONNX checkpoint via `thresh-inference` and predicts mode probabilities given a filter-state history. The adapter MUST accept filter state (state + covariance projection), not raw measurements.
- [ ] 6.7 Add an integration test in `crates/thresh-filter/tests/` that runs the existing IMM test suite with `learned-imm` enabled, using the trained checkpoint.
- [ ] 6.8 Track B exit criterion: on a held-out trajectory split, classifier accuracy ≥ 0.70 against analytic labels AND existing IMM tests still pass AND downstream tracker MOTA on the synthetic ADS-B scenario is no worse than with analytic mode probabilities.

## 7. Track A detector training

- [ ] 7.1 Pick the pretrained backbone (RT-DETR / 3DETR / Group-Free-3D); document the decision in `design.md` under Open Questions.
- [ ] 7.2 Implement `python/training/detector_dataset.py` that materialises `(point_cloud, gt_boxes_3D, gt_classes)` pairs by calling the synth pairing from Phase 4.
- [ ] 7.3 Implement `python/training/train_detector.py` with set-prediction loss (Hungarian matching + L1 + GIoU-3D + cross-entropy on class).
- [ ] 7.4 Define class taxonomy in `python/training/classes.py` per Decision 8 in design.md.
- [ ] 7.5 Run a first training pass on a single airport region (KSEA suggested) for 24–48 GPU-hours.
- [ ] 7.6 Implement `python/export/export_detector.py`: `torch.onnx.export` matching the contract `(1,1000,4) → (1,100,7) + (1,100,1) + (1,100,1)`.
- [ ] 7.7 Add an updated `scripts/generate_test_model.py` that emits a stub ONNX with the new three-output contract (still random weights) so the CI shape contract passes before the real model lands.
- [ ] 7.8 Update Rust-side ONNX parser in `crates/thresh-inference` to read the new `classes` output tensor (default to 0 if absent for backward compatibility).
- [ ] 7.9 Track A exit criterion: mAP@0.5 ≥ 0.30 on the holdout region AND downstream MOTA improvement on `thresh-eval`'s ADS-B scenario.

## 8. ONNX export and verification

- [ ] 8.1 Update `onnx-tests` workflow to assert the new three-output contract (boxes, scores, classes) for `test_detector.onnx`.
- [ ] 8.2 Add a similar contract test for `imm_mode_classifier.onnx`: input `(batch, 10, filter_state_dim)`, output `(batch, 4)`.
- [ ] 8.3 Add a `python/eval/onnx_parity.py` smoke test that runs both Python (`onnxruntime`) and Rust (via `thresh-inference`) on the same fixture batch and asserts outputs match within 1e-5.
- [ ] 8.4 Replace `test-data/models/test_detector.onnx` with the trained Track A checkpoint (only when exit criterion 7.9 is met).
- [ ] 8.5 Drop `test-data/models/imm_mode_classifier.onnx` (the trained Track B checkpoint) into the repository.

## 9. Evaluation harness

- [ ] 9.1 Add `python/eval/holdout_split.py` that splits trajectories by geographic region into train and held-out sets.
- [ ] 9.2 Add `python/eval/run_tracker.py` that drives the full thresh tracker (via `thresh-py`) on a held-out trajectory set and reports MOTA / MOTP / IDF1 from `thresh-eval`.
- [ ] 9.3 Add `python/eval/run_tracker.py --learned-imm` and `--learned-detector` flags so we can A/B test the classical and learned pipelines.
- [ ] 9.4 Generate an evaluation report committed to `docs/eval/flight-data-training-pipeline.md` summarising both tracks' exit criteria against the holdout numbers.

## 10. Documentation and reproducibility

- [ ] 10.1 Write `TRAINING.md` at repo root: end-to-end reproduction recipe from `uv sync` through `python/eval/run_tracker.py`.
- [ ] 10.2 Write `LICENSING.md` documenting OpenSky and ADSBx attribution / redistribution posture.
- [ ] 10.3 Write `test-data/models/MODEL_CARD.md` documenting both trained checkpoints' provenance, training data, and exit-criteria results. Include the explicit note that ADS-B is consumed as system-level truth, never as measurements.
- [ ] 10.4 Update `CLAUDE.md` to mention the new Python tree under `python/` and the `TRAINING.md` entry point.
- [ ] 10.5 Update each affected crate's README where the learned feature gates land.

## 11. Wrap-up

- [ ] 11.1 Final integration test: full `cargo test --workspace --features thresh-filter/learned-imm` passes.
- [ ] 11.2 Final lint pass: `ruff check`, `pyright`, `cargo clippy --workspace -- -D warnings`, `cargo fmt --all -- --check`.
- [ ] 11.3 Final OpenSpec validation: `openspec validate --all --strict --no-interactive` (or equivalent).
- [ ] 11.4 Update this change's `proposal.md` and `design.md` with any decisions made during implementation that diverged from the original plan.
- [ ] 11.5 Open the PR against `develop` once both exit criteria are met (or document any abandoned track in `design.md`'s Open Questions section).
