# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

Work in progress toward 0.3.0. Develop carries `version = "0.3.0-dev"`. The dated `[0.3.0]` section will be filled in at release-prep time.

JPDA and MHT data association, SafeTensors weight loading, Rust-native DETR detection (no ONNX Runtime required), `thresh-viz` track visualization GUI (v1 + v2 desktop-app milestone), and a comprehensive cognitive complexity refactor.

### Added

#### Track Visualization v2 (Desktop App Milestone)
- **Live streaming integration** in `thresh-viz`: new `streaming::SnapshotBridge` consumes `tokio::sync::broadcast::Receiver<TrackSnapshot>` from `StreamingTracker::subscribe()` on a dedicated tokio runtime, drains into a bounded deque, and surfaces `ConnectionStatus::{Connected, Lagging, Disconnected}` for the dashboard sidebar.
- **Per-frame MOT metrics** via new `thresh_eval::MotMetricsBuilder` (incremental MOTA / MOTP / IDF1 / ID switches in O(K·M) per frame). Re-exported as `thresh_eval::{MotMetrics, MotMetricsBuilder}`.
- **Track lifecycle events** derived from snapshot diffs: `events::diff_snapshots(prev, next) -> Vec<LifecycleEvent>` emits `Born` / `Died` / `IdSwitched` / `Merged` (last reserved). New `LifecycleEvent` enum on `VizFrame.events`.
- **Plot enhancements**: 2σ covariance ellipses (toggleable, hotkey `E`) via `geom::ellipse_axes` + `geom::ellipse_polyline`; association lines (toggleable, hotkey `A`) drawn from each measurement to its assigned track using new `VizFrame.associations`.
- **UI polish**: centralized `KeyBindings`, scrolling lifecycle event log panel (toggleable, hotkey `L`), live connection status indicator, PNG screenshot export with timestamped filenames (hotkey `S`, `--screenshot-dir <PATH>` flag), keyboard shortcuts help overlay (hotkey `?`, also dismissed by `Esc`).
- **CLI flags** on the `thresh-viz` binary: `--recording`, `--stream` (reserved), `--max-buffered-snapshots`, `--screenshot-dir`, `-h`/`--help`.
- **Cross-platform GUI build CI**: new `viz-build` job in `.github/workflows/ci.yml` runs `cargo build -p thresh-viz --features gui` on Ubuntu, macOS, and Windows runners with `fail-fast: false`. Build is the smoke test (no headless GUI run required).
- **Headless integration tests** using `egui_kittest`: 5 tests drive simulated keyboard input against a real `ThreshVizApp` and assert state transitions (help overlay open/close, ellipse / association / event-log toggles).

#### JPDA and MHT Data Association
- `jpda` module in `thresh-association` implementing Joint Probabilistic Data Association with soft assignment probabilities across all track-detection pairs (validation gating, joint event enumeration, marginal association probabilities). Integrates into the tracker via `AssociationStrategy::Jpda`.
- `mht` module implementing Multi-Hypothesis Tracking with a hypothesis tree, deferred hard decisions, N-scan pruning, and clutter-rejection via track-quality scoring. Integrates via `AssociationStrategy::Mht`.
- `MultiObjectTracker` now accepts `AssociationStrategy` (Hungarian | JPDA | MHT) — selectable at construction.
- Crossing-tracks integration test demonstrates JPDA produces better MOTA than Hungarian on tightly-spaced targets; dense-clutter test (`mht_dense_clutter_maintains_track`) verifies MHT robustness with 10 false alarms per frame.

#### SafeTensors Weight Loading
- `WeightLoader` trait + `SafeTensorsLoader` implementation in `thresh-core::weights` and `thresh-inference::weights` mapping named tensors to `nalgebra` matrices with shape validation on load.
- `WeightSet` API for organized access to detector weight collections.
- `OnnxDetector` reload-with-weights stub returns a structured error (full hot-swap deferred to future ONNX Runtime work).
- `scripts/generate_test_weights.py` produces `test-data/models/test_weights.safetensors` for CI and integration tests.

#### Rust-Native DETR Detector
- `NativeDetector` in `thresh-inference::native_detector` — a pure-Rust simplified DETR decoder (6 transformer layers, 256-dim embeddings, 8 attention heads) implementing the `DetectionPipeline` trait. Removes the ONNX Runtime dependency for the default detection path.
- Building blocks: `relu`, `softmax_rows`, `layer_norm`, multi-head attention, FFN, decoder layer, detection head (class + bbox).
- `NativeDetector::from_safetensors()` constructor wiring the SafeTensors loader to architecture weights.
- 11 unit tests covering forward-pass shapes; CPU-only inference latency benchmark.

#### Track Visualization (`thresh-viz`)
- New `crates/thresh-viz` crate with a native `egui` + `eframe` desktop GUI for real-time and recorded track visualization.
- 2D bird's-eye-view plot rendering color-coded track trails, measurement scatter, and association lines at ≥30 FPS.
- Real-time metric sidebar (MOTA, MOTP, track count, confirmed/tentative/lost breakdown).
- JSON recording format (`Recording`, `RecordingFrame`) with `--recording <file.json>` CLI flag for offline playback.
- Sample recording at `crates/thresh-viz/test-data/sample_recording.json` for manual exploration.

### Changed
- All 19 SonarCloud `rust:S3776` cognitive complexity violations resolved via phase-helper decomposition (see CLAUDE.md style guide). Touched: `hungarian.rs`, `adsb.rs::extract_ground_truth`, `orbital.rs` (RK4 stages, `extract_passes`, `trajectory step`), and stereographic tracker tests. `bin_adsb_detections` argument list grouped into an `EnuRef` struct.
- `MultiObjectTracker` integration: `step_detections` now dispatches through the configured `AssociationStrategy`.

### Notes
- `thresh-viz` is a workspace member but is **not** in `default-members` — build it explicitly with `cargo build -p thresh-viz` or `cargo run -p thresh-viz`.

### Migration
- **`VizFrame` gained two new fields**: `associations: Vec<(usize, u64)>` and `events: Vec<LifecycleEvent>`. Both are `#[serde(default)]` so existing JSON recordings load unchanged. Code that constructs `VizFrame` via struct literals must add `associations: Vec::new(), events: Vec::new()` (or use `VizFrame::from_raw` / `from_tracker`).
- **New incremental metrics API**: `thresh_eval::MotMetricsBuilder` is the recommended way to compute MOTA / MOTP / IDF1 in live tracking sessions. The existing one-shot `compute_mot_metrics` and `compute_idf1` continue to work unchanged.
- **`thresh-viz` `gui` feature now also activates `thresh-tracker/streaming`** transitively (needed for the live `SnapshotBridge`). No action needed unless you depended on the old, narrower feature graph.

## [0.2.0] - 2026-04-12

Major feature release: IMM adaptive filtering, streaming tracker, track-to-track fusion, detection pipeline, Python bindings, and performance optimizations.

### Added

#### Interacting Multiple Model (IMM) Filter
- `ImmFilter` in `thresh-filter` combining N motion models (CV, CA, CTRV, CT) with Markov-switching transition probabilities. Full 5-step cycle: interaction → model-conditioned predict → update → mode probability update (log-space likelihoods) → state/covariance combination.
- `StateMapping` trait for cross-model state projection (6D common representation).
- `ImmConfig` with factory methods: `cv_ca()`, `cv_ctrv()`, `cv_ca_ctrv_ct()`.
- `MultiObjectTracker::new_imm_position()` constructor for IMM-based tracking with automatic mode switching.

#### Real-Time Streaming Tracker
- `StreamingTracker` in `thresh-tracker` (feature-gated: `streaming`) wrapping `MultiObjectTracker` with tokio mpsc/broadcast channels.
- `TemporalBinner` for frame accumulation with configurable `frame_duration_s` and `max_latency_s`.
- `StreamingConfig` with `DropPolicy::DropOldest | Block` and latency management (predict-only frames when tracker falls behind).
- `TrackSnapshot` / `TrackState` broadcast output with track positions, velocities, and confirmation status.

#### Track-to-Track Fusion
- `TrackExchange` wire format, `t2t_association()` via augmented-state Mahalanobis distance + Hungarian assignment.
- Three fusion modes: `Naive` (inverse-covariance-weighted), `CovarianceIntersection` (safe for unknown cross-covariances), `OptimalWithCrossCovariance` (Bar-Shalom formula with P12 bookkeeping).
- `FederatedFusionManager` with stateful track management, temporal extrapolation (`extrapolate_track`, `align_to_common_time`), track birth/timeout, and `get_fused_tracks()` common operating picture.

#### Transformer-Based Detection Pipeline
- `Detection3D` type in `thresh-core` with position, dimensions, yaw, class_id, confidence.
- `DetectionPipeline` trait, `SensorInput`, `SensorInputType` enum (PointCloud, ImageTensor).
- 3D non-maximum suppression (`nms_3d`) and confidence filtering (`filter_by_confidence`).
- `OnnxDetector` (feature-gated: `onnx`) with placeholder inference; `MockDetector` for testing.
- `OnnxDetectorConfig` with voxel size and confidence threshold configuration.
- Point cloud voxelization and image normalization preprocessing in `thresh-inference::preprocess`.
- `MultiObjectTracker::step_detections(&[Detection3D], dt)` bridging detection output to tracker input.
- Synthetic ONNX test model (`test-data/models/test_detector.onnx`) for CI validation.

#### Python Bindings (`thresh-py`)
- New `crates/thresh-py` crate with PyO3 `#[pymodule]` exposing:
  - `PyMultiObjectTracker` — step with list-of-lists detections, get confirmed tracks.
  - `PyKalmanFilter` — predict/update with list-based matrix I/O.
  - `compute_mot_metrics_py` — MOTA/MOTP/ID-switches from Python lists.
  - `ThreshError` custom exception, `run_scenario` placeholder.
- `pyproject.toml` for maturin wheel builds.
- Conversion helpers: `lists_to_dvectors`, `dmatrix_to_lists`, `lists_to_dmatrix`.
- pytest suite (`tests/python/`) for tracker, filter, and metrics.
- CI: `python-tests` job (maturin develop + pytest) and cross-platform wheel build matrix.

#### Performance Optimization
- `HungarianSolver` with pre-allocated flat buffers — avoids per-call Vec allocation for repeated assignment problems. `hungarian_assignment()` remains as a convenience wrapper.
- `rayon` parallelism (feature-gated: `parallel`) for `predict_all` (≥32 tracks) and `build_track_cost_matrix` (per-row parallel computation).
- Criterion micro-benchmark suite: Hungarian (10/100/500), KF predict/update, Mahalanobis 6D, tracker step (10/50/200 targets).
- CI benchmark job with artifact upload; profiling documentation in `docs/reference/`.

#### Scenario Variants
- Three new synthetic benchmark scenarios: `synth-maneuvering` (CV/CTRV/CA mode switches), `synth-heterogeneous` (UAV/aircraft/missile mixed dynamics), `synth-low-pd` (P_d=0.7 + clutter).
- `scenario_type` field on `ScenarioParameters` with dispatcher pattern.
- `radar_config_for_scenario()` for scenario-specific detection parameters.

### Changed
- `ScenarioSource::AdsB` and `ScenarioSource::Orbital` extended with additional configuration fields (all backward-compatible via serde defaults).
- `FederatedFusionManager` is now stateful — maintains fused tracks across `fuse()` calls.
- SonarCloud CPD exclusion for IMM trait-impl files (false-positive duplication on `StateMapping` method signatures).

## [0.1.0] - 2026-04-12

Initial public release. Multi-sensor fusion multi-object tracking framework in Rust.

### Added

#### Core Framework
- **thresh-core**: State vectors, measurements (Radar, ADS-B, EO/IR, OTHR), covariance matrices, coordinate transforms (WGS84, ECEF, ENU, ECI, TEME), sensor models, track lifecycle types, and time utilities.
- **thresh-filter**: Kalman filter family (KF, EKF, UKF) with pluggable motion models (Constant Velocity, Constant Acceleration, Coordinated Turn Rate and Velocity, Coordinated Turn).
- **thresh-association**: Hungarian (Kuhn-Munkres) algorithm for optimal linear assignment with bipartite augmenting-path matching, Mahalanobis gating, 2D/3D IoU, and cascaded multi-stage association.
- **thresh-fusion**: Centralized fusion, information filter, and covariance intersection for multi-sensor track fusion.
- **thresh-tracker**: Multi-object tracker with M-of-N track confirmation, class-specific tracking heads, and four long-range tracking frame variants (Cartesian ENU, ECEF, great-circle geodetic, local stereographic projection) plus automatic variant selection.
- **thresh-eval**: MOT evaluation metrics — MOTA, MOTP, IDF1, HOTA, AMOTA — with ground-truth-to-track matching via Hungarian assignment.

#### Sensor Simulation (thresh-synth)
- Synthetic trajectory generation with CV, CA, CTRV, and coordinated-turn segment types.
- Three-level radar measurement fidelity: Level 0 (fixed P_d), Level 1 (radar equation + Swerling I–IV RCS fluctuation + ITU-R P.676 atmospheric loss), Level 2 (RadarSimPy full-wave simulation via PyO3 bridge, feature-gated).
- Swerling RCS models (cases 0–IV) with chi-squared distribution sampling, RCS lookup tables with bilinear interpolation, and preset profiles (fighter, airliner, cruise missile, UAV, satellite).
- Radar equation: monostatic SNR, Albersheim/Shnidman P_d, system noise temperature, atmospheric attenuation, range-dependent measurement noise.
- EO/IR physics: Planck blackbody radiance, MWIR/LWIR band integration, Beer-Lambert atmospheric transmission, IR detection probability, preset sensor configs and target IR signatures.
- OTHR sensor model: Chapman-layer ionospheric propagation, MUF/skip-zone computation, Vincenty coordinate registration, multi-path disambiguation, Doppler-based detection, diurnal coverage variation, preset ROTHR/JORN configs.
- Orbital propagation: RK4 integrator with J2 perturbation and atmospheric drag, Keplerian initialization, impulsive maneuvers, ground-station visibility and slant-range computation.
- RCS computation bridge (`thresh-rcs-compute` CLI) to PyPOFacets via PyO3 (feature-gated: `rcs-compute`).
- Radar scene simulation bridge to RadarSimPy via PyO3 (feature-gated: `radar-scene`).
- JSBSim trajectory bridge via PyO3 (feature-gated: `jsbsim`).

#### Data Pipeline (thresh-data)
- ADS-B ingestion: OpenSky REST client with rate limiting and exponential backoff, SBS BaseStation CSV parser, state-vector-to-measurement conversion, ground-truth trajectory extraction with 1-second grid interpolation, content-hash download caching.
- Orbital ingestion: TLE two-line/three-line parser, GP JSON parser, SGP4 propagation via the `sgp4` crate, TEME→ECEF→ENU coordinate chain, pass prediction, ground-station radar measurement generation.
- Space-Track REST client with session-cookie authentication and CelesTrak public GP fetcher (feature-gated: `orbital`).
- nuScenes ingestion: PyO3 bridge to nuScenes devkit, scene/sample iteration, 3D annotation parsing, instance-level tracking, LiDAR/radar point cloud loading, sensor calibration (feature-gated: `nuscenes`).
- Dataset abstraction: `Dataset` trait, `SyntheticDataset` adapter, `MixedDataset` k-way merge with temporal bucketing.
- Benchmark scenario runner (`thresh-data` CLI): `list` / `run` / `help` subcommands with per-source-type runners (synthetic, ADS-B, orbital, nuScenes), TOML manifest format with regression baselines, deterministic seeded RNG for reproducibility.
- Six scenario manifests: `synth-cv-clean`, `adsb-single-flight`, `adsb-tracon`, `orbital-iss`, `orbital-starlink-train`, `nuscenes-mini`.

#### Optional Bridges (feature-gated, require Python)
- `thresh-bridge`: PyO3 bridge to Stone Soup for JPDA, MHT, and IMM algorithms (feature: `stonesoup`).
- `thresh-inference`: ONNX Runtime pipeline orchestration (feature: `onnx`).

#### CI and Quality
- GitHub Actions CI: Rustfmt, Clippy (warnings-as-errors), workspace tests, doc build, OpenSpec validation, CodeQL, SonarCloud analysis, Codecov.
- Synthetic / orbital / ADS-B benchmark-gate CI jobs with offline cached fixtures.
- Nightly network canary for Space-Track, CelesTrak, and OpenSky HTTP endpoints.
- SonarCloud cognitive-complexity enforcement (≤15 per function via `rust:S3776`).
- Pre-commit hooks: fmt, clippy, cargo check, openspec validate (on commit); cargo test (on push).

#### Documentation
- `CLAUDE.md` with build commands, architecture overview, OpenSpec workflow, gitflow strategy, worktree conventions, and phase-helper decomposition style guide.
- `README.md` with workspace overview, quick-start example, sensor fidelity levels table, and feature-gate documentation.
- Mathematical reference documents in `docs/reference/`.
- 22 validated OpenSpec capability specifications across 8 domains.

[0.2.0]: https://github.com/Chaffgold/thresh/releases/tag/v0.2.0
[0.1.0]: https://github.com/Chaffgold/thresh/releases/tag/v0.1.0
