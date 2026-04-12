# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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

[0.1.0]: https://github.com/Chaffgold/thresh/releases/tag/v0.1.0
