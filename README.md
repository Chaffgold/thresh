# thresh

[![CI](https://github.com/Chaffgold/thresh/actions/workflows/ci.yml/badge.svg?branch=develop)](https://github.com/Chaffgold/thresh/actions/workflows/ci.yml)
[![Benchmarks](https://github.com/Chaffgold/thresh/actions/workflows/benchmarks.yml/badge.svg?branch=develop)](https://github.com/Chaffgold/thresh/actions/workflows/benchmarks.yml)
[![codecov](https://codecov.io/gh/Chaffgold/thresh/branch/develop/graph/badge.svg)](https://codecov.io/gh/Chaffgold/thresh)
[![Quality Gate Status](https://sonarcloud.io/api/project_badges/measure?project=Chaffgold_thresh&metric=alert_status)](https://sonarcloud.io/summary/new_code?id=Chaffgold_thresh)
[![Coverage](https://sonarcloud.io/api/project_badges/measure?project=Chaffgold_thresh&metric=coverage)](https://sonarcloud.io/summary/new_code?id=Chaffgold_thresh)
[![Lines of Code](https://sonarcloud.io/api/project_badges/measure?project=Chaffgold_thresh&metric=ncloc)](https://sonarcloud.io/summary/new_code?id=Chaffgold_thresh)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)

Multi-sensor fusion multi-object tracking framework in Rust.

Hybrid architecture: transformer-based detection (via ONNX Runtime) + classical Bayesian state estimation (Kalman filter family). Designed for heterogeneous aerospace targets spanning UAVs through ballistic missiles.

## Workspace

| Crate | Description |
|---|---|
| `thresh-core` | Common types: state vectors, measurements, covariance matrices, coordinates |
| `thresh-filter` | KF, EKF, UKF with CV, CA, CTRV, Coordinated Turn motion models |
| `thresh-association` | Hungarian algorithm, Mahalanobis gating, 2D/3D IoU, cascaded association |
| `thresh-fusion` | Centralized fusion, information filter, covariance intersection |
| `thresh-inference` | ONNX Runtime pipeline orchestration (feature-gated) |
| `thresh-tracker` | Track lifecycle, M-of-N confirmation, class-specific heads |
| `thresh-bridge` | PyO3 bridge to Stone Soup (feature-gated) |
| `thresh-synth` | Synthetic trajectory + sensor data generation |
| `thresh-eval` | MOT metrics: MOTA, MOTP, IDF1, HOTA, AMOTA |

## Quick start

```rust
use thresh_tracker::tracker::MultiObjectTracker;
use nalgebra::DVector;

// Create a tracker with 10m measurement noise, 100 chi-squared gate
let mut tracker = MultiObjectTracker::new_cv_position(10.0, 100.0);

// Feed detections each frame
let detections = vec![
    DVector::from_column_slice(&[1000.0, 2000.0, 5000.0]),
];
tracker.step(&detections, 1.0); // dt = 1 second
```

## Building

```sh
cargo build --workspace
cargo test --workspace
```

Optional features:
- `onnx` on `thresh-inference`: enables ONNX Runtime (requires runtime binaries)
- `stonesoup` on `thresh-bridge`: enables PyO3 Stone Soup integration
- `jsbsim`, `rcs-compute`, `radar-scene` on `thresh-synth`: PyO3 bridges to JSBSim, PyPOFacets, and RadarSimPy for high-fidelity sensor simulation (all require a Python environment with the corresponding package installed)
- `adsb`, `orbital`, `nuscenes` on `thresh-data`: dataset adapters (HTTP, SGP4, PyO3)

## Sensor fidelity levels

`thresh-synth` supports three progressive fidelity levels for radar measurement generation. Pick the lowest that still answers the question you're asking; each step up adds realism but also randomness, runtime cost, and/or external dependencies.

| Level | Module / entry point | Detection model | RCS model | When to use |
|---|---|---|---|---|
| **0 — Simple** | `measurement_gen::generate_radar` with `RadarConfig { radar_equation: None, p_detection: 0.85, … }` | Fixed `p_detection` | Not modeled | Algorithm development, deterministic regression tests, tracker integration tests where you want to isolate tracker behaviour from sensor noise. |
| **1 — Radar equation** | `radar_equation::generate_radar_full` with `FullRadarConfig { radar: RadarParameters::x_band_surveillance(), apply_atmosphere: true, use_shnidman: true, … }` + `swerling::RcsProfile::fighter()` (or similar) | `albersheim_pd` / `shnidman_pd` from range-dependent SNR, ITU-R P.676 atmospheric loss, Swerling I/II/III/IV fluctuation | Swerling chi-squared distribution + optional aspect-angle lookup table (`RcsLookupTable`) | Realistic P_d vs range behaviour, tracker robustness testing under realistic drop-outs, benchmarks that need physically meaningful baselines. Pure Rust — runs in default CI. |
| **2 — Full wave/ray simulation** | `radar_scene::RadarSceneBridge` (feature `radar-scene`) → PyO3 to RadarSimPy; `rcs_compute::RcsComputeBridge` (feature `rcs-compute`) → PyO3 to PyPOFacets | High-fidelity waveform / return-pulse simulation via RadarSimPy | Method-of-Moments / Physical Optics RCS computed from an STL geometry via PyPOFacets | Sensor design validation, proof that a specific radar / target / environment combination meets a requirement, data generation for transformer-based detectors. Requires a Python environment and is excluded from default CI builds because of the heavy external dependency. |

The `thresh-rcs-compute` CLI binary (`cargo install --path crates/thresh-synth --features rcs-compute`) wraps Level 2 RCS computation so you can precompute an `RcsLookupTable` JSON from an STL file and then feed it into Level 1 without paying the PyO3 cost on every measurement.

End-to-end integration tests for each scenario live in `crates/thresh/tests/hifi_integration.rs`:

- `hifi_radar_swerling_scenario` — fighter trajectory, Level 1 radar with Swerling I fluctuation.
- `hifi_orbital_radar_scenario` — ISS-like overhead pass against a ground station radar, range-dependent `P_d`.
- `hifi_multisensor_radar_eoir` — radar + MWIR EO/IR physics fusion on a maneuvering aircraft.
- `fidelity_level_comparison` — same trajectory run at Level 0 and Level 1, logs the MOTA delta.

Run them with `cargo test -p thresh --test hifi_integration`.

## License

Apache-2.0
