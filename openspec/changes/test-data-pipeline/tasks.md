## 1. Crate Setup

- [x] 1.1 Create `thresh-data` crate with feature flags: `adsb`, `orbital`, `nuscenes`
- [x] 1.2 Workspace dependencies wired: `reqwest` (optional under `adsb`), `csv` (optional under `adsb`), `sgp4` (optional under `orbital`), `toml` (for credentials and scenario manifests)
- [x] 1.3 Define `Dataset` trait: `metadata()`, `frames()`, `ground_truth()`
- [x] 1.4 Define `Frame` struct: timestamp, measurements, optional ground truth, sensor metadata
- [x] 1.5 Define `DatasetMetadata`: name, source, target count, time span, coordinate frame
- [x] 1.6 Implement credential loader: env vars → `~/.thresh/credentials.toml` fallback
- [x] 1.7 Implement cache directory management: `~/.thresh/data/<source>/<dataset>/`

## 2. Coordinate Transforms

- [x] 2.1 Implement WGS84 geodetic (lat, lon, alt) → ECEF (x, y, z)
- [x] 2.2 Implement ECEF → ENU relative to configurable reference point
- [x] 2.3 Implement WGS84 → ENU convenience function (compose 2.1 + 2.2)
- [x] 2.4 Implement TEME → ECEF (for SGP4 output, includes Earth rotation via GMST)
- [x] 2.5 Implement ECI (J2000/GCRF) coordinate frame type and conversions
- [x] 2.6 Implement ECI → ECEF transform (Earth rotation angle via GMST/ERA)
- [x] 2.7 Implement ECEF → ECI inverse transform
- [x] 2.8 Implement ECI → ENU convenience function (compose ECI→ECEF + ECEF→ENU)
- [x] 2.9 Write tests: known reference points (e.g., JFK airport) match published ECEF coordinates
- [x] 2.10 Write tests: ENU roundtrip (WGS84 → ENU → WGS84) within 1 cm
- [x] 2.11 Write tests: ECI↔ECEF roundtrip at known epoch matches reference values
- [x] 2.12 Write tests: ECI→ENU for ISS pass matches published ground station observations

## 3. ADS-B Ingestion

- [x] 3.1 `OpenSkyClient::fetch_states` in `crates/thresh-data/src/adsb.rs` fetches state vectors with optional `BoundingBox` and time parameters, cached under `~/.thresh/data/opensky/states/`.
- [x] 3.2 `OpenSkyClient::fetch_track` fetches per-ICAO24 flight tracks against `/api/tracks/all`, cached under `~/.thresh/data/opensky/tracks/`.
- [x] 3.3 `parse_states_response` / `parse_track_response` parse OpenSky JSON into `StateVector` / `TrackPoint` structs.
- [x] 3.4 `parse_sbs_line` implements the SBS BaseStation CSV parser (MSG type discrimination + field extraction).
- [x] 3.5 `state_vector_to_measurement` converts to `Measurement::AdsB` via WGS84→ENU.
- [x] 3.6 `extract_ground_truth` groups state vectors by ICAO24 and interpolates to a 1-second grid (refactored in PR #33 for cognitive complexity).
- [x] 3.7 `AdsBDataset` implements the `Dataset` trait.
- [x] 3.8 Added `content_hash_key(namespace, parts)` helper in `crates/thresh-data/src/cache.rs` — derives a deterministic 16-char hex cache key from a namespace + arbitrary request-identifying parts via `DefaultHasher`, with 6 unit tests covering determinism, part / namespace distinction, hex length, order sensitivity, and the zero-part path. `OpenSkyClient::fetch_states` and `fetch_track` now route through it; floats in the bbox are hashed via `f64::to_bits()` so `-0.0` vs `0.0` and locale-sensitive formatting can't introduce silent collisions.
- [x] 3.9 Rate limiting with exponential backoff implemented via `RateLimiter::{wait, failure, success}` in `adsb.rs` — doubles backoff on HTTP errors / 429 responses up to a configured max.
- [x] 3.10 `parse_sbs_msg4_velocity` and `parse_sbs_rejects_non_msg` tests cover SBS message parsing.
- [x] 3.11 `parse_opensky_states_json`, `parse_opensky_states_empty`, and `parse_opensky_track_json` round-trip mock OpenSky JSON payloads through the parser.
- [x] 3.12 `integration_fetch_live_states` is wired with `#[test] #[ignore]` so CI default runs skip it; invoke via `cargo test -- --ignored integration_fetch_live_states` when credentials and network are available.

## 4. Orbital Data Ingestion

- [x] 4.1 `SpaceTrackClient` in `crates/thresh-data/src/orbital.rs` POSTs to `/ajaxauth/login`, captures the session cookie from the `Set-Cookie` response header into a hand-rolled jar (avoids pulling in reqwest's `cookies` feature and its `cookie_store` / `publicsuffix` transitive deps), and replays it on subsequent `/basicspacedata/query/class/gp/...` GETs. `fetch_tle(norad_id)` and `fetch_tles(&[u32])` both parse responses through the shared `parse_gp_json` path. Unit tests cover cookie parsing and the missing-credentials error path; the network-gated `spacetrack_fetch_tle` integration test now exercises the real HTTP pipeline.
- [x] 4.2 `Tle::from_3le` and `Tle::from_2le` implement the two-line / three-line format parser.
- [x] 4.3 `parse_gp_json` parses CelesTrak-style GP JSON arrays into `Tle` structs; covered by `parse_gp_json_basic` unit test.
- [x] 4.4 `propagate_tle` integrates the `sgp4` crate to produce `TemeState` at arbitrary minutes-since-epoch.
- [x] 4.5 `teme_to_ecef_to_enu_chain` and the SGP4 → TEME → ECEF → ENU pipeline are implemented and tested.
- [x] 4.6 `radar_measurements_from_enu` produces synthetic radar measurements from orbital positions given a ground-station configuration.
- [x] 4.7 `predict_passes` computes rise / set / max-elevation for (station, object) pairs.
- [x] 4.8 `OrbitalDataset` implements the `Dataset` trait.
- [x] 4.9 `CelestrakClient` in `crates/thresh-data/src/orbital.rs` fetches public GP JSON from `celestrak.org/NORAD/elements/gp.php`. Both `fetch_gp_group(name)` and `fetch_catnr(norad_id)` delegate to a shared `fetch_json` path that maps non-success HTTP to `OrbitalError::HttpStatus` and parses via `parse_gp_json`. Network-gated integration tests `celestrak_fetch_gp` and `celestrak_fetch_catnr_iss` now exercise the real pipeline.
- [x] 4.10 `parse_3le_iss` / `parse_2le_iss` / `propagate_iss_position_reasonable` validate ISS TLE parsing and 10 km propagation accuracy.
- [x] 4.11 `teme_to_ecef_to_enu_chain` checks the reference-frame chain against a known epoch.
- [x] 4.12 `spacetrack_fetch_tle`, `celestrak_fetch_gp`, and `celestrak_fetch_catnr_iss` are wired as `#[ignore]` integration tests — they now exercise the real HTTP pipeline against www.space-track.org and celestrak.org when run with `--ignored` and appropriate credentials / network access.

## 5. nuScenes Ingestion

- [x] 5.1 Implement PyO3 bridge to nuScenes devkit (feature-gated `nuscenes`)
- [x] 5.2 Implement scene/sample iteration via Python bridge
- [x] 5.3 Parse 3D annotation boxes → `BoundingBox3D` with class mapping
- [x] 5.4 Extract instance-level tracks for ground truth (instance token → target ID)
- [x] 5.5 Parse LiDAR point clouds from binary files (x, y, z, intensity, ring)
- [x] 5.6 Parse radar point clouds with RCS and velocity
- [x] 5.7 Load sensor calibration (extrinsics + intrinsics) for multi-modal alignment
- [x] 5.8 Implement `NuScenesDataset` implementing `Dataset` trait
- [x] 5.9 Write tests: parse annotations from mini split, verify box dimensions
- [x] 5.10 Write integration test (requires mini split): load scene, run tracker, compute AMOTA

## 6. Dataset Abstraction and Mixing

- [x] 6.1 `SyntheticDataset` in `synthetic.rs` adapts `thresh-synth::Scenario` to the `Dataset` trait.
- [x] 6.2 `MixedDataset` in `mixing.rs` performs k-way merge of multiple sources in time order with lazy iteration.
- [x] 6.3 `bucket_frames` in `mixing.rs` groups frames within a configurable time window.
- [x] 6.4 `MixedDataset::frames()` yields a lazy iterator that only holds one frame per source in memory at a time.
- [x] 6.5 `mixed_dataset_merges_in_time_order` test covers the k-way merge.
- [x] 6.6 `bucket_frames_groups_within_window` test covers temporal bucketing at 50 ms.

## 7. Benchmark Scenarios

- [x] 7.1 `ScenarioManifest` struct in `benchmark.rs` defines the TOML manifest format (source, parameters, baselines).
- [x] 7.2 `thresh-data` CLI binary (`crates/thresh-data/src/bin/thresh_data.rs`) with `list` / `run` / `help` subcommands. `list` walks the scenario directory (env `THRESH_DATA_SCENARIOS` or `--dir <path>`); `run <file.toml>` dispatches to the appropriate runner (synthetic or, when built with `--features orbital`, orbital) and prints MOTA / MOTP / IDF1 / HOTA + regression status. ADS-B sources surface "feature required" errors rather than silently running an empty scenario. Orbital sources dispatch to `run_orbital_benchmark` under the `orbital` feature and produce the "rebuild with --features orbital" error otherwise. CLI unit tests cover subcommand dispatch, dir-arg parsing, and source description; integration tests cover both the default and feature-gated build paths.
- [ ] 7.3 Create scenario config: `adsb-single-flight` — blocked on an ADS-B runner dispatch in `run_manifest` (the HTTP clients from §4.1 / §4.9 are live, but an ADS-B-specific ground-truth extraction + tracker wiring path still needs to be written analogous to `run_orbital_benchmark`).
- [ ] 7.4 Create scenario config: `adsb-tracon` — same blockers as 7.3.
- [x] 7.5 `crates/thresh-data/scenarios/orbital-iss.toml` + `orbital-iss.tle` (cached ISS TLE). Runs end-to-end via `run_orbital_benchmark`: loads the TLE, propagates via SGP4 over a 3-hour window, converts ECI → ENU relative to a mid-latitude ground station, generates noisy radar measurements, runs the Cartesian ENU tracker, and computes MOTA / HOTA / IDF1 against the noise-free ground truth. Baselines are intentionally permissive because the CV Kalman is not optimal for the orbital regime — the scenario is a *pipeline* regression gate, not an orbital tracking-accuracy benchmark. Covered by the `run_orbital_iss_cached_tle_end_to_end` integration test and the new `orbital-benchmark-gate` CI job.
- [x] 7.6 `crates/thresh-data/scenarios/orbital-starlink-train.toml` + `orbital-starlink-train.tle` (hand-crafted 5-satellite synthetic train; checksums verified). Same pipeline as §7.5 but exercises the multi-target association path across 5 overlapping tracks. Covered by the `run_orbital_starlink_train_cached_tle_end_to_end` integration test and the same `orbital-benchmark-gate` CI job.
- [ ] 7.7 Create scenario config: `nuscenes-mini` — blocked on adding nuScenes dispatch in `run_manifest`.
- [x] 7.8 Created `crates/thresh-data/scenarios/synth-cv-clean.toml`; it round-trips through `load_scenario` → `run_synthetic_benchmark` and clears its MOTA baseline (MOTA ≈ 0.94 with the default 5-CV trajectory set). Additional synth variants (`synth-maneuvering`, `synth-heterogeneous`, `synth-low-pd`) are deferred until `build_trajectories` learns to dispatch on a scenario-flavour field rather than hard-coding 5 CV targets.
- [x] 7.9 `run_synthetic_benchmark` in `benchmark.rs` is the synthetic scenario runner — builds trajectories, runs the tracker, computes MOTA/MOTP/IDF1/HOTA.
- [x] 7.10 `check_regression` returns a list of baseline failures (empty = pass); the CLI prints each failure and exits non-zero.
- [x] 7.11 Added `synth-benchmark-gate` job to `.github/workflows/ci.yml` — builds the `thresh-data` binary and iterates over every `crates/thresh-data/scenarios/synth-*.toml` file with `thresh-data run`, failing the build if any regression check trips. The glob + `set -euo pipefail` wrapper means new synthetic scenario manifests are picked up automatically without another workflow edit.
- [ ] 7.12 Add nightly CI job for network-dependent benchmarks — pending, needs the scenarios from 7.3-7.7 and a nightly workflow.
