# Scenario Variants -- Tasks

## 1. ScenarioParameters extension

- [ ] 1.1 Add `scenario_type: Option<String>` field to `ScenarioParameters` in `crates/thresh-data/src/benchmark.rs` with `#[serde(default)]` attribute.
- [ ] 1.2 Extract the existing CV-clean logic from `build_trajectories` into a new `build_cv_clean_trajectories(params: &ScenarioParameters) -> Vec<Trajectory>` function (no behaviour change).
- [ ] 1.3 Convert `build_trajectories` into a dispatcher that matches on `params.scenario_type.as_deref()` and routes to the appropriate builder. `None` and `Some("cv-clean")` both route to `build_cv_clean_trajectories`.

## 2. Radar config dispatch

- [ ] 2.1 Add a `radar_config_for_scenario(params: &ScenarioParameters) -> RadarConfig` helper that returns the appropriate `RadarConfig` based on `scenario_type`. Default (cv-clean / maneuvering / heterogeneous) returns the current config (`p_detection = 1.0`, `clutter_rate = 0.0`). `"low-pd"` returns `p_detection = 0.7`, `clutter_rate = 5.0`.
- [ ] 2.2 Update `run_synthetic_benchmark` to call `radar_config_for_scenario` instead of inline-constructing the `RadarConfig`.

## 3. Trajectory builders

- [ ] 3.1 Implement `build_maneuvering_trajectories(params: &ScenarioParameters) -> Vec<Trajectory>`: 4 targets with multi-segment CV/CTRV/CA trajectories, 10 km spacing, using `params.duration_s` and `params.dt`.
- [ ] 3.2 Implement `build_heterogeneous_trajectories(params: &ScenarioParameters) -> Vec<Trajectory>`: 5 targets across three kinematic classes (UAV-like, aircraft-like, missile-like) with class-appropriate speeds, altitudes, and segment types.
- [ ] 3.3 Implement `build_low_pd_trajectories(params: &ScenarioParameters) -> Vec<Trajectory>`: 5 CV targets reusing the same geometry as `build_cv_clean_trajectories` (delegates to it directly).

## 4. Scenario manifests

- [ ] 4.1 Create `crates/thresh-data/scenarios/synth-maneuvering.toml` with `scenario_type = "maneuvering"`, `duration_s = 30.0`, `dt = 1.0`, `measurement_noise_sigma = 50.0`, `gate_threshold = 500.0`, and `mota = 0.3` baseline.
- [ ] 4.2 Create `crates/thresh-data/scenarios/synth-heterogeneous.toml` with `scenario_type = "heterogeneous"` and `mota = 0.3` baseline. Same base parameters.
- [ ] 4.3 Create `crates/thresh-data/scenarios/synth-low-pd.toml` with `scenario_type = "low-pd"` and `mota = 0.2` baseline. Same base parameters.
- [ ] 4.4 Update `synth-cv-clean.toml` to explicitly set `scenario_type = "cv-clean"` (optional but good for documentation; the runner handles its absence via the `None` fallback).

## 5. Tests and verification

- [ ] 5.1 Add a unit test `test_build_maneuvering_trajectories` that verifies the returned vec has 4 trajectories, each with multiple segments, and that `generate()` produces waypoints spanning `duration_s`.
- [ ] 5.2 Add a unit test `test_build_heterogeneous_trajectories` that verifies 5 trajectories with distinct initial velocity magnitudes (UAV < aircraft < missile).
- [ ] 5.3 Add a unit test `test_build_low_pd_trajectories` that verifies it produces the same trajectory count and geometry as `build_cv_clean_trajectories`.
- [ ] 5.4 Add a unit test `test_radar_config_for_scenario` that verifies `p_detection` and `clutter_rate` differ for `"low-pd"` vs the default.
- [ ] 5.5 Add a manifest deserialization test that loads each new TOML file and asserts `scenario_type` parses correctly and baselines are present.
- [ ] 5.6 Run `cargo test -p thresh-data` and `cargo clippy --workspace --all-targets -- -D warnings` to confirm no regressions.
