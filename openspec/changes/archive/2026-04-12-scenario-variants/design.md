# Scenario Variants -- Design

## Context

The synthetic benchmark runner in `thresh-data` currently hard-codes a single scenario shape inside `build_trajectories`: five constant-velocity targets with perfect detection probability and zero clutter (`synth-cv-clean`). This is a useful smoke test but too easy to be a meaningful regression gate. Real tracker failures -- track fragmentation during maneuvers, ID switches between dissimilar targets, missed associations under clutter -- are invisible to this single scenario.

The benchmark infrastructure (manifest loading, `run_synthetic_benchmark`, metric evaluation, regression checking, CI glob-based scenario discovery) is already in place. The only missing piece is the ability to select different trajectory/sensor configurations from the scenario manifest.

## Goals / Non-Goals

**Goals:**

- Support three new synthetic scenario flavours alongside the existing `synth-cv-clean`: maneuvering targets, heterogeneous target classes, and low-probability-of-detection with clutter.
- Each variant exercises a distinct tracker failure mode with fully deterministic ground truth.
- Backwards-compatible: existing manifests that omit the new field behave identically to today.
- No changes to the CI workflow -- the `synth-benchmark-gate` job already globs `synth-*.toml`.

**Non-Goals:**

- New sensor models or observation types (the existing `RadarConfig` is sufficient).
- Scenario-specific tracker parameter tuning (the tracker runs with default settings).
- Non-synthetic scenarios (ADS-B, orbital, nuScenes are separate concerns).
- Multi-sensor fusion scenarios (single radar per scenario for now).

## Decisions

### D1: `scenario_type` field on `ScenarioParameters`

Add an optional string field to `ScenarioParameters`:

```rust
#[serde(default)]
pub scenario_type: Option<String>,
```

Accepted values: `"cv-clean"` (explicit), `"maneuvering"`, `"heterogeneous"`, `"low-pd"`. When `None`, the runner falls back to the existing `build_trajectories` logic (CV-clean), preserving backwards compatibility for every manifest already on disk.

**Why `Option<String>` instead of an enum?** Consistency with `tracker_variant`, which already uses `Option<TrackerVariant>` with `#[serde(default)]`. A string keeps the TOML human-editable without requiring users to know Rust enum syntax. Validation happens at dispatch time inside `build_trajectories` with a clear error message for unknown values.

### D2: Dispatch inside `build_trajectories`

`build_trajectories` becomes a thin dispatcher:

```rust
fn build_trajectories(params: &ScenarioParameters) -> Vec<Trajectory> {
    match params.scenario_type.as_deref() {
        None | Some("cv-clean") => build_cv_clean_trajectories(params),
        Some("maneuvering")     => build_maneuvering_trajectories(params),
        Some("heterogeneous")   => build_heterogeneous_trajectories(params),
        Some("low-pd")          => build_low_pd_trajectories(params),
        Some(other)             => panic!("unknown scenario_type: {other}"),
    }
}
```

The existing CV-clean logic moves into `build_cv_clean_trajectories` unchanged.

### D3: `build_maneuvering_trajectories`

Produces 4 targets with multi-segment trajectories that stitch CV, CTRV, and CT segments:

- Target 0: CV 10 s, CTRV (omega = 0.05 rad/s) 10 s, CV 10 s
- Target 1: CV 5 s, CA (lateral acceleration) 10 s, CV 15 s
- Target 2: CTRV (omega = -0.03) 15 s, CV 15 s
- Target 3: CV 15 s, CTRV (omega = 0.08) 15 s

Initial positions are well-separated (10 km spacing) so association difficulty comes from the dynamics, not proximity. The `RadarConfig` stays at `p_detection = 1.0` and `clutter_rate = 0.0` to isolate the maneuvering challenge.

### D4: `build_heterogeneous_trajectories`

Produces 5 targets representing three classes with distinct kinematic regimes:

- Targets 0-1 (UAV-like): low speed (30-50 m/s), low altitude (500 m), tight turns (CTRV segments with high turn rate).
- Target 2 (aircraft-like): medium speed (200 m/s), medium altitude (8000 m), gentle turns.
- Targets 3-4 (missile-like): high speed (600-800 m/s), high altitude (15000 m), CA boost then ballistic.

This exercises the tracker's ability to maintain correct associations when targets have very different state magnitudes and update rates.

### D5: `build_low_pd_trajectories`

Produces 5 CV targets (same geometry as `cv-clean`) but the runner sets `RadarConfig` to:

- `p_detection = 0.7`
- `clutter_rate = 5.0` (false alarms per scan)

This means `run_synthetic_benchmark` needs to read `scenario_type` when building the `RadarConfig`, not just when building trajectories. The cleanest approach: `build_trajectories` returns the trajectory vec as before, and a new helper `radar_config_for_scenario` returns the `RadarConfig`. Both are dispatched by `scenario_type`.

### D6: Scenario manifests

Three new TOML files in `crates/thresh-data/scenarios/`:

| File | `scenario_type` | Baselines |
|---|---|---|
| `synth-maneuvering.toml` | `"maneuvering"` | `mota = 0.3` (lenient initial gate) |
| `synth-heterogeneous.toml` | `"heterogeneous"` | `mota = 0.3` |
| `synth-low-pd.toml` | `"low-pd"` | `mota = 0.2` |

Initial baseline thresholds are intentionally low -- the point is to establish a floor that catches catastrophic regressions, not to demand high performance from day one. Baselines will be tightened as the tracker improves.

All three use `duration_s = 30.0`, `dt = 1.0`, and `measurement_noise_sigma = 50.0` to match the existing `synth-cv-clean` parameters and keep comparison fair.

### D7: CI impact

The existing `synth-benchmark-gate` CI job discovers scenarios via a `synth-*.toml` glob. Adding new files with the `synth-` prefix is sufficient -- no workflow file changes are needed.

## Risks / Trade-offs

- **Lenient baselines may not catch real regressions.** Mitigated by starting low and tightening after a few CI runs establish the tracker's actual performance on each variant.
- **`panic!` on unknown `scenario_type`.** Acceptable for a developer-facing benchmark tool. If this were user-facing API surface, a `Result` return would be preferred, but the benchmark runner already panics on other invalid inputs (e.g., missing TOML fields).
- **String-based dispatch instead of enum.** Slightly less type-safe, but keeps TOML authoring simple and matches the existing pattern for `tracker_variant`. If the variant count grows beyond ~6, converting to a proper serde enum is straightforward.
- **Deterministic but non-randomized trajectories.** Each builder uses fixed initial conditions. This is intentional -- reproducible benchmarks are more valuable than stochastic ones for regression gating. Monte Carlo runs are a separate concern.

## Open Questions

1. **Should `build_low_pd_trajectories` also vary trajectory geometry, or reuse CV-clean geometry?** Current decision: reuse CV-clean to isolate the detection/clutter challenge. Could revisit if both challenges should be combined.
2. **Should the `scenario_type` field eventually become a proper Rust enum with `#[serde(rename_all = "kebab-case")]`?** Deferring until the number of variants stabilizes.
