# Synthetic Benchmark Scenario Variants

## What

Extend the synthetic benchmark runner in thresh-data to support maneuvering, heterogeneous (mixed target classes), and low-probability-of-detection scenarios beyond the current synth-cv-clean baseline. Each variant exercises a different failure mode of the tracker with controlled ground truth for regression testing.

## Why

The synth-cv-clean scenario only tests constant-velocity targets with perfect detection probability, making it too easy to pass. Real tracker regressions -- missed associations under clutter, track fragmentation during maneuvers, ID switches between dissimilar targets -- hide behind this single easy scenario. The benchmark runner infrastructure already exists, but `build_trajectories` hard-codes 5 CV targets with no missed detections. Adding scenario variants turns the benchmark suite into a meaningful regression gate.

## How

- Add a `scenario_type` enum field to `ScenarioParameters` with variants: `CvClean` (existing), `Maneuvering`, `Heterogeneous`, `LowPd`
- Implement `build_maneuvering_trajectories`: targets that switch between CV and CT segments with known switch times
- Implement `build_heterogeneous_trajectories`: mixed target classes (UAV, aircraft, missile) with class-appropriate dynamics and RCS
- Implement `build_low_pd_trajectories`: CV targets with configurable P_d (0.5-0.9) and spatially varying clutter density
- Add TOML scenario manifests in `scenarios/` that parameterize each variant
- Wire scenario selection into the benchmark CLI and CI test matrix

## Out of scope

- Non-synthetic scenario variants (ADS-B replay, orbital propagation scenarios already exist separately)
- New sensor models beyond what thresh-synth already provides
- Scenario-specific tuning of tracker parameters (the tracker should be tested with default settings)

## Affected crates

- thresh-data: benchmark runner extensions, trajectory builders, scenario manifests, CLI integration
