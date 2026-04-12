## Capability: Benchmark Scenarios

### Overview
Curated benchmark scenarios with known characteristics and expected metric baselines, enabling regression testing and cross-domain tracker validation. Scenarios span synthetic (thresh-synth), ADS-B (OpenSky), orbital (TLE/SGP4), and automotive (nuScenes) domains with cached offline fixtures so the CI benchmark gate runs without network access.

## ADDED Requirements

### Requirement: Scenario catalog
The system MUST provide named scenarios with documented characteristics (target count, dynamics type, sensor modality, difficulty level) loadable from TOML manifest files.

#### Scenario: List available benchmark scenarios
- **WHEN** the scenario directory is scanned by the CLI
- **THEN** it SHALL return all named scenarios with their source type, description, and manifest path

### Requirement: Expected baselines
Each scenario MUST define MOTA / HOTA / IDF1 baseline thresholds so the regression gate can verify tracker output against a known floor.

#### Scenario: Retrieve metric baselines for a scenario
- **WHEN** a scenario manifest with baselines is loaded
- **THEN** the baselines SHALL include optional MOTA, HOTA, and IDF1 thresholds

### Requirement: Benchmark runner
The system MUST execute the tracker against a scenario, compute MOT metrics, and compare to baselines. Source-specific runners (synthetic, ADS-B, orbital, nuScenes) SHALL be feature-gated so they only link their dependencies when explicitly requested.

#### Scenario: Run tracker against a benchmark and compare metrics
- **WHEN** the CLI runs a scenario manifest
- **THEN** it SHALL execute the appropriate source-specific runner, print MOTA / MOTP / IDF1 / HOTA, and exit non-zero if any baseline is violated

### Requirement: Regression detection
The system MUST flag when computed metrics drop below the baseline thresholds defined in the manifest.

#### Scenario: Detect metric regression
- **WHEN** a benchmark run produces metrics below the manifest's baseline thresholds
- **THEN** the runner SHALL report each violated threshold and exit with a non-zero status code

### Requirement: CI benchmark gate
The system MUST provide CI workflow jobs that run all checked-in scenario manifests and fail the build on any regression. Synthetic, ADS-B, and orbital gates SHALL run offline using cached fixtures; network-dependent tests SHALL run on a nightly schedule.

#### Scenario: Offline CI gate catches regression
- **WHEN** a PR changes code that degrades tracker performance on a cached scenario
- **THEN** the benchmark-gate CI job SHALL fail, blocking the merge
