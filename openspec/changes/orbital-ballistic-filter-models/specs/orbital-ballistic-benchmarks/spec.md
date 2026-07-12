# orbital-ballistic-benchmarks Specification (Delta)

## ADDED Requirements

### Requirement: Ballistic tracking benchmark scenario
The system SHALL provide at least one ballistic benchmark scenario, defined declaratively (scenario TOML), that drives the full pipeline — phased ballistic truth generation, synthetic radar measurements, ballistic-head tracking — and reports standard MOT metrics. The scenario SHALL carry a recorded accuracy baseline that the run is checked against.

#### Scenario: Ballistic benchmark end-to-end run
- **WHEN** the benchmark runner executes the ballistic scenario
- **THEN** it SHALL generate a phased boost/midcourse/reentry truth trajectory, produce radar measurements from it, track with the ballistic head, and report MOT metrics including MOTA

#### Scenario: Ballistic baseline enforced
- **WHEN** the ballistic benchmark's reported accuracy falls below the scenario's recorded baseline
- **THEN** the benchmark run SHALL fail so CI surfaces the regression

### Requirement: Real accuracy gate for the orbital benchmark
The existing orbital CI benchmark scenarios (ISS and Starlink-train) SHALL be gated on a measured accuracy baseline rather than the pipeline-smoke placeholders (`mota = -2.0` for ISS, `mota = -5.0` for the Starlink train). The baseline SHALL be established from the tracker's measured performance once it uses the Kepler+J2 orbital model, and it SHALL be strictly positive.

#### Scenario: Orbital gate reflects real accuracy
- **WHEN** the orbital benchmark gate runs after the orbital motion model is integrated into the tracker
- **THEN** each orbital scenario's recorded MOTA baseline SHALL be a positive value derived from a measured run, and the gate SHALL pass at that level

#### Scenario: Orbital accuracy regression detected
- **WHEN** a code change degrades orbital tracking so that MOTA drops below the recorded baseline
- **THEN** the orbital benchmark gate SHALL fail CI, where the previous negative smoke baselines (`-2.0` / `-5.0`) would have passed silently

### Requirement: Benchmark determinism
Orbital and ballistic benchmark scenarios SHALL be deterministic: repeated runs of the same scenario at the same code revision SHALL produce identical metric values, so baseline comparisons are meaningful.

#### Scenario: Repeated runs agree
- **WHEN** the same benchmark scenario is executed twice on the same machine and revision
- **THEN** the reported MOT metrics SHALL be identical between runs
