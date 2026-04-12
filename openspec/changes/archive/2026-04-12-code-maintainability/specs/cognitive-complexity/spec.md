## Capability: Cognitive Complexity

### Overview
Enforce that six specific functions flagged by SonarCloud (rule `rust:S3776`) remain at or below cognitive complexity 15 while preserving their existing behavior and test coverage. These functions span `thresh-association`, `thresh-data`, `thresh-tracker`, and `thresh-synth`, and are refactored by extracting phase helpers with descriptive names.

## ADDED Requirements

### Requirement: Hungarian algorithm cognitive complexity
The `hungarian_assignment` function in `crates/thresh-association/src/hungarian.rs` SHALL have a cognitive complexity of at most 15 as measured by SonarCloud rule `rust:S3776`, decomposed into named phase helpers (cost-matrix reduction, augmenting-path search, label/slack update). The system MUST support this.

#### Scenario: Hungarian assignment below complexity threshold
- **WHEN** SonarCloud analyzes the refactored `hungarian.rs` and the existing association tests are run
- **THEN** the `hungarian_assignment` function SHALL report cognitive complexity ≤ 15, AND all existing `thresh-association` tests SHALL pass unchanged, AND a randomized comparison harness SHALL confirm the new implementation produces a valid one-to-one assignment with the same minimal total cost as the previous implementation over at least 10,000 random cost matrices, requiring identical assignment vectors only when the optimum is unique or when deterministic tie-breaking is explicitly specified

### Requirement: ADS-B `extract_ground_truth` cognitive complexity
The `extract_ground_truth` function in `crates/thresh-data/src/adsb.rs` SHALL have a cognitive complexity of at most 15 as measured by SonarCloud rule `rust:S3776`, with per-ICAO24 grouping, 1-second grid interpolation, and short-trajectory handling extracted into helpers. The system MUST support this.

#### Scenario: Ground-truth extraction below complexity threshold
- **WHEN** SonarCloud analyzes the refactored `adsb.rs` and the existing ADS-B tests are run
- **THEN** `extract_ground_truth` SHALL report cognitive complexity ≤ 15, AND all existing `thresh-data` ADS-B tests SHALL pass unchanged, AND the refactored function SHALL produce byte-identical ground-truth output to the previous implementation on the existing test fixtures

### Requirement: Stereographic tracker long-traverse test cognitive complexity
The long-traverse integration test in `crates/thresh-tracker/tests/stereographic_tracker_tests.rs` SHALL have a cognitive complexity of at most 15 as measured by SonarCloud rule `rust:S3776`, with measurement generation, per-step tracker updates, and final-error computation extracted into helpers. The system MUST support this.

#### Scenario: Long-traverse test below complexity threshold
- **WHEN** SonarCloud analyzes the refactored stereographic tracker test and `cargo test -p thresh-tracker` is run
- **THEN** the long-traverse test function SHALL report cognitive complexity ≤ 15, AND the test SHALL exercise the same measurements and assertions as before the refactor, AND the test SHALL pass

### Requirement: Orbital dataset frame generation cognitive complexity
The `OrbitalDataset::frames` function in `crates/thresh-data/src/orbital.rs` SHALL have a cognitive complexity of at most 15 as measured by SonarCloud rule `rust:S3776`, with per-frame construction and ground-truth entry construction extracted into helpers. The system MUST support this.

#### Scenario: Orbital dataset frame generation below complexity threshold
- **WHEN** SonarCloud analyzes the refactored `orbital.rs` and the existing orbital dataset tests are run
- **THEN** `OrbitalDataset::frames` SHALL report cognitive complexity ≤ 15, AND all existing `thresh-data` orbital tests SHALL pass unchanged, AND the refactored function SHALL produce identical frame output to the previous implementation on the existing test fixtures

### Requirement: `Trajectory::generate` cognitive complexity
The `Trajectory::generate` function in `crates/thresh-synth/src/trajectory.rs` SHALL have a cognitive complexity of at most 15 as measured by SonarCloud rule `rust:S3776`, with per-segment waypoint generation extracted into a helper. The system MUST support this.

#### Scenario: Trajectory generation below complexity threshold
- **WHEN** SonarCloud analyzes the refactored `trajectory.rs` and the existing trajectory tests are run
- **THEN** `Trajectory::generate` SHALL report cognitive complexity ≤ 15, AND all existing `thresh-synth` trajectory tests SHALL pass unchanged, AND the refactored function SHALL produce identical waypoint output to the previous implementation for the same input segments

### Requirement: Orbital RK4 propagator step cognitive complexity
The RK4 propagator step function in `crates/thresh-synth/src/orbital.rs` SHALL have a cognitive complexity of at most 15 as measured by SonarCloud rule `rust:S3776`, with the per-stage (k1/k2/k3/k4) computation extracted into a shared helper. The system MUST support this.

#### Scenario: RK4 propagator step below complexity threshold
- **WHEN** SonarCloud analyzes the refactored orbital `orbital.rs` propagator and the existing orbital propagation tests are run
- **THEN** the RK4 step function SHALL report cognitive complexity ≤ 15, AND all existing `thresh-synth` orbital propagation tests SHALL pass unchanged, AND the refactored propagator SHALL produce numerically identical state updates to the previous implementation for the same initial conditions
