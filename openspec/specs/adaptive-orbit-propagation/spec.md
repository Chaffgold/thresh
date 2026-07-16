# adaptive-orbit-propagation Specification

## Purpose
Deterministic adaptive-step Dormand–Prince RK5(4) orbit propagation over the shared time-aware force closures — PI step-size control with configurable tolerances, grid-exact output sampling, and GCRF/`Epoch`-disciplined entry points — validated against independently generated golden trajectories while leaving the fixed-step RK4 seam, existing filter models, and calibrated benchmarks bitwise unchanged.

## Requirements

### Requirement: Dormand-Prince adaptive integration
The system SHALL provide an adaptive-step Dormand–Prince RK5(4) integrator (embedded pair, FSAL) with PI step-size control, configurable absolute and relative tolerances, and min/max step clamps, operating on time-aware acceleration closures. Tightening tolerances SHALL reduce global error at the documented order, and the integrator SHALL be deterministic: identical inputs produce bitwise-identical trajectories.

#### Scenario: Convergence at the design order on a Kepler orbit
- **WHEN** a two-body orbit with known analytic solution is propagated for one period at two tolerance settings an order of magnitude apart
- **THEN** the tighter tolerance SHALL yield a smaller position error against the analytic solution, both errors within their tolerance-implied envelopes

#### Scenario: Deterministic adaptive stepping
- **WHEN** the same arc is propagated twice with identical configuration
- **THEN** the accepted-step sequence and every sampled state SHALL be bitwise identical

### Requirement: Grid-exact output sampling
The integrator SHALL deliver states at caller-requested sample times exactly — by clamping steps to land on requested times rather than interpolating — so downstream consumers (truth grids, golden comparisons) see integration-accurate states at their own timestamps.

#### Scenario: Samples land exactly on the requested grid
- **WHEN** a propagation requests samples at a fixed cadence
- **THEN** every returned state SHALL carry exactly the requested time offsets, each produced by an integration step ending at that time (no interpolation)

### Requirement: GCRF and Epoch-disciplined propagation entry points
High-fidelity propagation SHALL integrate in GCRF with the arc anchored to a time-scale-aware `Epoch`, accepting and returning frame-tagged states, so the propagation output composes with the reference-frame transforms without implicit-frame ambiguity.

#### Scenario: Propagation preserves frame and epoch discipline
- **WHEN** a GCRF-tagged state with epoch t₀ is propagated for Δt
- **THEN** the result SHALL be GCRF-tagged with epoch exactly t₀ + Δt

### Requirement: Trajectory golden validation
End-to-end propagation SHALL be validated against committed golden trajectories produced by an independently implemented force stack and integrator, covering at least: a full-force LEO arc, a harmonics-plus-lunisolar MEO arc, and an SRP-dominant arc crossing Earth shadow. Envelope tests SHALL run under default features with no Python and no network, with per-arc tolerances derived from measured cross-implementation agreement and recorded with the fixtures, and repeated runs SHALL be bitwise identical.

#### Scenario: Golden arcs reproduced within measured tolerance
- **WHEN** the three committed golden arcs are propagated from their fixture initial states with the fixture force configurations
- **THEN** every sampled position SHALL agree with the fixture within its documented tolerance

#### Scenario: Higher fidelity than the J2 baseline
- **WHEN** the full-force LEO golden arc is propagated once with the full force stack and once with two-body+J2 only
- **THEN** the full-stack trajectory SHALL track the golden fixture measurably closer than the J2-only trajectory (the fidelity this capability exists to add, demonstrated numerically)

### Requirement: Existing consumers unaffected
The fixed-step RK4 seam, the `KeplerJ2` and `BallisticReentry` filter models, the thresh-synth default truth paths, and every calibrated benchmark SHALL be unaffected: the adaptive integrator and force stack are additive. A verification run SHALL confirm the calibrated benchmark scenarios produce bitwise-identical metrics before and after this change.

#### Scenario: Calibrated benchmarks are bitwise unchanged
- **WHEN** the four calibrated benchmark scenarios run on the pre-change and post-change trees
- **THEN** every reported metric SHALL be digit-for-digit identical
