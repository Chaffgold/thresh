# equinoctial-filter-model Specification

## Purpose
An additive element-space orbital motion model for sigma-point filters — 6D direct equinoctial state propagated through the Gauss variational equations, unwrapped mean longitude so no circular statistics are needed, an exact element-to-ECI-position measurement mapping, and a seeded coast-gap Monte-Carlo demonstration showing the equinoctial filter stays statistically consistent through measurement-free gaps where the Cartesian EKF baseline does not — all without touching existing Cartesian consumers or calibrated benchmarks.

## Requirements

### Requirement: Equinoctial motion model for sigma-point filters
The system SHALL provide a motion model over the 6D direct equinoctial element state, propagating through the Gauss variational equations with the two-body secular term, compatible with the existing UKF/CKF sigma-point machinery (parameterized gravitational constant and maximum sub-step, following the existing Cartesian orbital model's conventions).

#### Scenario: Sigma-point prediction through the element dynamics
- **WHEN** a UKF predicts an equinoctial state with a non-degenerate covariance through a J2-perturbed arc
- **THEN** the predicted mean SHALL match a direct propagation of the prior mean within sigma-point tolerance and the predicted covariance SHALL be symmetric positive definite

### Requirement: Mean longitude survives sigma-point statistics unwrapped
The filter-state mean longitude SHALL be continuous (unwrapped — never reduced modulo 2π in state space), with periodicity owned by the element-to-Cartesian conversion, so sigma-point means and residuals require no circular statistics. This convention SHALL be documented on the model and proven by tests at the wrap boundary and over long horizons.

#### Scenario: Wrap-straddling sigma points produce no artifact
- **WHEN** a prediction runs from a state whose mean longitude sits just below a multiple of 2π with a sigma spread crossing it
- **THEN** the predicted mean and covariance SHALL be free of wrap artifacts (continuous in the input mean, matching a run shifted by exactly 2π)

#### Scenario: Long-horizon accumulation stays accurate
- **WHEN** the model propagates until the unwrapped mean longitude accumulates hundreds of radians
- **THEN** the converted Cartesian state SHALL still agree with the Cartesian-path propagation within the documented cross-formulation tolerance

### Requirement: Element-space measurement mapping
The system SHALL provide the measurement mapping from the equinoctial state to inertial Cartesian position (via the existing element conversions) for position-measurement filters, exact and allocation-conscious, so UKF/CKF measurement updates operate on element states against ECI position measurements.

#### Scenario: Measurement mapping matches the conversion
- **WHEN** the measurement mapping is evaluated on an element state
- **THEN** it SHALL equal the position of the state's existing element-to-Cartesian conversion exactly

### Requirement: Coast-gap consistency demonstration against the Cartesian baseline
A seeded Monte-Carlo experiment SHALL demonstrate the change's payoff with recorded numbers: tracking identical truth arcs (Cartesian two-body+J2) with identical measurements and a shared physical process-noise assumption, then coasting through measurement-free gaps of increasing duration, the equinoctial sigma-point filter SHALL remain inside the two-sided 95% ANEES band at at least one recorded gap duration where the Cartesian EKF baseline has exited it. The full sweep (gap durations, both filters' ANEES, run count, seeds) SHALL be recorded at the test, and repeated runs SHALL be bitwise identical.

#### Scenario: Element filter outlasts the Cartesian filter through coast
- **WHEN** the recorded gap sweep runs
- **THEN** there SHALL exist a recorded gap duration at which the Cartesian baseline's ANEES lies outside the two-sided 95% χ² band while the equinoctial filter's lies inside, with both values recorded

#### Scenario: Demonstration is deterministic
- **WHEN** the sweep runs twice with the same seeds
- **THEN** every recorded ANEES value SHALL be bitwise identical

### Requirement: Existing consumers unaffected
The existing Cartesian models, trackers, synth defaults, and every calibrated benchmark SHALL be unaffected — the element-space model is additive. A verification run SHALL confirm the calibrated benchmark scenarios produce digit-for-digit identical metrics before and after this change.

#### Scenario: Calibrated benchmarks are bitwise unchanged
- **WHEN** the four calibrated benchmark scenarios run on the pre-change and post-change trees
- **THEN** every reported metric SHALL be digit-for-digit identical
