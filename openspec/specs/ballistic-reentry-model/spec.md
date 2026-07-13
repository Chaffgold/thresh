# ballistic-reentry-model Specification

## Purpose
Physics-based 7D ballistic reentry motion model in `thresh-filter`: position, velocity, and estimated ballistic coefficient beta, combining round-rotating-Earth gravity with altitude-dependent exponential atmospheric drag, usable by EKF/UKF/CKF via the `MotionModel` trait.

## Requirements

### Requirement: 7D ballistic reentry motion model
`thresh-filter` SHALL provide a ballistic reentry motion model whose state is 7D: position (3), velocity (3), and ballistic coefficient beta. Its dynamics SHALL combine gravity over a round rotating Earth with atmospheric drag using an altitude-dependent exponential density profile, where drag deceleration scales inversely with beta. The model SHALL implement the existing `MotionModel` trait so EKF, UKF, and CKF filters accept it without filter-code changes.

#### Scenario: Drag deceleration matches the beta parameterization
- **WHEN** the model dynamics are evaluated at a state with altitude h, speed v, and ballistic coefficient beta inside the atmosphere
- **THEN** the drag deceleration magnitude SHALL equal rho(h) * v^2 / (2 * beta) with rho(h) from the shared exponential atmosphere profile, directed opposite the air-relative velocity

#### Scenario: Exoatmospheric limit
- **WHEN** the state's altitude is above the atmosphere model's ceiling
- **THEN** the drag term SHALL vanish and the predicted trajectory SHALL match gravity-only ballistic propagation of the same initial state

#### Scenario: Prediction through all three nonlinear filters
- **WHEN** an EKF, a UKF, and a CKF are each constructed with the reentry model and predicted forward from the same reentry-interface state and covariance
- **THEN** each filter SHALL produce a physically consistent predicted mean and a symmetric positive-definite predicted covariance

### Requirement: Ballistic coefficient estimation
The reentry model SHALL treat beta as an estimated state component so that a filter tracking a reentering object refines beta from observed deceleration. Beta SHALL remain physically valid (strictly positive) throughout filtering.

#### Scenario: Beta convergence during high-drag flight
- **WHEN** a filter using the reentry model tracks a synthetic reentry trajectory with position measurements, starting from a beta estimate biased by at least 50% from truth
- **THEN** the beta estimate SHALL converge toward the true value during the high-drag portion of the trajectory, ending with substantially smaller error than it started with

#### Scenario: Beta stays positive
- **WHEN** measurement updates would otherwise drive the beta estimate negative
- **THEN** the model or filter integration SHALL keep the effective beta strictly positive so the dynamics remain well-defined

### Requirement: Reentry prediction validated against truth generation
Reentry model prediction SHALL be validated against the phased ballistic truth generator: predicting with the true beta over a truth-generated reentry arc SHALL reproduce the truth trajectory, since both sides share the same force math.

#### Scenario: Prediction matches truth with true beta
- **WHEN** the model predicts across a truth-generated reentry segment using the truth's initial state and true beta
- **THEN** the predicted positions SHALL track the truth trajectory within a documented tolerance over the segment
