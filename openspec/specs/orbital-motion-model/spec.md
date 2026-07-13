# orbital-motion-model Specification

## Purpose
Kepler+J2 orbital motion model in `thresh-filter` on a 6D ECI Cartesian state, with force math shared between filter prediction and `thresh-synth` truth generation, validated against checked-in golden vectors.

## Requirements

### Requirement: Kepler+J2 orbital motion model
`thresh-filter` SHALL provide an orbital motion model whose state is 6D ECI Cartesian position and velocity and whose dynamics are two-body gravity plus the J2 zonal-harmonic perturbation. The model SHALL implement the existing `MotionModel` trait so that EKF, UKF, and CKF filters accept it without any filter-code changes, and it SHALL provide the state-transition Jacobian required for EKF linearization. The gravitational parameter SHALL be a constructor parameter, not a hardcoded Earth constant.

#### Scenario: Prediction through all three nonlinear filters
- **WHEN** an EKF, a UKF, and a CKF are each constructed with the orbital motion model and predicted forward one time step from the same LEO initial state and covariance
- **THEN** each filter SHALL produce a predicted mean consistent with Kepler+J2 propagation of the initial state, and each predicted covariance SHALL remain symmetric positive definite

#### Scenario: Non-Earth gravitational parameter
- **WHEN** the model is constructed with a gravitational parameter other than Earth's (e.g., the lunar mu)
- **THEN** predictions SHALL use the supplied mu throughout, with no Earth-specific constant leaking into the two-body term

#### Scenario: Jacobian consistency
- **WHEN** the state-transition Jacobian reported by the model is compared against an independently derived linearization of the same dynamics (e.g., the analytic two-body+J2 partials) at a representative LEO state
- **THEN** the two SHALL agree element-wise within a tight numerical tolerance

### Requirement: Shared force math between filters and truth generation
The two-body, J2, and atmospheric-drag acceleration functions SHALL live in a single shared module consumed by both the filter motion models and the `thresh-synth` truth generator, so that prediction and truth use identical physics. Existing `thresh-synth` behavior SHALL be preserved (re-export shims are acceptable), and no new external dependency SHALL be introduced for this math.

#### Scenario: Identical accelerations from both consumers
- **WHEN** the same ECI state is evaluated through the filter model's dynamics and through the `thresh-synth` propagator's force function
- **THEN** the computed acceleration vectors SHALL be bitwise identical (same code path), not merely approximately equal

#### Scenario: Synth propagation unchanged after extraction
- **WHEN** the existing `thresh-synth` orbital propagation tests run after the force math moves to its shared home
- **THEN** they SHALL pass without modification to their expected values

### Requirement: Golden-vector validation of orbital prediction
Orbital model prediction SHALL be validated against golden-vector fixtures checked into the repository and generated offline: Vallado worked examples for Kepler/J2 propagation and SGP4-derived state vectors (via the Rust `sgp4` crate's validated output). Validation SHALL run as plain `cargo test` — deterministic, with no Python and no network access in CI.

#### Scenario: Vallado Kepler propagation fixture
- **WHEN** a fixture initial state from a Vallado worked example is propagated for the fixture's time span using the two-body dynamics
- **THEN** the resulting position and velocity SHALL match the fixture's expected state within the documented tolerance

#### Scenario: J2 secular drift fixture
- **WHEN** a LEO orbit is propagated over multiple revolutions with J2 enabled
- **THEN** the secular drift of the ascending node SHALL match the analytic J2 nodal-regression rate within the documented tolerance

#### Scenario: Deterministic CI execution
- **WHEN** the golden-vector tests execute in CI
- **THEN** they SHALL require no Python interpreter, no network access, and no feature gates beyond the default build, and repeated runs SHALL produce identical results
