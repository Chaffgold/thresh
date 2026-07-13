# cubature-kalman-filter Specification (Delta)

## ADDED Requirements

### Requirement: CKF update diagnostics parity with UKF
The `CubatureKalmanFilter` SHALL expose the same update-diagnostics surface added to the UKF: at the completion of each measurement update, the innovation and the cubature-point-derived innovation covariance `S` actually used in that update; plus the predicted (pre-update) state and covariance, readable after `predict` and before the subsequent `update`. The exposed `S` SHALL be the value computed from the propagated cubature points inside `update` / `update_linear` — it is not reconstructable by callers. The addition SHALL be non-breaking for existing CKF callers and SHALL preserve the CKF's spec'd API parity with `UnscentedKalmanFilter`: the diagnostics SHALL be exposed through the same mechanism, with the same names, types, and pre-first-update semantics as the UKF's.

#### Scenario: Cubature-point S is exposed from the update
- **WHEN** a CKF performs a nonlinear update and its internally computed cubature-point innovation covariance is compared against the exposed `S`
- **THEN** they SHALL be identical (same computation, not an approximation)

#### Scenario: Diagnostics parity with UKF on a linear-Gaussian problem
- **WHEN** a CKF and a UKF are initialised identically and driven with the same linear measurement model and the same measurement sequence
- **THEN** their exposed innovations SHALL agree within `1e-6` per element and their exposed innovation covariances within `1e-4` per element, mirroring the existing CKF/UKF posterior-parity tolerances

#### Scenario: CKF diagnostics feed NIS
- **WHEN** the CKF's exposed innovation and `S` from a consistent linear-Gaussian run of `N` updates with measurement dimension `m` are passed to the `thresh-eval` NIS computation
- **THEN** the average NIS SHALL fall inside the two-sided 95% bounds of the time-average test (chi-squared with `N·m` degrees of freedom, divided by `N`)

#### Scenario: Existing CKF callers and tests are unaffected
- **WHEN** the workspace is built and tested after the diagnostics are added
- **THEN** all pre-existing CKF call sites SHALL compile without modification and all pre-existing CKF tests SHALL pass with unchanged expected values
