# state-estimation Specification (Delta)

## ADDED Requirements

### Requirement: Filter update diagnostics exposure
The KF, EKF, and UKF SHALL expose the diagnostics required for NEES/NIS consistency evaluation: at the completion of each measurement update, the innovation `nu = z - z_pred` and the innovation covariance `S` that were actually used inside that update; and the predicted (pre-update) state and covariance, readable after `predict` and before the subsequent `update`. For the sigma-point UKF, the exposed `S` SHALL be the sigma-point-derived innovation covariance computed inside the update — not a caller-side `H P H^T + R` reconstruction, which does not exist for the sigma-point formulation. The exposure SHALL be non-breaking: every existing `predict` / `update` call site SHALL compile and behave unchanged, with no change to the existing method signatures' accepted arguments.

#### Scenario: Linear KF diagnostics match closed form
- **WHEN** a linear KF performs an update with measurement `z`, observation matrix `H`, and noise `R` from a known predicted state `x_pred` and covariance `P_pred`
- **THEN** the exposed innovation SHALL equal `z - H x_pred` and the exposed `S` SHALL equal `H P_pred H^T + R`, each within floating-point tolerance, and the predicted state/covariance readable between `predict` and `update` SHALL equal `(x_pred, P_pred)`

#### Scenario: EKF diagnostics reflect the linearized update
- **WHEN** an EKF performs an update with nonlinear observation function `h` and Jacobian `H_k` evaluated at the predicted state
- **THEN** the exposed innovation SHALL equal `z - h(x_pred)` and the exposed `S` SHALL equal `H_k P_pred H_k^T + R`, matching the quantities used to form the Kalman gain in that update

#### Scenario: UKF sigma-point S is exposed, not reconstructed
- **WHEN** a UKF performs a nonlinear update and its internally computed sigma-point innovation covariance is compared against the exposed `S`
- **THEN** they SHALL be identical (same computation, not an approximation), and on a linear measurement model the exposed `S` SHALL agree with the linear KF's `H P H^T + R` within sigma-point numerical tolerance

#### Scenario: Diagnostics feed NIS directly
- **WHEN** the exposed innovation and `S` from any of the three filters are passed to the `thresh-eval` NIS computation over a consistent linear-Gaussian run of `N` updates with measurement dimension `m`
- **THEN** the resulting average NIS SHALL fall inside the two-sided 95% bounds of the time-average test (chi-squared with `N·m` degrees of freedom, divided by `N`), with no ground-truth input required

#### Scenario: Existing callers are unaffected
- **WHEN** the workspace is built and tested after the diagnostics are added
- **THEN** all pre-existing callers of the filters SHALL compile without modification and all pre-existing filter tests SHALL pass with unchanged expected values

#### Scenario: Diagnostics validity is well-defined before any update
- **WHEN** client code attempts to obtain update diagnostics before any `update` call has completed
- **THEN** the API SHALL make the not-yet-available state unrepresentable or explicit — e.g., diagnostics that exist only as each `update` call's returned value (so no pre-update query surface exists), an `Option`/`Result`, or an explicitly documented absent value — rather than returning uninitialized or misleading numbers
