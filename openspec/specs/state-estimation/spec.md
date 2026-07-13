# state-estimation Specification

## Purpose
TBD - created by archiving change transformer-fusion-tracker. Update Purpose after archive.
## Requirements
### Requirement: Linear Kalman Filter implementation
The system SHALL implement a linear Kalman filter with configurable state transition matrix **F**, observation matrix **H**, process noise covariance **Q**, and measurement noise covariance **R**. The filter SHALL use the Joseph form for covariance updates to ensure numerical stability: P_{k|k} = (I - K_k H_k) P_{k|k-1} (I - K_k H_k)^T + K_k R_k K_k^T.

#### Scenario: Constant velocity prediction and update
- **WHEN** a KF is initialized with a constant-velocity state [x, vx, y, vy] and receives a position-only measurement [x_meas, y_meas]
- **THEN** the filter SHALL produce a predicted state using F, compute the Kalman gain K = P_{k|k-1} H^T S^{-1}, and update the state estimate with the innovation z - H x_hat

#### Scenario: Covariance remains positive semi-definite
- **WHEN** the filter runs for 1000+ sequential updates with noisy measurements
- **THEN** the state covariance P SHALL remain symmetric and positive semi-definite (all eigenvalues >= 0)

### Requirement: Extended Kalman Filter implementation
The system SHALL implement an EKF that accepts user-defined nonlinear state transition function f(x, u) and observation function h(x), along with their Jacobians F_k = df/dx and H_k = dh/dx. The Jacobians SHALL be evaluated at the current state estimate each step.

#### Scenario: CTRV motion model tracking
- **WHEN** an EKF is configured with a CTRV motion model (state [x, y, theta, v, omega]) and receives position measurements
- **THEN** the filter SHALL propagate the state using the nonlinear CTRV equations, linearize via Jacobians, and correctly track a target executing a coordinated turn

#### Scenario: Degenerate turn rate handling
- **WHEN** the turn rate omega approaches zero in a CTRV model
- **THEN** the filter SHALL gracefully degenerate to straight-line constant-velocity motion without numerical instability

### Requirement: Unscented Kalman Filter implementation
The system SHALL implement a UKF using the scaled sigma point selection (Van der Merwe) with configurable parameters alpha, beta, kappa. The UKF SHALL generate 2n+1 sigma points, propagate them through the nonlinear function, and recover mean and covariance from weighted combinations.

#### Scenario: Sigma point generation
- **WHEN** a UKF is initialized with state dimension n=5, alpha=1e-3, beta=2, kappa=0
- **THEN** the system SHALL generate 11 sigma points with correct mean and covariance weights (W_0^m = lambda/(n+lambda), W_0^c includes the (1 - alpha^2 + beta) correction)

#### Scenario: Second-order accuracy on nonlinear transform
- **WHEN** a UKF propagates a Gaussian state through a known nonlinear function (e.g., polar-to-Cartesian)
- **THEN** the recovered mean and covariance SHALL be accurate to second order, outperforming EKF linearization for the same scenario

### Requirement: Configurable motion models
The system SHALL provide a trait/interface for motion models with implementations for: constant velocity (CV), constant acceleration (CA), constant turn rate and velocity (CTRV), and Cartesian coordinated turn. Each model SHALL define its state vector, transition function, transition Jacobian, and default process noise.

#### Scenario: Swappable motion models
- **WHEN** a filter instance is configured with a CV model and the user switches to CTRV
- **THEN** the state vector dimension and transition dynamics SHALL update accordingly, with appropriate state augmentation or projection

#### Scenario: Custom motion model
- **WHEN** a user implements the motion model trait with custom dynamics (e.g., ballistic trajectory with drag)
- **THEN** the filter SHALL accept and use the custom model without modification to the filter code

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

