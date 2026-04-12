## ADDED Requirements

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
