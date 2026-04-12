# sensor-fusion Specification

## Purpose
TBD - created by archiving change transformer-fusion-tracker. Update Purpose after archive.
## Requirements
### Requirement: Centralized measurement-level fusion
The system SHALL support stacking measurements from multiple sensors into a centralized measurement vector z_stacked = [z_1; z_2; ...; z_N] with block-diagonal measurement noise R_stacked = blkdiag(R_1, R_2, ..., R_N) and stacked observation matrix H_stacked = [H_1; H_2; ...; H_N], enabling a single filter update with all sensor data.

#### Scenario: Dual-sensor fusion update
- **WHEN** a radar provides range/bearing measurements and an EO/IR sensor provides azimuth/elevation measurements for the same target
- **THEN** the system SHALL construct the stacked measurement vector and matrices and perform a single Kalman filter update incorporating both sensors

#### Scenario: Heterogeneous sensor rates
- **WHEN** sensors report at different rates (e.g., radar at 10 Hz, EO/IR at 30 Hz)
- **THEN** the system SHALL support asynchronous updates, applying each sensor's measurement independently when it arrives rather than waiting for all sensors

### Requirement: Information filter form
The system SHALL implement the information filter parameterized by information matrix Y = P^{-1} and information state y_hat = P^{-1} x_hat, with additive update: Y_{k|k} = Y_{k|k-1} + H^T R^{-1} H and y_hat_{k|k} = y_hat_{k|k-1} + H^T R^{-1} z.

#### Scenario: Decentralized sensor contributions
- **WHEN** multiple sensor nodes each compute their local information contribution (H_i^T R_i^{-1} H_i, H_i^T R_i^{-1} z_i)
- **THEN** the fusion center SHALL produce an equivalent result to centralized fusion by summing the information contributions

#### Scenario: Information-to-covariance conversion
- **WHEN** the fused information state and matrix are computed
- **THEN** the system SHALL convert back to state estimate x_hat = Y^{-1} y_hat and covariance P = Y^{-1} for downstream consumers

### Requirement: Covariance intersection for unknown correlations
The system SHALL implement Covariance Intersection (CI) fusing two estimates (x_hat_A, P_A) and (x_hat_B, P_B) with unknown cross-correlations: P_fused^{-1} = omega * P_A^{-1} + (1 - omega) * P_B^{-1}, where omega in [0, 1] is optimized to minimize tr(P_fused) or det(P_fused).

#### Scenario: Conservative fusion guarantee
- **WHEN** two local trackers with unknown cross-correlations provide estimates for the same target
- **THEN** the fused covariance P_fused SHALL be a valid upper bound on the true MSE for any degree of unknown correlation

#### Scenario: Omega optimization
- **WHEN** CI is applied with det(P_fused) minimization criterion
- **THEN** the system SHALL find the optimal omega via a 1D line search and the resulting fused estimate SHALL have smaller determinant than either input covariance

### Requirement: Sensor registration and coordinate transforms
The system SHALL maintain sensor registration parameters (position, orientation, calibration) and transform measurements from sensor-local coordinates to a common tracking frame before fusion.

#### Scenario: Radar polar to Cartesian conversion
- **WHEN** a radar provides measurements in polar coordinates (range, azimuth, elevation)
- **THEN** the system SHALL convert to the common Cartesian tracking frame using the sensor's known position and orientation

#### Scenario: Multi-sensor alignment validation
- **WHEN** a new sensor is registered with the system
- **THEN** the system SHALL validate that the coordinate transform produces consistent positions for a known reference target across all registered sensors

