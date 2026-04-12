# othr-tracker-integration Specification

## Purpose
TBD - created by archiving change othr-sensor-model. Update Purpose after archive.
## Requirements
### Requirement: OTHR observation model
The system SHALL implement an observation matrix mapping the track state to OTHR observables (ground range, azimuth, Doppler). The system MUST support this.

#### Scenario: OTHR observation of CV target
- **WHEN** a track with state [x, vx, y, vy, z, vz] is observed by OTHR
- **THEN** the observation model SHALL produce predicted ground range (great-circle distance), azimuth (bearing from transmitter), and Doppler (radial velocity component)

### Requirement: OTHR observation Jacobian
The system SHALL implement the Jacobian of the OTHR observation function for use with EKF. The system MUST support this.

#### Scenario: EKF update with OTHR measurement
- **WHEN** an EKF track receives an OTHR measurement
- **THEN** the Jacobian SHALL correctly linearize the nonlinear ground-range/azimuth mapping, and the updated state covariance SHALL decrease in the horizontal plane while vertical uncertainty remains large

### Requirement: OTHR-specific gating
The system SHALL implement Mahalanobis gating that accounts for OTHR's coarse resolution. The system MUST support this.

#### Scenario: Gating with large measurement noise
- **WHEN** OTHR measurements have 20 km range sigma and 1° azimuth sigma
- **THEN** the gate threshold SHALL be set appropriately to avoid rejecting valid associations while limiting false matches

### Requirement: OTHR + conventional radar fusion
The system SHALL support fusing OTHR measurements with conventional radar measurements on the same track. The system MUST support this.

#### Scenario: OTHR cueing followed by radar refinement
- **WHEN** an OTHR measurement initiates a track with large position uncertainty (20 km)
- **THEN** a subsequent conventional radar measurement (50 m accuracy) SHALL dramatically reduce the track position uncertainty

#### Scenario: Position accuracy improvement
- **WHEN** OTHR and conventional radar measurements are fused on the same target
- **THEN** the fused position error SHALL be less than either sensor alone

