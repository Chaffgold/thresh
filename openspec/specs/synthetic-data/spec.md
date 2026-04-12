# synthetic-data Specification

## Purpose
TBD - created by archiving change transformer-fusion-tracker. Update Purpose after archive.
## Requirements
### Requirement: Configurable target trajectory generation
The system SHALL generate synthetic target trajectories with configurable dynamics: constant velocity, constant acceleration, CTRV maneuvers, ballistic arcs (with optional drag), and orbital mechanics. Trajectories SHALL be parameterized by initial state, duration, time step, and maneuver profiles.

#### Scenario: Multi-segment trajectory with maneuvers
- **WHEN** the user specifies a trajectory with 30s constant velocity, 10s coordinated turn at 3 deg/s, then 20s constant velocity
- **THEN** the system SHALL generate a smooth state history with correct kinematics at each segment and transitions between segments

#### Scenario: Ballistic trajectory generation
- **WHEN** the user specifies a ballistic target with initial velocity, launch angle, and drag coefficient
- **THEN** the system SHALL generate a trajectory following ballistic equations of motion with gravity and optional atmospheric drag

### Requirement: Synthetic radar measurement generation
The system SHALL generate radar measurements from ground-truth trajectories, producing range, azimuth, elevation, and optionally range-rate with configurable noise statistics (Gaussian, Rayleigh for range), detection probability (P_d), false alarm rate (Poisson-distributed clutter), and RCS-dependent SNR.

#### Scenario: Noisy radar detections with missed detections
- **WHEN** a target at range 50km with RCS 1 m^2 is observed by a radar with P_d = 0.9 and sigma_range = 50m
- **THEN** the generated measurements SHALL include the target detection ~90% of frames with appropriate range/angle noise, plus Poisson-distributed false alarms

#### Scenario: RCS-dependent detection probability
- **WHEN** targets with different RCS values (0.1 m^2 vs 100 m^2) are at the same range
- **THEN** the high-RCS target SHALL have higher detection probability consistent with the radar equation

### Requirement: Synthetic EO/IR measurement generation
The system SHALL generate EO/IR sensor measurements providing angular measurements (azimuth, elevation) with configurable noise, field-of-view constraints, and detection probability dependent on target IR signature and background clutter.

#### Scenario: Angle-only EO/IR observations
- **WHEN** a target is within the sensor's field of view
- **THEN** the system SHALL generate azimuth/elevation measurements with Gaussian noise at the configured angular accuracy (e.g., 0.1 mrad)

### Requirement: ADS-B message generation
The system SHALL generate synthetic ADS-B messages containing position (lat, lon, alt), velocity, and aircraft identification at 1 Hz update rate with configurable message loss probability and position quantization matching real ADS-B resolution.

#### Scenario: Cooperative target with ADS-B
- **WHEN** a cooperative aircraft target is configured with ADS-B enabled
- **THEN** the system SHALL generate 1 Hz position reports with NACp-appropriate position uncertainty and configurable message dropout rate

### Requirement: Multi-target scenario composition
The system SHALL support composing scenarios with multiple simultaneous targets, each with independent trajectories, sensor visibilities, and target types. The output SHALL be a time-ordered stream of ground-truth states and sensor measurements suitable for tracker evaluation.

#### Scenario: Dense multi-target scenario
- **WHEN** a scenario is configured with 50 simultaneous targets of mixed types (20 aerodynamic, 20 UAVs, 10 ballistic) observed by 3 radars and 2 EO/IR sensors
- **THEN** the system SHALL generate a coherent dataset with per-target ground truth and per-sensor measurement streams with correct spatial and temporal alignment

