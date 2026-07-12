# track-management Specification (Delta)

## MODIFIED Requirements

### Requirement: Class-specific track heads
The system SHALL support class-specific tracking configurations where different target classes (e.g., aerodynamic, ballistic, orbital) use different motion models, process noise parameters, and track management policies. The ballistic head SHALL use the physics-based 7D ballistic reentry model (position, velocity, ballistic coefficient) rather than a generic constant-acceleration model, and a dedicated orbital head SHALL exist that uses the Kepler+J2 orbital motion model on a 6D ECI Cartesian state.

#### Scenario: Heterogeneous target class tracking
- **WHEN** detections are classified as either "aerodynamic" (using CTRV model) or "ballistic" (using the ballistic reentry model)
- **THEN** each track SHALL use the motion model and noise parameters appropriate to its classified target type

#### Scenario: Class reclassification
- **WHEN** a track's classification confidence changes (e.g., initially classified as aerodynamic, later reclassified as ballistic after observing trajectory)
- **THEN** the track SHALL switch to the appropriate motion model with state vector adaptation

#### Scenario: Ballistic head uses the reentry model
- **WHEN** a track is created for a target classified as ballistic
- **THEN** its filter SHALL run the 7D ballistic reentry motion model, and the track's state SHALL expose the estimated ballistic coefficient alongside position and velocity

#### Scenario: Orbital head tracks a satellite
- **WHEN** a track is created for a target classified as orbital
- **THEN** its filter SHALL run the Kepler+J2 orbital motion model, and prediction between measurement gaps SHALL follow the orbit (curving with gravity) rather than a straight-line or constant-acceleration extrapolation
