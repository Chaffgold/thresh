# track-management Specification

## Purpose
TBD - created by archiving change transformer-fusion-tracker. Update Purpose after archive.
## Requirements
### Requirement: Track lifecycle state machine
The system SHALL manage tracks through a lifecycle: Tentative → Confirmed → Coasting → Deleted. Transitions SHALL be governed by configurable policies (M-of-N for confirmation, max-coast-age for deletion).

#### Scenario: M-of-N track confirmation
- **WHEN** a new tentative track receives M successful associations out of N consecutive frames (e.g., 3-of-5)
- **THEN** the track SHALL transition to Confirmed state and be reported as an active track

#### Scenario: Track coasting on missed detection
- **WHEN** a confirmed track receives no association for a single frame
- **THEN** the track SHALL transition to Coasting state and continue state propagation via prediction only (no measurement update)

#### Scenario: Track deletion after max coast age
- **WHEN** a coasting track has not received an association for max_coast_frames consecutive frames (configurable, default 5)
- **THEN** the track SHALL be deleted and its ID retired

### Requirement: Track birth from unassigned detections
The system SHALL create new tentative tracks from detections that were not assigned to any existing track during association. Each new track SHALL be initialized with state derived from the detection (position, velocity if available) and default covariance.

#### Scenario: Single detection initialization
- **WHEN** a single unassigned detection with position [x, y, z] is received
- **THEN** a new tentative track SHALL be created with state [x, 0, y, 0, z, 0] (zero velocity) and configurable initial covariance

#### Scenario: Multi-sensor corroborated initialization
- **WHEN** unassigned detections from multiple sensors fall within a spatial gate of each other within the same frame
- **THEN** the system SHALL create a single track initialized from the fused measurement rather than multiple redundant tracks

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

### Requirement: Track identity management
The system SHALL assign globally unique track IDs and maintain identity through occlusions, sensor gaps, and re-associations. Track IDs SHALL never be reused within a session.

#### Scenario: Identity preservation through occlusion
- **WHEN** a confirmed track coasts for 3 frames and then re-associates with a detection
- **THEN** the track SHALL retain its original ID

#### Scenario: Unique ID guarantee
- **WHEN** 10,000 tracks are created and deleted over a session
- **THEN** no two tracks SHALL ever share the same ID

