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
The system SHALL support class-specific tracking configurations where different target classes (e.g., aerodynamic, ballistic, orbital) use different motion models, process noise parameters, and track management policies.

#### Scenario: Heterogeneous target class tracking
- **WHEN** detections are classified as either "aerodynamic" (using CTRV model) or "ballistic" (using ballistic trajectory model)
- **THEN** each track SHALL use the motion model and noise parameters appropriate to its classified target type

#### Scenario: Class reclassification
- **WHEN** a track's classification confidence changes (e.g., initially classified as aerodynamic, later reclassified as ballistic after observing trajectory)
- **THEN** the track SHALL switch to the appropriate motion model with state vector adaptation

### Requirement: Track identity management
The system SHALL assign globally unique track IDs and maintain identity through occlusions, sensor gaps, and re-associations. Track IDs SHALL never be reused within a session.

#### Scenario: Identity preservation through occlusion
- **WHEN** a confirmed track coasts for 3 frames and then re-associates with a detection
- **THEN** the track SHALL retain its original ID

#### Scenario: Unique ID guarantee
- **WHEN** 10,000 tracks are created and deleted over a session
- **THEN** no two tracks SHALL ever share the same ID

