## Capability: Automotive Sensor Measurements

### Overview

Measurement types and an ego-motion input for automotive tracking. Adds LiDAR
and camera measurements alongside the existing `Radar`/`EoIr`/`AdsB`/`Othr`
variants, and an ego-motion channel so the tracker can compensate for the
moving sensor platform — the defining difference from aerospace tracking.

## ADDED Requirements

### Requirement: LiDAR and camera measurements

The measurement type MUST support LiDAR 3D detections and camera detections,
each tagged with a sensor id and timestamp for multi-sensor fusion.

#### Scenario: LiDAR 3D detection is represented

**WHEN** a LiDAR detection is produced (3D position, 3D box extent, yaw, class)

**THEN** it is expressible as a LiDAR measurement carrying those fields plus a
sensor id and timestamp.

#### Scenario: Camera detection is represented

**WHEN** a camera detection is produced

**THEN** it is expressible as a camera measurement carrying its geometry, a
sensor id, and a timestamp.

#### Scenario: Existing measurement variants are unchanged

**WHEN** existing aerospace code constructs `Radar`, `EoIr`, `AdsB`, or `Othr`
measurements

**THEN** those variants are unchanged by this addition.

### Requirement: Ego-motion input to the tracker

The tracker MUST accept an ego-motion input (ego pose plus linear and angular
velocity) and use it to compensate the prediction step for sensor-platform
motion, without affecting trackers that do not supply ego-motion.

#### Scenario: Ego-motion compensates prediction

**WHEN** the automotive tracker is stepped with an ego-motion sample describing
the platform's motion over the interval

**THEN** track prediction accounts for that motion so stationary world objects
do not drift in the world frame purely due to ego movement.

#### Scenario: Aerospace trackers are unaffected

**WHEN** a tracker is constructed without ego-motion (the existing aerospace
constructors)

**THEN** its behaviour is identical to before this change.
