# automotive-sensor-measurements Specification

## Purpose
Define the automotive sensor measurement types (`Measurement::Lidar`,
`Measurement::Camera`) and the ego-motion contract (`EgoPose`/`EgoMotion`,
ego→world ENU, `[w,x,y,z]` quaternions): the tracker compensates for
sensor-platform motion at the measurement boundary so stationary world
objects do not drift, while aerospace entry points remain unchanged.

## Requirements
### Requirement: LiDAR and camera measurements

The measurement type MUST support LiDAR 3D detections and camera detections,
each tagged with a sensor id and timestamp for multi-sensor fusion.

#### Scenario: LiDAR 3D detection is represented

**WHEN** a LiDAR detection is produced (3D position, 3D box extent, yaw, class)

**THEN** it is expressible as a LiDAR measurement carrying those fields plus a
sensor id and timestamp.

#### Scenario: Camera detection is represented

**WHEN** a camera detection is produced

**THEN** it is expressible as a camera measurement carrying, at minimum, an
image-plane 2D bounding box (or center + extent in pixels) and, when available,
a monocular 3D estimate (position + yaw), plus a sensor id and a timestamp.

#### Scenario: Existing measurement variants are unchanged

**WHEN** existing aerospace code constructs `Radar`, `EoIr`, `AdsB`, or `Othr`
measurements

**THEN** those variants are unchanged by this addition.

### Requirement: Ego-motion input to the tracker

The tracker MUST accept an ego-motion input — ego pose (position in metres and
orientation) and linear (m/s) and angular (rad/s) velocity, in a defined frame
(ego/body relative to a world ENU frame) — and use it to compensate the
prediction step for sensor-platform motion, without affecting trackers that do
not supply ego-motion.

#### Scenario: Ego-motion compensates prediction

**WHEN** the automotive tracker is stepped with an ego-motion sample describing
the platform's motion over the interval

**THEN** track prediction accounts for that motion so stationary world objects
do not drift in the world frame purely due to ego movement.

#### Scenario: Aerospace trackers are unaffected

**WHEN** a tracker is constructed without ego-motion (the existing aerospace
constructors)

**THEN** its behaviour is identical to before this change.

