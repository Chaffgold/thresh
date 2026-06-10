# automotive-synthetic-scenarios Specification

## Purpose
Define synthetic automotive trajectory generation in `thresh-synth`: the
2-DOF kinematic-bicycle segment (`ω = v·tan(δ)/L`, speed clamped at
standstill) and deterministic road-scenario presets whose agents stay within
per-class physical speed and turn-rate bounds.

## Requirements
### Requirement: Kinematic vehicle motion model

`thresh-synth` MUST provide a kinematic bicycle trajectory segment (steering
angle, acceleration, wheelbase) alongside the existing `Cv`/`Ca`/`Ctrv`/
`Ballistic` segments.

#### Scenario: Bicycle-model segment produces a steered path

**WHEN** a trajectory is generated from a kinematic-bicycle segment with a
non-zero steering angle and a finite wheelbase

**THEN** the resulting path curves consistently with bicycle kinematics
(heading changes at a rate set by speed, steering angle, and wheelbase).

#### Scenario: Existing segment types are unchanged

**WHEN** a trajectory is generated from a `Cv`, `Ca`, `Ctrv`, or `Ballistic`
segment

**THEN** its output is unchanged by this addition.

### Requirement: Road-scenario presets

`thresh-synth` MUST offer automotive scenario presets (e.g. lane-follow,
intersection, multi-agent) that yield physically plausible per-class motion.

#### Scenario: Generated scenarios respect per-class limits

**WHEN** an automotive scenario preset is generated

**THEN** each agent's speed and turn rate stay within plausible bounds for its
class (e.g. pedestrians are slow, cars do not exceed road speeds).

