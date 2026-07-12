# synthetic-data Specification (Delta)

## MODIFIED Requirements

### Requirement: Configurable target trajectory generation
The system SHALL generate synthetic target trajectories with configurable dynamics: constant velocity, constant acceleration, CTRV maneuvers, phased ballistic trajectories, and orbital mechanics. Trajectories SHALL be parameterized by initial state, duration, time step, and maneuver profiles. Ballistic trajectories SHALL be generated in three phases over a round rotating Earth: boost (constant thrust with a gravity turn), midcourse (exoatmospheric propagation using the same two-body + J2 propagator as orbital truth generation), and reentry (atmospheric drag parameterized by ballistic coefficient beta with altitude-dependent exponential density). The flat-Earth ballistic segment parameterized by a bare drag coefficient is replaced by this phased model.

#### Scenario: Multi-segment trajectory with maneuvers
- **WHEN** the user specifies a trajectory with 30s constant velocity, 10s coordinated turn at 3 deg/s, then 20s constant velocity
- **THEN** the system SHALL generate a smooth state history with correct kinematics at each segment and transitions between segments

#### Scenario: Phased ballistic trajectory generation
- **WHEN** the user specifies a ballistic target with boost parameters (thrust profile, burn time, gravity-turn pitch-over), a midcourse span, and a reentry ballistic coefficient
- **THEN** the system SHALL generate a single continuous trajectory whose boost phase accelerates under thrust plus gravity, whose midcourse phase follows the two-body + J2 propagator with negligible drag, and whose reentry phase decelerates under beta-parameterized drag, with position and velocity continuous at every phase boundary

#### Scenario: Round rotating Earth
- **WHEN** a long-range ballistic trajectory is generated
- **THEN** the trajectory SHALL account for Earth curvature and rotation: the ground track SHALL follow the round Earth (not a flat-Earth tangent plane), and the impact point SHALL reflect Earth rotation during the flight time

#### Scenario: Midcourse consistency with the orbital propagator
- **WHEN** a midcourse segment's initial state is propagated independently through the existing J2 RK4 orbital propagator
- **THEN** the trajectory generator's midcourse states SHALL match that propagation, because both use the same shared force math
