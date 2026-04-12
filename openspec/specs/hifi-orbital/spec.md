# hifi-orbital Specification

## Purpose
TBD - created by archiving change hifi-sensor-simulation. Update Purpose after archive.
## Requirements
### Requirement: Numerical propagation with nyx-space
The system MUST integrate the nyx-space Rust crate as the primary high-fidelity orbital propagator. The propagator MUST use a numerical integrator (e.g., Dormand-Prince RK7(8) or similar adaptive-step method) to propagate spacecraft state vectors in the ECI J2000 frame. The integration MUST support configurable absolute and relative tolerances (default 1e-12) and automatic step-size control.

#### Scenario: Propagate LEO satellite for one orbit
- **GIVEN** a spacecraft in a 400 km circular orbit (ISS-like) with initial state vector in ECI J2000
- **WHEN** the state is propagated for one orbital period (~92 minutes) using nyx-space with default tolerances
- **THEN** the propagated position MUST be accurate to within 10 meters compared to a high-precision reference ephemeris, and the energy error over one orbit MUST be less than 1e-10 relative

### Requirement: Configurable force models
The system MUST support the following configurable force models for orbital propagation: Earth gravity field up to degree and order 70 using EGM96 or JGM3 coefficients, atmospheric drag using NRLMSISE-00 (or equivalent) density model with configurable drag coefficient (Cd) and area-to-mass ratio, solar radiation pressure with configurable reflectivity coefficient (Cr) and area-to-mass ratio, and third-body gravitational perturbations from the Sun and Moon using JPL DE ephemerides. Each force model MUST be individually toggleable.

#### Scenario: J2-only vs full force model comparison
- **GIVEN** a spacecraft at 500 km altitude with area-to-mass ratio 0.01 m^2/kg
- **WHEN** the orbit is propagated for 24 hours using (a) J2-only gravity and (b) full force model (70x70 gravity + drag + SRP + Sun/Moon)
- **THEN** the position difference between the two propagations MUST be at least 100 meters after 24 hours, demonstrating the impact of higher-order perturbations

#### Scenario: Atmospheric drag effect on LEO orbit
- **GIVEN** a satellite in a 300 km circular orbit with Cd=2.2 and area-to-mass ratio 0.02 m^2/kg
- **WHEN** the orbit is propagated for 7 days with NRLMSISE-00 drag enabled
- **THEN** the semi-major axis MUST decrease due to drag (orbit decay), and the altitude drop MUST be between 0.5 km and 5 km depending on solar activity level

### Requirement: Higher accuracy than SGP4
The system MUST provide orbital propagation accuracy significantly exceeding SGP4/SDP4. For LEO objects, the position error over a 24-hour propagation MUST be less than 100 meters (compared to SGP4 errors of 1-10 km at 24 hours). For GEO objects, the position error over 7 days MUST be less than 1 km. Accuracy MUST be validated against reference ephemerides or precision orbit determination solutions.

#### Scenario: LEO accuracy vs SGP4 over 24 hours
- **GIVEN** a LEO satellite with a known precision ephemeris and corresponding TLE
- **WHEN** both SGP4 (from TLE) and nyx-space (from precise initial state with full force model) propagate for 24 hours
- **THEN** the nyx-space position error MUST be at least 10x smaller than the SGP4 position error relative to the precision ephemeris

### Requirement: Maneuver modeling
The system MUST support modeling orbital maneuvers including impulsive delta-V burns (instantaneous velocity change at a specified epoch) and finite-duration burns (constant or variable thrust over a time interval with configurable thrust magnitude, specific impulse, and direction in RTN or body frame). Maneuvers MUST correctly update the spacecraft mass when fuel consumption is modeled.

#### Scenario: Hohmann transfer via impulsive burns
- **GIVEN** a spacecraft in a 400 km circular orbit
- **WHEN** two impulsive burns are applied to execute a Hohmann transfer to 800 km altitude (first burn at perigee to raise apogee, second burn at apogee to circularize)
- **THEN** the final orbit MUST be circular at 800 km altitude within 1 km, and the total delta-V MUST match the analytical Hohmann transfer delta-V within 1 m/s

#### Scenario: Finite burn orbit raise
- **GIVEN** a spacecraft with mass 500 kg, thrust 50 N, and Isp 300 s in a 400 km orbit
- **WHEN** a finite along-track burn of 120 seconds is executed
- **THEN** the resulting orbit MUST have a higher apogee than the initial orbit, the spacecraft mass MUST decrease by the consumed propellant mass (dm = thrust * dt / (Isp * g0)), and the orbit change MUST be consistent with the applied impulse

### Requirement: Covariance propagation for uncertainty quantification
The system MUST support propagation of the state covariance matrix alongside the state vector for uncertainty quantification. Covariance propagation MUST use either a linearized state transition matrix (STM) approach or an unscented transform. The covariance MUST account for force model uncertainties (drag coefficient uncertainty, atmospheric density uncertainty) and initial state uncertainty. The output MUST provide position and velocity uncertainty (1-sigma) at each propagation epoch.

#### Scenario: Covariance growth over one orbit
- **GIVEN** a LEO satellite with initial position uncertainty of 10 m (1-sigma) and velocity uncertainty of 0.01 m/s (1-sigma)
- **WHEN** the state and covariance are propagated for one orbital period
- **THEN** the position uncertainty MUST grow (primarily in the along-track direction due to velocity uncertainty) and the along-track 1-sigma MUST be approximately 5-10 meters larger than the initial position uncertainty

### Requirement: Orekit fallback via PyO3
The system MUST provide a fallback orbital propagation path using Orekit via a PyO3 bridge, feature-gated behind `hifi-orbital`. This fallback MUST support the same force models and maneuver types as the nyx-space primary path. The Orekit bridge MUST be used only when nyx-space lacks a needed capability (e.g., specific atmospheric model or gravity field). The API MUST present a unified interface so callers do not need to know which backend is active.

#### Scenario: Unified API with nyx-space and Orekit backends
- **GIVEN** identical initial conditions and force model configuration
- **WHEN** a 24-hour propagation is executed using both the nyx-space backend and the Orekit backend
- **THEN** the position results from both backends MUST agree within 100 meters, confirming the unified API produces consistent results regardless of backend

