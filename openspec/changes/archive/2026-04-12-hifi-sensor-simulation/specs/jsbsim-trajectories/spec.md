## Capability: JSBSim Flight Dynamics Trajectory Generation

### Overview
Provides a PyO3-based bridge to the JSBSim open-source flight dynamics model for generating high-fidelity 6DOF aircraft and missile trajectories. This capability enables initialization of JSBSim aircraft models (e.g., F-16, 737, generic missile), definition of waypoints or autopilot commands, execution of the 6DOF simulation, and extraction of position/velocity/attitude time histories in the thresh trajectory format. All JSBSim functionality is feature-gated behind `jsbsim` to keep the core library free of Python dependencies.

## ADDED Requirements

### Requirement: Feature-gated JSBSim module
The system MUST gate all JSBSim trajectory generation functionality behind the `jsbsim` Cargo feature flag. When the feature is not enabled, no Python, PyO3, or JSBSim dependencies MUST be compiled or required. When enabled, the module MUST link against PyO3 and require a Python environment with the JSBSim Python bindings installed.

#### Scenario: Build without jsbsim feature
- **GIVEN** the thresh crate is compiled with default features (no `jsbsim`)
- **WHEN** the build completes
- **THEN** the binary MUST compile successfully without any Python or JSBSim installation and MUST NOT expose JSBSim trajectory APIs

#### Scenario: Build with jsbsim feature
- **GIVEN** the thresh crate is compiled with `--features jsbsim` and Python 3.9+ with JSBSim bindings is available
- **WHEN** the build completes
- **THEN** the JSBSim trajectory generation API MUST be accessible and functional

### Requirement: Initialize aircraft and missile models
The system MUST support initialization of JSBSim flight dynamic models including at minimum: fighter aircraft (F-16 or equivalent), commercial transport (737 or equivalent), and generic missile models from the JSBSim aircraft library. Initialization MUST accept initial conditions including geodetic position (latitude, longitude, altitude), airspeed or Mach number, heading, and flight path angle. The system MUST validate that the requested model exists and report a descriptive error if not found.

#### Scenario: Initialize F-16 model at specified conditions
- **GIVEN** a request to initialize a JSBSim F-16 model at position (lat=35.0 N, lon=-120.0 W, alt=20000 ft), Mach 0.85, heading 090
- **WHEN** the JSBSim simulation is initialized
- **THEN** the model MUST be loaded with the correct initial state, trimmed for steady-level flight, and ready to propagate

#### Scenario: Invalid model name error
- **GIVEN** a request to initialize a JSBSim model named "nonexistent_aircraft"
- **WHEN** initialization is attempted
- **THEN** the system MUST return an error indicating the model was not found in the JSBSim aircraft directory

### Requirement: Define waypoints and autopilot commands
The system MUST accept trajectory definitions as a sequence of waypoints (latitude, longitude, altitude, optional speed constraint) or autopilot command segments (target heading, target altitude, target speed, duration). The system MUST translate waypoints into autopilot guidance commands that JSBSim's flight control system can follow. Command segments MUST support heading hold, altitude hold, speed hold, and coordinated turns.

#### Scenario: Waypoint-based flight plan
- **GIVEN** an F-16 model initialized at waypoint A and a flight plan with waypoints B, C, D each specifying position and altitude
- **WHEN** the simulation is executed
- **THEN** the aircraft MUST fly toward each waypoint in sequence, executing turns at each waypoint, and the resulting trajectory MUST pass within 1 nm of each waypoint

#### Scenario: Autopilot command segments
- **GIVEN** an initialized 737 model and a command sequence: climb to FL350 at 250 KIAS, accelerate to Mach 0.78, hold heading 270 for 300 seconds
- **WHEN** the simulation is executed
- **THEN** the aircraft MUST follow each command segment in order, achieving the target altitude within 500 ft, target speed within 10 kt, and target heading within 2 degrees by the end of each segment

### Requirement: Run 6DOF simulation and extract state history
The system MUST execute the JSBSim 6DOF simulation at a configurable time step (default 1/120 s for JSBSim internal, output sampled at user-specified rate) and extract a complete state history including: geodetic position (lat, lon, alt), ECEF or ECI position, NED or body-frame velocity, Euler angles (roll, pitch, yaw), angular rates, indicated airspeed, Mach number, angle of attack, and load factor. The output MUST be available at a user-specified sample rate (e.g., 10 Hz, 100 Hz).

#### Scenario: Extract full state history at 10 Hz
- **GIVEN** an F-16 model executing a 60-second simulation with a 3g turn
- **WHEN** the state history is extracted at 10 Hz
- **THEN** the output MUST contain 600 state samples, each with position, velocity, attitude, and the load factor during the turn MUST be approximately 3g (within 0.5g)

### Requirement: Convert to thresh trajectory format
The system MUST convert JSBSim state history output into the thresh internal trajectory format, including ECEF or ENU position vectors, velocity vectors, and timestamps. The conversion MUST handle coordinate frame transformations (geodetic to ECEF, NED to ECEF) and time base alignment. The resulting trajectory MUST be directly usable by thresh measurement generators and tracker evaluation pipelines.

#### Scenario: JSBSim to thresh trajectory conversion
- **GIVEN** a completed JSBSim simulation producing 1000 state records in geodetic/NED coordinates
- **WHEN** the output is converted to thresh trajectory format
- **THEN** the resulting trajectory MUST contain 1000 time-stamped ECEF position and velocity states, and the round-trip conversion from geodetic back to ECEF MUST be accurate to within 1 meter

### Requirement: Realistic maneuver constraints
The system MUST enforce realistic maneuver constraints through the JSBSim flight dynamics model, including g-limits (structural and physiological), maximum turn rate as a function of speed and altitude, climb/descent rate limits based on available thrust, and stall speed boundaries. Autopilot commands that exceed the aircraft's performance envelope MUST be limited by the flight dynamics model rather than producing unrealistic trajectories.

#### Scenario: G-limit enforcement during turn
- **GIVEN** an F-16 model commanded to execute a turn requiring 12g at current speed and altitude
- **WHEN** the simulation is executed
- **THEN** the actual load factor MUST be limited by the flight control system to approximately 9g (F-16 structural limit), and the turn radius MUST be larger than the geometrically minimum radius at 12g

#### Scenario: Altitude ceiling constraint
- **GIVEN** a 737 model at FL350 commanded to climb to FL600 (above service ceiling)
- **WHEN** the simulation is executed
- **THEN** the aircraft MUST climb until thrust can no longer sustain climb rate, leveling off at or below the aircraft's practical ceiling (approximately FL410-FL430 for a 737), and MUST NOT reach the commanded FL600

## Reference Information

### JSBSim Aircraft Models
| Model | Type | Typical Use |
|-------|------|-------------|
| F-16 | Fighter | Air-to-air, maneuvering targets |
| 737 | Transport | Commercial air traffic |
| A-4 | Attack | Medium-performance maneuvering |
| Generic missile | Missile | Ballistic and cruise missile trajectories |

### Key References
- Berndt, J. "JSBSim: An Open Source Flight Dynamics Model in C++," AIAA-2004-4923
- JSBSim GitHub: https://github.com/JSBSim-Team/jsbsim
- Stevens, B.L. and Lewis, F.L. "Aircraft Control and Simulation," Wiley, 2003
