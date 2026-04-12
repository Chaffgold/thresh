## Capability: OTHR Synthetic Data Generation

### Overview
Generate realistic synthetic OTHR measurements including ionospheric propagation effects, skip zones, Doppler-based detection, and multi-path returns.

## ADDED Requirements

### Requirement: Synthetic OTHR measurement generation
The system SHALL generate OTHR measurements from target waypoints through the full ionospheric propagation path. The system MUST support this.

#### Scenario: Target within OTHR coverage
- **WHEN** a target at 2000 km from the transmitter is within the operating frequency's coverage (above skip zone, below max range)
- **THEN** the system SHALL produce an Othr measurement with ground range, azimuth, and Doppler, each with configurable noise

#### Scenario: Target within skip zone
- **WHEN** a target is closer than the skip zone minimum range for the operating frequency
- **THEN** the system SHALL NOT produce a measurement (detection is physically impossible)

### Requirement: Doppler-based detection
The system SHALL model detection probability as a function of target Doppler separation from ground/sea clutter. The system MUST support this.

#### Scenario: Target with high radial velocity
- **WHEN** a target has significant radial velocity (>50 m/s) separating it from clutter
- **THEN** the detection probability SHALL be higher than for a target with near-zero radial velocity

### Requirement: Multi-path returns
The system SHALL optionally generate both E-layer and F-layer returns for the same target when both propagation paths are viable. The system MUST support this.

#### Scenario: Dual-path detection
- **WHEN** both E-layer and F-layer paths are available for a target
- **THEN** the system SHALL produce two measurements with different ground ranges and propagation modes

### Requirement: Preset OTHR configurations
The system SHALL provide preset configurations for representative OTHR systems. The system MUST support this.

#### Scenario: ROTHR-class preset
- **WHEN** the ROTHR-class preset is selected
- **THEN** the configuration SHALL reflect 5-28 MHz operating range, ~1000-3500 km coverage, and appropriate resolution parameters
