# othr-measurement-type Specification

## Purpose
TBD - created by archiving change othr-sensor-model. Update Purpose after archive.
## Requirements
### Requirement: OTHR measurement variant
The system SHALL support an `Othr` variant in the `Measurement` enum with ground range, azimuth, Doppler velocity, and propagation mode. The system MUST support this.

#### Scenario: OTHR detection of distant aircraft
- **WHEN** an OTHR system detects an aircraft at 2000 km ground range, 045° azimuth, with 250 m/s radial velocity via F-layer single hop
- **THEN** the measurement SHALL contain ground_range_m=2000000, azimuth_rad≈0.785, doppler_m_s=250.0, propagation_mode=FLayer, and valid time/sensor_id

### Requirement: Propagation mode metadata
Each OTHR measurement SHALL carry metadata indicating the ionospheric propagation path (E-layer, F-layer, or multi-hop with hop count). The system MUST support this.

#### Scenario: Multi-hop detection at extended range
- **WHEN** a target is detected at 5000 km via 2-hop F-layer propagation
- **THEN** the propagation_mode SHALL be MultiHop(2) and the ground range SHALL reflect the full surface distance

### Requirement: Serialization compatibility
The OTHR measurement type SHALL serialize and deserialize via serde consistently with existing measurement variants. The system MUST support this.

#### Scenario: JSON roundtrip
- **WHEN** an OTHR measurement is serialized to JSON and deserialized back
- **THEN** all fields SHALL be preserved exactly

