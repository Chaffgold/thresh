# othr-coordinate-registration Specification

## Purpose
TBD - created by archiving change othr-sensor-model. Update Purpose after archive.
## Requirements
### Requirement: Vincenty's direct formula
The system SHALL implement Vincenty's direct geodetic formula to compute the endpoint (lat, lon) given a start point, azimuth, and geodesic distance. The system MUST support this.

#### Scenario: Known great-circle endpoint
- **WHEN** starting from (0°N, 0°E) with azimuth 45° and distance 1000 km
- **THEN** the computed endpoint SHALL match published geodetic values within 1 m

### Requirement: Vincenty's inverse formula
The system SHALL implement Vincenty's inverse formula to compute geodesic distance and azimuth between two points. The system MUST support this.

#### Scenario: Direct-inverse roundtrip
- **WHEN** the direct formula computes an endpoint, and the inverse formula computes distance/azimuth back
- **THEN** the roundtrip error SHALL be less than 1 cm

### Requirement: OTHR to ENU conversion
The system SHALL convert OTHR ground-range/azimuth measurements to ENU coordinates via the geodetic→ECEF→ENU chain. The system MUST support this.

#### Scenario: OTHR detection at known range
- **WHEN** an OTHR at (40°N, 74°W) detects a target at 2000 km ground range, azimuth 045°
- **THEN** the ENU coordinates SHALL place the target approximately 1414 km east and 1414 km north (cos/sin 45°) with altitude estimated from ionospheric geometry

### Requirement: Altitude estimation
The system SHALL estimate target altitude from the ionospheric propagation geometry when elevation is unobservable. The system MUST support this.

#### Scenario: Mid-path altitude estimate
- **WHEN** a target is detected via F-layer single hop at 2000 km range
- **THEN** the estimated altitude SHALL be derived from the propagation geometry (tangent height at mid-path)

