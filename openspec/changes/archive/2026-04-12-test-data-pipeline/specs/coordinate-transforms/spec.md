## Capability: Coordinate Transforms

### Overview
Implement geodetic, Earth-centered, and local coordinate frame transformations in `thresh-core` to support multi-sensor fusion across aerospace domains. Covers WGS84 geodetic, ECEF, ECI (J2000/GCRF), ENU, and TEME coordinate systems with validated transforms between them.

## ADDED Requirements

### Requirement: WGS84 geodetic to ECEF
Convert geodetic coordinates (latitude, longitude, altitude) to Earth-Centered Earth-Fixed (ECEF) Cartesian coordinates using WGS84 ellipsoid parameters. The system SHALL implement this.

#### Scenario: Known reference point conversion
- **WHEN** given JFK airport coordinates (40.6413° N, 73.7781° W, 13 m altitude)
- **THEN** the ECEF output SHALL match published ECEF coordinates within 1 m

#### Scenario: Pole and equator edge cases
- **WHEN** given coordinates at the North Pole (90° N, 0° E, 0 m) and on the equator (0° N, 0° E, 0 m)
- **THEN** the ECEF output SHALL match analytical WGS84 values (equatorial radius at equator, polar radius at pole)

### Requirement: ECEF to ENU
Convert ECEF coordinates to East-North-Up (ENU) local tangent plane coordinates relative to a configurable reference point. The system SHALL implement this.

#### Scenario: ENU at reference point origin
- **WHEN** the target position equals the reference point
- **THEN** the ENU output SHALL be (0, 0, 0)

#### Scenario: ENU roundtrip
- **WHEN** a WGS84 position is converted to ENU via ECEF and then back to WGS84
- **THEN** the roundtrip error SHALL be less than 1 cm

### Requirement: WGS84 to ENU convenience
Compose WGS84→ECEF and ECEF→ENU into a single convenience function. The system SHALL implement this.

#### Scenario: Direct vs composed conversion
- **WHEN** a WGS84 position is converted using the convenience function
- **THEN** the result SHALL be identical to manually chaining WGS84→ECEF→ENU

### Requirement: TEME to ECEF
Convert True Equator Mean Equinox (TEME) coordinates to ECEF using Greenwich Mean Sidereal Time (GMST) Earth rotation. Required for SGP4 propagator output. The system SHALL implement this.

#### Scenario: TEME to ECEF at known epoch
- **WHEN** given a TEME state vector at a known epoch
- **THEN** the ECEF output SHALL match reference implementations (e.g., Vallado's test cases) within 1 km

### Requirement: ECI (J2000/GCRF) coordinate frame
Define an ECI coordinate frame type and implement conversions to/from ECEF. ECI is inertial (non-rotating), required for orbital mechanics and high-fidelity propagation. The system SHALL implement this.

#### Scenario: ECI frame definition
- **WHEN** an ECI position and velocity are specified
- **THEN** the frame SHALL represent J2000/GCRF with X toward vernal equinox, Z toward celestial north pole, Y completing the right-hand system

### Requirement: ECI to ECEF transform
Convert ECI (J2000/GCRF) coordinates to ECEF using Earth Rotation Angle (ERA) or GMST. The system SHALL implement this.

#### Scenario: ECI to ECEF at J2000 epoch
- **WHEN** given an ECI position at J2000.0 epoch (2000-01-01T12:00:00 TT)
- **THEN** the ECEF output SHALL account for Earth rotation angle at that epoch, matching IAU reference values within 1 km

#### Scenario: Rotation rate consistency
- **WHEN** the same ECI position is converted to ECEF at two epochs 86164.1 seconds apart (one sidereal day)
- **THEN** the ECEF positions SHALL be approximately equal (within numerical precision of Earth rotation model)

### Requirement: ECEF to ECI inverse transform
Convert ECEF coordinates back to ECI (J2000/GCRF). The system SHALL implement this.

#### Scenario: ECI-ECEF roundtrip
- **WHEN** an ECI position is converted to ECEF and back to ECI at the same epoch
- **THEN** the roundtrip error SHALL be less than 1 mm

### Requirement: ECI to ENU convenience
Compose ECI→ECEF and ECEF→ENU into a single function for ground station observation modeling. The system SHALL implement this.

#### Scenario: ISS pass observation
- **WHEN** given ISS ECI position during a known pass over a ground station
- **THEN** the ENU output SHALL produce positive Up component when above the horizon and azimuth/elevation SHALL match published pass predictions within 0.1°
