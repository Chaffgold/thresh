## Capability: Orbital Data Ingestion

### Overview
Ingest Two-Line Element (TLE) data from space-track.org, propagate orbits via SGP4, and generate synthetic sensor measurements from orbital positions for space situational awareness tracking validation.

### Requirement: Data Sources

**space-track.org** The system MUST support this.
- Requires free account (username/password)
- REST API with class-based queries
- TLE/3LE format for 40,000+ cataloged objects
- GP (General Perturbations) data in JSON/XML
- Historical TLEs available for conjunction analysis
- Update frequency: multiple times daily for active objects

**CelesTrak** (backup/supplementary)
- Public, no auth required
- Curated TLE sets by category (active satellites, debris, stations)
- Supplemental GP data

## ADDED Requirements

### Requirement: Download TLE data
Fetch TLEs by NORAD catalog number, time range, or object class (debris, payload, rocket body). The system MUST support this.

#### Scenario: Fetch TLEs for a specific satellite
- Given a NORAD catalog number and a time range
- When the download function is called
- Then it returns the matching TLE records for that object within the time range

### Requirement: Parse TLE formats
Parse TLE two-line format, 3LE three-line format, and GP JSON format. The system MUST support this.

#### Scenario: Parse standard two-line element set
- Given a raw TLE string in two-line format
- When the parser processes it
- Then it extracts epoch, inclination, eccentricity, mean motion, and all other orbital elements

### Requirement: Propagate orbits with SGP4
SGP4/SDP4 orbit propagation to generate position/velocity at arbitrary times. The system MUST support this.

#### Scenario: Propagate ISS orbit forward in time
- Given a TLE for the ISS and a sequence of future timestamps
- When SGP4 propagation is performed
- Then it produces position and velocity vectors at each requested time

### Requirement: Coordinate system transforms
TEME to ECEF to ENU transforms, handling Earth rotation. The system MUST support this.

#### Scenario: Convert TEME state vector to ENU
- Given a satellite state vector in TEME frame and a ground station location
- When coordinate transforms are applied
- Then the position is expressed in ENU coordinates relative to the ground station

### Requirement: Generate ground truth from propagation
SGP4-propagated state vectors serve as ground truth (accuracy ~1 km for well-tracked objects). The system MUST support this.

#### Scenario: Produce ground truth trajectory for a cataloged object
- Given a TLE and a time span
- When ground truth generation is invoked
- Then it returns a time-ordered sequence of state vectors with NORAD ID as target identifier

### Requirement: Generate synthetic radar measurements
Generate radar observations (range/azimuth/elevation) from orbital positions as seen from configurable ground stations. The system MUST support this.

#### Scenario: Simulate radar detections from a ground station
- Given an orbital trajectory and a ground station location
- When synthetic measurement generation is invoked
- Then it produces range, azimuth, and elevation measurements for visible passes with realistic noise

### Requirement: Find conjunction scenarios
Find close approaches between objects for multi-target tracking stress tests. The system MUST support this.

#### Scenario: Detect close approach between two objects
- Given TLEs for two objects over a time window
- When conjunction analysis is performed
- Then it identifies the time of closest approach and the minimum separation distance

### Requirement: Cache TLE database
TLE database cached locally, refreshed on demand. The system MUST support this.

#### Scenario: Serve TLEs from local cache
- Given a TLE query that has been fetched previously
- When the same query is made again
- Then the data is returned from the local cache without a network request

### Output Format
- `Vec<GroundTruth>` with NORAD ID as target ID, ECI or ENU position/velocity
- `Vec<Measurement::Radar>` synthetic radar measurements from ground station perspectives
- Pass predictions: rise/set times, max elevation for each station-object pair

### Test Scenarios
- ISS tracking from a single ground station (well-known orbit, fast-moving LEO)
- GEO satellite cluster (close proximity, slow apparent motion)
- Starlink train (many objects, similar orbits, tests association at scale)
- Debris conjunction event (converging tracks, tests prediction accuracy)
