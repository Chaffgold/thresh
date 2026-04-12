## Capability: ADS-B Data Ingestion

### Overview
Ingest real ADS-B surveillance data from OpenSky Network and ADS-B Exchange, converting to thresh measurement and ground truth formats. Supports two data sources: OpenSky REST API (JSON state vectors, flight tracks) and ADS-B Exchange SBS BaseStation format (CSV, 1 Hz native). Download caching and API rate limiting are built in.

## ADDED Requirements

### Requirement: Download ADS-B data
Fetch historical flight data by time range, bounding box, and optional ICAO24 filter. The system MUST support this.

#### Scenario: Fetch single flight track
- **WHEN** an ICAO24 identifier and time range are provided
- **THEN** the download function SHALL return a time-ordered sequence of ADS-B position reports

### Requirement: Parse ADS-B formats
Parse SBS BaseStation format and OpenSky JSON state vectors. The system MUST support this.

#### Scenario: Parse SBS BaseStation messages
- **WHEN** a file containing raw SBS BaseStation messages is processed
- **THEN** the parser SHALL extract position (MSG type 3) and velocity (MSG type 4) records with correct field mapping

### Requirement: Convert to Measurement type
Map parsed ADS-B data to `Measurement::AdsB { lat, lon, alt, velocity, time }`. The system MUST support this.

#### Scenario: Convert OpenSky state vectors to measurements
- **WHEN** a set of parsed OpenSky JSON state vectors is converted
- **THEN** each state vector SHALL become a `Measurement::AdsB` with lat, lon, alt, velocity, and time fields populated

### Requirement: Extract ground truth trajectories
Interpolate ADS-B positions to a regular time grid for ground truth trajectories (ADS-B is itself the "truth" for cooperative targets). The system MUST support this.

#### Scenario: Interpolate positions to regular time grid
- **WHEN** irregularly-spaced ADS-B reports for a single ICAO24 are processed for ground truth
- **THEN** the extractor SHALL produce a uniformly-sampled trajectory with interpolated positions and velocities

### Requirement: Coordinate transform
Convert WGS84 lat/lon/alt to ENU (East-North-Up) local frame relative to a configurable reference point. The system MUST support this.

#### Scenario: Transform WGS84 to ENU coordinates
- **WHEN** ADS-B positions in WGS84 lat/lon/alt and a reference point are provided
- **THEN** positions SHALL be expressed in ENU metres relative to the reference point

### Requirement: Cache downloaded data
Download data once, then cache locally in a standard directory (~/.thresh/data/). The system MUST support this.

#### Scenario: Retrieve previously downloaded data from cache
- **WHEN** a flight data query that has been fetched before is requested again
- **THEN** the data SHALL be loaded from the local cache without making a network request

### Requirement: Rate limiting
Respect OpenSky API rate limits with backoff. The system MUST support this.

#### Scenario: Back off when rate limited
- **WHEN** a sequence of API requests exceeds the OpenSky free-tier rate limit and a rate-limit response is received
- **THEN** the client SHALL wait with exponential backoff before retrying
