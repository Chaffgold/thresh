## Capability: ADS-B Data Ingestion

### Overview
Ingest real ADS-B surveillance data from OpenSky Network and ADS-B Exchange, converting to thresh measurement and ground truth formats.

### Requirement: Data Sources

**OpenSky Network** The system MUST support this.
- REST API: historical state vectors, flight tracks, arrivals/departures
- Trino/Impala: bulk historical queries (academic accounts)
- Rate: free tier 100 req/day, academic tier unlimited
- Format: JSON state vectors (time, icao24, lat, lon, baro_alt, velocity, heading, vertical_rate)
- Coverage: global, ~30M flights/year, 1 Hz update rate (aggregated to ~5s in API)

**ADS-B Exchange**
- Raw SBS BaseStation format (CSV-like, real-time or historical)
- Higher fidelity than OpenSky (1 Hz native)
- Format: MSG type 3 (position), type 4 (velocity), type 1 (ID)
- API key required for historical data

## ADDED Requirements

### Requirement: Download ADS-B data
Fetch historical flight data by time range, bounding box, and optional ICAO24 filter. The system MUST support this.

#### Scenario: Fetch single flight track
- Given an ICAO24 identifier and time range
- When the download function is called
- Then it returns a time-ordered sequence of ADS-B position reports

### Requirement: Parse ADS-B formats
Parse SBS BaseStation format, OpenSky JSON state vectors, and OpenSky CSV bulk exports. The system MUST support this.

#### Scenario: Parse SBS BaseStation messages
- Given a file containing raw SBS BaseStation messages
- When the parser processes the file
- Then it extracts position (MSG type 3) and velocity (MSG type 4) records with correct field mapping

### Requirement: Convert to Measurement type
Map parsed ADS-B data to `Measurement::AdsB { lat, lon, alt, velocity, time }`. The system MUST support this.

#### Scenario: Convert OpenSky state vectors to measurements
- Given a set of parsed OpenSky JSON state vectors
- When the conversion function is applied
- Then each state vector becomes a `Measurement::AdsB` with lat, lon, alt, velocity, and time fields populated

### Requirement: Extract ground truth trajectories
Interpolate ADS-B positions to a regular time grid for ground truth trajectories (ADS-B is itself the "truth" for cooperative targets). The system MUST support this.

#### Scenario: Interpolate positions to regular time grid
- Given a set of irregularly-spaced ADS-B reports for a single ICAO24
- When ground truth extraction is performed
- Then it produces a uniformly-sampled trajectory with interpolated positions and velocities

### Requirement: Coordinate transform
Convert WGS84 lat/lon/alt to ENU (East-North-Up) local frame relative to a configurable reference point. The system MUST support this.

#### Scenario: Transform WGS84 to ENU coordinates
- Given ADS-B positions in WGS84 lat/lon/alt and a reference point
- When the coordinate transform is applied
- Then positions are expressed in ENU meters relative to the reference point

### Requirement: Cache downloaded data
Download data once, then cache locally in a standard directory (~/.thresh/data/). The system MUST support this.

#### Scenario: Retrieve previously downloaded data from cache
- Given a flight data query that has been fetched before
- When the same query is requested again
- Then the data is loaded from the local cache without making a network request

### Requirement: Rate limiting
Respect OpenSky API rate limits with backoff. The system MUST support this.

#### Scenario: Back off when rate limited
- Given a sequence of API requests that exceeds the OpenSky free-tier rate limit
- When a rate-limit response is received
- Then the client waits with exponential backoff before retrying

### Output Format
- `Vec<Measurement::AdsB>` time-ordered measurement stream
- `Vec<GroundTruth>` with ICAO24 as target ID, ENU position, interpolated velocity
- Metadata: number of targets, time span, geographic bounds

### Test Scenarios
- Single flight track (e.g., JFK→LAX) for basic tracking validation
- Crossing traffic at a busy airport approach (multiple targets, close proximity)
- Oceanic track with sparse ADS-B coverage (tests coast handling)
- High-density airspace (e.g., KJFK TRACON, 50+ simultaneous targets)
