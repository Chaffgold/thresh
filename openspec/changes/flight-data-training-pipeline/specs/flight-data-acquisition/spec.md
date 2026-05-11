## Capability: Flight Data Acquisition

### Overview

A unified Python ingestion layer that pulls real flight-data trajectories from two providers — OpenSky Network (primary, historical, redistributable) and ADS-B Exchange v2 (live, geographically scoped, edge cases) — into a single canonical Parquet/Arrow schema. The acquisition layer is the shared foundation for both Track A (learned detector) and Track B (learned tracker components).

**Data semantics.** Records produced by this layer are **system-level / track-level truth**: aircraft self-reports of GPS-derived state, identity-tagged by ICAO address, with quality encoded as NIC/NAC flags rather than a measurement covariance. They are consumed downstream as ground-truth target trajectories and are never used directly as sensor measurements. All measurement-level data in this pipeline is synthesised downstream by `thresh-synth` from these trajectories — see `specs/learned-detector/spec.md` and `specs/learned-tracker-components/spec.md` for the consumption contracts.

## ADDED Requirements

### Requirement: Canonical trajectory schema

The acquisition layer MUST define a single canonical schema for flight trajectories that both OpenSky and ADSBx are translated into. Every trajectory record SHALL include: 24-bit ICAO address, timestamp (UTC, microsecond precision), WGS84 latitude and longitude, geometric and barometric altitudes, ground velocity and track, vertical rate, ADS-B emitter category, callsign, NIC and NAC_p quality indicators, and the source name (`opensky` or `adsbx`).

#### Scenario: OpenSky state vector translates to canonical schema

**WHEN** an OpenSky state-vector record is fetched containing `(icao24, time_position, lat, lon, geo_altitude, baro_altitude, velocity, true_track, vertical_rate, category, callsign)`

**THEN** the acquisition layer emits a canonical trajectory record with `source = "opensky"`, lat/lon as WGS84, altitudes in metres, velocity in m/s, and all timestamps as microseconds since the UNIX epoch.

**SHALL** preserve original quality fields (`nic`, `nac_p`) when available and emit `null` when not.

#### Scenario: ADSBx readsb record translates to canonical schema

**WHEN** an ADSBx v2 aircraft record is fetched containing `(hex, flight, lat, lon, alt_baro, alt_geom, gs, track, baro_rate, category, nic, nac_p)`

**THEN** the acquisition layer emits a canonical trajectory record with `source = "adsbx"`, `icao24` set from `hex`, callsign from `flight`, ground velocity from `gs` converted to m/s, vertical rate from `baro_rate` converted to m/s, and altitudes converted to metres.

**SHALL** also emit a `provenance` field indicating which subfields came from MLAT or TIS-B per the `mlat[]` / `tisb[]` arrays in the source record.

### Requirement: OpenSky historical client

The acquisition layer MUST provide a Python client that pulls historical state vectors from OpenSky's REST endpoints and from the Zenodo-published trajectory dump, writing results to Parquet under the canonical schema.

#### Scenario: Pulling historical state vectors by bounding box and time range

**WHEN** a developer calls `opensky.fetch_state_vectors(bbox, time_range)` with a WGS84 bounding box and a `(start, end)` UTC time pair

**THEN** the client returns an iterator of canonical trajectory records covering all aircraft within the box during that window

**SHALL** retry transient HTTP failures up to 3 times with exponential backoff and surface non-recoverable errors as a Python exception with the OpenSky response body included.

#### Scenario: Loading a Zenodo trajectory dump

**WHEN** a developer calls `opensky.load_zenodo_dump(path)` with a path to a Zenodo-published trajectory file

**THEN** the loader yields canonical trajectory records preserving original timestamps and quality indicators

**SHALL** validate the file's checksum against the published Zenodo metadata and fail loudly on mismatch.

### Requirement: ADSBx v2 live client

The acquisition layer MUST provide a Python client that polls the ADS-B Exchange v2 gateway's geospatial-filtering endpoints, translates responses to canonical trajectory records, and appends them to Parquet storage with deduplication.

#### Scenario: Polling airport-scoped snapshots

**WHEN** a developer runs `adsbx_poller.run(airport_icao, api_key, rate_limit_hz, duration)` with an airport ICAO code (e.g. `KSEA`), an ADSBx API key, a polling rate (default 1 Hz), and a polling duration

**THEN** the poller fetches `/api/aircraft/v2/airport/{icao}` at the requested rate, translates each response to canonical records, and appends them to a Parquet file partitioned by source and airport

**SHALL** respect the rate limit by sleeping between requests, deduplicate on `(icao24, timestamp)` before appending, and stop cleanly when `duration` elapses.

#### Scenario: Handling rate-limit responses

**WHEN** the ADSBx API returns an HTTP 429 (Too Many Requests) response

**THEN** the poller backs off using the `Retry-After` header value and resumes when the cooldown expires

**SHALL NOT** crash on a single 429; SHALL crash on three consecutive 429s after backoff to surface a misconfigured rate limit.

### Requirement: Track stitching from snapshots

The acquisition layer MUST stitch point-in-time snapshots into per-ICAO trajectories. A "track" is a contiguous sequence of state-vector observations for a single `icao24` with no gap larger than a configurable threshold (default 60 seconds).

#### Scenario: Splitting a series with a large temporal gap

**WHEN** a stitcher receives state vectors for a single `icao24` with timestamps `[t0, t0+10s, t0+20s, t0+200s, t0+210s]` and the gap threshold is 60 seconds

**THEN** the stitcher emits two distinct tracks: `[t0, t0+10s, t0+20s]` and `[t0+200s, t0+210s]`

**SHALL** assign each emitted track a stable identifier of the form `{icao24}-{first_timestamp_iso}`.

### Requirement: Parquet storage layout

The acquisition layer MUST write trajectory records to Parquet files partitioned by source and date (UTC), with a stable file naming convention.

#### Scenario: Writing a day's worth of OpenSky data

**WHEN** the OpenSky client writes records spanning a single UTC day

**THEN** records are written to a single Parquet file at `<root>/source=opensky/date=YYYY-MM-DD/trajectories.parquet`

**SHALL** ensure timestamps are monotonically increasing within each `icao24` partition.

### Requirement: Class taxonomy mapping

The acquisition layer MUST provide a Python utility that maps ADS-B emitter categories to a thresh-internal class enum of five buckets: `light-fixed-wing`, `heavy-fixed-wing`, `rotorcraft`, `glider-or-balloon-or-uav`, `other`.

#### Scenario: Mapping common emitter categories

**WHEN** the mapping utility is called with category codes `A1`, `A4`, `A7`, `B1`, and `C2`

**THEN** it returns `light-fixed-wing`, `heavy-fixed-wing`, `rotorcraft`, `glider-or-balloon-or-uav`, and `other` respectively

**SHALL** return `other` for any unrecognised or null category code without raising.

### Requirement: License posture documentation

The acquisition layer MUST document the redistribution posture of each source in a top-level `LICENSING.md` file. OpenSky-derived data is documented as redistributable under the OpenSky terms with attribution; ADSBx-derived data is documented as not redistributable, with the acquisition script shipped and an API-key bootstrap procedure provided.

#### Scenario: Reviewing licensing posture

**WHEN** a contributor opens `LICENSING.md`

**THEN** they find a section per source listing: license name, attribution requirement, redistribution permission, and the exact attribution string to include in derived artefacts

**SHALL** include the OpenSky attribution string verbatim in any redistributed dataset metadata file.

### Requirement: CI dry-run for acquisition layer

The acquisition layer MUST be exercised by a PR-time CI job that performs a tiny live query against OpenSky's free REST tier and asserts the response parses into the canonical schema without error.

#### Scenario: PR-time schema parity check

**WHEN** a pull request modifies any file under `python/acquisition/`

**THEN** the CI workflow runs a small OpenSky query (e.g. a 5-minute window over a 50 nm box) and asserts at least one canonical trajectory record is produced

**SHALL** skip the job if no `python/acquisition/` files are touched, and SHALL NOT call ADSBx in CI (no shared API key).
