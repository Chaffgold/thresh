## 1. Crate Setup

- [ ] 1.1 Create `thresh-data` crate with feature flags: `adsb`, `orbital`, `nuscenes`
- [ ] 1.2 Add workspace dependencies: `reqwest` (blocking), `csv`, `sgp4`, `toml` for credentials
- [ ] 1.3 Define `Dataset` trait: `metadata()`, `frames()`, `ground_truth()`
- [ ] 1.4 Define `Frame` struct: timestamp, measurements, optional ground truth, sensor metadata
- [ ] 1.5 Define `DatasetMetadata`: name, source, target count, time span, coordinate frame
- [ ] 1.6 Implement credential loader: env vars → `~/.thresh/credentials.toml` fallback
- [ ] 1.7 Implement cache directory management: `~/.thresh/data/<source>/<dataset>/`

## 2. Coordinate Transforms

- [ ] 2.1 Implement WGS84 geodetic (lat, lon, alt) → ECEF (x, y, z)
- [ ] 2.2 Implement ECEF → ENU relative to configurable reference point
- [ ] 2.3 Implement WGS84 → ENU convenience function (compose 2.1 + 2.2)
- [ ] 2.4 Implement TEME → ECEF (for SGP4 output, includes Earth rotation via GMST)
- [ ] 2.5 Write tests: known reference points (e.g., JFK airport) match published ECEF coordinates
- [ ] 2.6 Write tests: ENU roundtrip (WGS84 → ENU → WGS84) within 1 cm

## 3. ADS-B Ingestion

- [ ] 3.1 Implement OpenSky REST client: fetch state vectors by time range and bounding box
- [ ] 3.2 Implement OpenSky flight track endpoint: get full trajectory for a specific flight
- [ ] 3.3 Parse OpenSky JSON state vectors into intermediate structs
- [ ] 3.4 Implement SBS BaseStation format parser (ADS-B Exchange CSV)
- [ ] 3.5 Convert parsed ADS-B records → `Measurement::AdsB` with WGS84→ENU transform
- [ ] 3.6 Extract ground truth trajectories: group by ICAO24, interpolate to regular time grid
- [ ] 3.7 Implement `AdsBDataset` implementing `Dataset` trait
- [ ] 3.8 Add download caching with content-hash deduplication
- [ ] 3.9 Implement rate limiting with exponential backoff for OpenSky API
- [ ] 3.10 Write tests: parse known SBS messages, verify field extraction
- [ ] 3.11 Write tests: round-trip a known flight (e.g., mock OpenSky response)
- [ ] 3.12 Write integration test (network, gated): fetch 1 hour of JFK approach data

## 4. Orbital Data Ingestion

- [ ] 4.1 Implement space-track.org REST client with session cookie auth
- [ ] 4.2 Implement TLE two-line format parser
- [ ] 4.3 Implement GP JSON format parser
- [ ] 4.4 Integrate `sgp4` crate: propagate TLE to TEME state vector at arbitrary epoch
- [ ] 4.5 Chain SGP4 → TEME → ECEF → ENU for ground-station-relative positions
- [ ] 4.6 Generate synthetic radar measurements from orbital positions + ground station geometry
- [ ] 4.7 Compute pass predictions (rise/set/max elevation) for station-object pairs
- [ ] 4.8 Implement `OrbitalDataset` implementing `Dataset` trait
- [ ] 4.9 Implement CelesTrak TLE fetcher as backup source
- [ ] 4.10 Write tests: parse known ISS TLE, verify SGP4 position within 10 km of published
- [ ] 4.11 Write tests: TEME→ECEF→ENU chain matches reference implementation
- [ ] 4.12 Write integration test (network, gated): fetch ISS TLE, propagate, generate radar scenario

## 5. nuScenes Ingestion

- [ ] 5.1 Implement PyO3 bridge to nuScenes devkit (feature-gated `nuscenes`)
- [ ] 5.2 Implement scene/sample iteration via Python bridge
- [ ] 5.3 Parse 3D annotation boxes → `BoundingBox3D` with class mapping
- [ ] 5.4 Extract instance-level tracks for ground truth (instance token → target ID)
- [ ] 5.5 Parse LiDAR point clouds from binary files (x, y, z, intensity, ring)
- [ ] 5.6 Parse radar point clouds with RCS and velocity
- [ ] 5.7 Load sensor calibration (extrinsics + intrinsics) for multi-modal alignment
- [ ] 5.8 Implement `NuScenesDataset` implementing `Dataset` trait
- [ ] 5.9 Write tests: parse annotations from mini split, verify box dimensions
- [ ] 5.10 Write integration test (requires mini split): load scene, run tracker, compute AMOTA

## 6. Dataset Abstraction and Mixing

- [ ] 6.1 Implement `SyntheticDataset` adapter wrapping `thresh-synth::Scenario`
- [ ] 6.2 Implement `MixedDataset` combining multiple sources with time alignment
- [ ] 6.3 Implement temporal bucketing: group measurements within configurable time window
- [ ] 6.4 Implement lazy frame iteration (don't load all data into memory)
- [ ] 6.5 Write tests: MixedDataset merges two synthetic streams in time order
- [ ] 6.6 Write tests: temporal bucketing groups measurements within 50ms window

## 7. Benchmark Scenarios

- [ ] 7.1 Define scenario manifest format (TOML): source, parameters, expected baselines
- [ ] 7.2 Implement `thresh-data fetch <scenario>` CLI for data download
- [ ] 7.3 Create scenario config: `adsb-single-flight`
- [ ] 7.4 Create scenario config: `adsb-tracon`
- [ ] 7.5 Create scenario config: `orbital-iss`
- [ ] 7.6 Create scenario config: `orbital-starlink-train`
- [ ] 7.7 Create scenario config: `nuscenes-mini`
- [ ] 7.8 Create scenario configs: `synth-cv-clean`, `synth-maneuvering`, `synth-heterogeneous`, `synth-low-pd`
- [ ] 7.9 Implement benchmark runner: load scenario → run tracker → compute metrics → compare baselines
- [ ] 7.10 Implement regression check: fail if MOTA/HOTA drops below threshold
- [ ] 7.11 Add CI job for synthetic benchmarks (no network required)
- [ ] 7.12 Add nightly CI job for network-dependent benchmarks
