## Why

The thresh tracker has a synthetic data pipeline for controlled testing, but no real-world data ingestion. The reference literature identifies the data gap as the single largest barrier to extending transformer tracking to aerospace/defense. Validating against real sensor data from multiple modalities (ADS-B, radar, EO/IR, orbital TLEs) is essential to prove the tracker works beyond synthetic scenarios and to benchmark against known ground truth.

## What Changes

- Add data source connectors for OpenSky Network (ADS-B), space-track.org (TLEs/orbital), nuScenes (multi-modal automotive), and ADS-B Exchange (raw ADS-B)
- Implement format parsers that convert each source's native format into thresh `Measurement` types
- Create ground truth extraction from each source (ADS-B positions as GT, TLE-propagated orbits as GT, nuScenes annotations as GT)
- Build a unified dataset abstraction that presents heterogeneous sources as time-ordered measurement streams compatible with the tracker
- Add download/caching scripts for dataset acquisition
- Create benchmark scenarios from real data for regression testing

## Capabilities

### New Capabilities
- `adsb-ingest`: OpenSky Network and ADS-B Exchange data download, parsing (SBS/CSV/JSON), conversion to `Measurement::AdsB`, trajectory extraction for ground truth
- `orbital-ingest`: space-track.org TLE download, SGP4 orbit propagation for ground truth trajectories, synthetic radar/EO-IR measurement generation from orbital positions
- `nuscenes-ingest`: nuScenes devkit integration, 3D annotation parsing, multi-modal measurement extraction (LiDAR, camera, radar), conversion to `BoundingBox3D` and `Measurement` types
- `dataset-abstraction`: Unified `Dataset` trait providing time-ordered measurement streams and ground truth, dataset caching/storage, format-agnostic benchmark runner
- `benchmark-scenarios`: Curated real-data scenarios (single aircraft, crossing tracks, high-density airspace, orbital conjunction, nuScenes urban driving) with expected metric baselines

### Modified Capabilities
- `synthetic-data`: Add ability to mix synthetic measurements with real data for augmentation and gap-filling
- `evaluation-metrics`: Add dataset-aware evaluation that handles different ground truth formats and coordinate systems

## Impact

- New crate: `thresh-data` for all data ingestion and dataset abstraction
- New dependencies: `reqwest` (HTTP), `csv`, `sgp4` (orbit propagation), optional `nuscenes` Python bridge via PyO3
- Network access required for data download (cached locally after first fetch)
- Credential management for space-track.org (username/password) and OpenSky (optional API key)
- `thresh-eval` gains dataset-aware evaluation modes
- `thresh-synth` gains real-data augmentation capability
