## Capability: Unified Dataset Abstraction

### Overview
Provide a common interface over all data sources (ADS-B, orbital, nuScenes, synthetic) so the tracker and evaluation pipeline can consume any source without specialization.

## ADDED Requirements

### Requirement: Dataset trait
Common interface returning time-ordered frames with measurements and optional ground truth. The system MUST support this.

#### Scenario: Consume different data sources through a common trait
- Given implementations of the Dataset trait for ADS-B, orbital, and nuScenes sources
- When a tracker is configured with any of these sources
- Then it can iterate frames and access measurements and ground truth without source-specific code

### Requirement: Frame abstraction
Each frame provides a timestamp, `Vec<Measurement>` from one or more sensors, optional `Vec<GroundTruth>` for evaluation, optional `Vec<BoundingBox3D>` for detection-based sources (nuScenes), and sensor metadata indicating which sensors contributed. The system MUST support this.

#### Scenario: Access all data from a single frame
- Given a frame yielded by any Dataset implementation
- When the frame is queried
- Then it provides a timestamp, measurements, optional ground truth, optional bounding boxes, and sensor metadata

### Requirement: Multi-source fusion
Combine measurements from multiple sources into a single time-ordered stream (e.g., ADS-B + synthetic radar). The system MUST support this.

#### Scenario: Merge ADS-B and synthetic radar into one stream
- Given an ADS-B dataset and a synthetic radar dataset covering the same time window
- When multi-source fusion is applied
- Then the resulting stream interleaves measurements from both sources in time order

### Requirement: Coordinate normalization
All sources produce data in a common frame (configurable: ENU, ECEF, or scene-local). The system MUST support this.

#### Scenario: Normalize coordinates across sources
- Given datasets using different native coordinate systems
- When coordinate normalization is configured to ENU
- Then all measurements and ground truth positions are expressed in the same ENU frame

### Requirement: Time alignment
Handle different sensor rates (ADS-B at 1 Hz, radar at 10 Hz, LiDAR at 20 Hz) via temporal bucketing. The system MUST support this.

#### Scenario: Bucket measurements from different rates
- Given ADS-B data at 1 Hz and radar data at 10 Hz
- When time alignment is applied with a configurable bucket width
- Then measurements are grouped into time buckets preserving their original timestamps

### Requirement: Data caching
Local storage in `~/.thresh/data/<source>/<dataset>/` with manifest files. The system MUST support this.

#### Scenario: Cache a dataset locally with manifest
- Given a dataset that has been downloaded from a remote source
- When caching is performed
- Then the data is stored under `~/.thresh/data/<source>/<dataset>/` with a manifest file describing contents and provenance

### Requirement: Lazy loading
Don't load all frames into memory; iterate on demand. The system MUST support this.

#### Scenario: Iterate a large dataset without loading it all into memory
- Given a dataset with thousands of frames
- When the frame iterator is used
- Then frames are loaded one at a time on demand, keeping memory usage bounded

### Requirement: Reproducibility
Each dataset instance is deterministic given a config; support seeded random for synthetic augmentation. The system MUST support this.

#### Scenario: Reproduce identical frame sequences from the same config
- Given a dataset configuration with a fixed random seed
- When the dataset is instantiated twice with the same config
- Then both instances produce identical frame sequences

### Output Interface
```
trait Dataset {
    fn metadata(&self) -> DatasetMetadata;
    fn frames(&self) -> impl Iterator<Item = Frame>;
    fn ground_truth(&self) -> Option<impl Iterator<Item = GroundTruth>>;
}
```

### Implementations
- `AdsBDataset` — from OpenSky/ADS-B Exchange
- `OrbitalDataset` — from space-track.org TLEs
- `NuScenesDataset` — from nuScenes devkit (feature-gated)
- `SyntheticDataset` — from thresh-synth scenarios
- `MixedDataset` — combines multiple sources with time alignment
