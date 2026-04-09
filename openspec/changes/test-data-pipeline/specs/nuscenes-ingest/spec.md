## Capability: nuScenes Dataset Ingestion

### Overview
Ingest nuScenes multi-modal autonomous driving data for validating the transformer fusion pipeline. nuScenes is the benchmark used by TransFusion, BEVFusion, and CenterPoint — the architectures thresh's inference pipeline targets.

### Requirement: Data Source

**nuScenes** The system MUST support this.
- 1,000 scenes, 20 seconds each, collected in Boston and Singapore
- 6 cameras (360° coverage), 1 LiDAR (32-beam), 5 radars, GPS, IMU
- 1.4M 3D bounding box annotations across 23 object classes
- 2 Hz keyframe annotations, 20 Hz sensor data
- ~350 GB full dataset, ~4 GB mini split for development

**nuScenes devkit** (Python)
- Official Python package for data access
- Provides: scene iteration, annotation lookup, sensor calibration, coordinate transforms
- Required for reading the binary/proprietary formats

## ADDED Requirements

### Requirement: Dataset access
Support both mini (4 GB dev) and full (350 GB) splits. The system MUST support this.

#### Scenario: Load nuScenes mini split
- Given the nuScenes mini split is downloaded to the local data directory
- When the dataset is opened with the mini split configuration
- Then it provides access to all scenes, samples, and annotations in the mini split

### Requirement: Python bridge
Use PyO3 to call nuScenes devkit for data loading (feature-gated like Stone Soup). The system MUST support this.

#### Scenario: Call nuScenes devkit via PyO3
- Given the nuscenes-ingest feature flag is enabled and Python with nuscenes-devkit is available
- When the bridge initializes the NuScenes Python object
- Then it can query scenes, samples, and annotations through the Python interop layer

### Requirement: LiDAR parsing
Parse point clouds from .pcd.bin files to Vec<[f64; 4]> (x, y, z, intensity). The system MUST support this.

#### Scenario: Parse a LiDAR point cloud file
- Given a .pcd.bin file from a nuScenes keyframe
- When the LiDAR parser reads the file
- Then it returns a vector of 4D points with x, y, z coordinates and intensity values

### Requirement: Radar parsing
Parse radar point clouds to range/velocity/RCS per return. The system MUST support this.

#### Scenario: Parse radar returns from a keyframe
- Given radar data from a nuScenes sample
- When the radar parser processes the data
- Then it extracts range, velocity, and RCS for each radar return

### Requirement: Camera parsing
Provide image paths for passing to ONNX camera encoders (not pixel processing in Rust). The system MUST support this.

#### Scenario: Retrieve camera image paths for a keyframe
- Given a nuScenes keyframe sample token
- When camera data is requested
- Then it returns file paths for all 6 camera images associated with that keyframe

### Requirement: Annotation parsing
Parse 3D boxes with class, velocity, and visibility into `BoundingBox3D` and `GroundTruth`. The system MUST support this.

#### Scenario: Load ground truth annotations for a keyframe
- Given a nuScenes keyframe with annotated objects
- When annotation parsing is performed
- Then it returns BoundingBox3D structs with class label, position, dimensions, orientation, and velocity

### Requirement: Calibration loading
Load sensor extrinsics and intrinsics for multi-modal alignment. The system MUST support this.

#### Scenario: Load sensor calibration for a scene
- Given a nuScenes scene with multiple sensors
- When calibration data is requested
- Then it provides extrinsic and intrinsic parameters for all cameras, LiDAR, and radar sensors

### Requirement: Coordinate transform
Convert between nuScenes global frame, ego frame, and sensor frames. The system MUST support this.

#### Scenario: Transform points from sensor frame to global frame
- Given a LiDAR point cloud in sensor coordinates and the associated calibration
- When coordinate transform is applied
- Then the points are expressed in the nuScenes global coordinate frame

### Requirement: Scene iteration
Iterate keyframes in temporal order, providing synchronized multi-modal data per frame. The system MUST support this.

#### Scenario: Iterate through a scene in temporal order
- Given a nuScenes scene identifier
- When the scene iterator is used
- Then it yields keyframes in chronological order, each with synchronized LiDAR, radar, camera, and annotation data

### Output Format
- Per keyframe: LiDAR points, radar returns, camera paths, GT annotations
- `Vec<BoundingBox3D>` ground truth detections per frame
- `Vec<GroundTruth>` with instance tokens as target IDs for tracking evaluation
- Sensor calibration data for fusion pipeline configuration

### Test Scenarios
- Urban intersection (pedestrians, cyclists, vehicles — tests multi-class)
- Highway driving (high-speed vehicles, long tracking distances)
- Night/rain scenes (tests sensor degradation handling)
- Parking lot (dense slow-moving targets, tests association)

### Notes
- nuScenes tracking evaluation uses AMOTA metric — our eval module already implements this
- Mini split is sufficient for integration testing; full split for benchmarking
- Camera data is only needed when running the ONNX inference pipeline (BEVFusion/TransFusion)
