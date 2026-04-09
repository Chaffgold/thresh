## Why

Multi-object tracking (MOT) for heterogeneous targets — spanning UAVs, manned aircraft, and ballistic missiles — requires fusing data from radar, EO/IR, and ADS-B sensors with dynamics that span orders of magnitude (10^0 to 10^4 m/s). No open-source Rust framework exists for transformer-based multi-sensor fusion tracking. The autonomous driving community has proven these architectures work (TransFusion, BEVFusion achieve 70+ mAP on nuScenes), but the defense/aerospace domain lacks both the tooling and the public datasets to apply them. Building a modular, high-performance Rust pipeline — with Stone Soup as a Python dependency for classical tracking algorithms until native Rust implementations are ready — fills this gap now while the research is mature enough to be actionable.

## What Changes

- Introduce a new Rust crate (`thresh`) implementing a modular transformer-based multi-sensor fusion tracking pipeline
- Implement core state estimation (Kalman filter family: KF, EKF, UKF) in Rust with CTRV and coordinated-turn motion models
- Implement data association algorithms (Hungarian/linear assignment, gated Mahalanobis distance) in Rust
- Build multi-sensor fusion mathematics: centralized measurement-level fusion, information filter form, covariance intersection
- Create a transformer inference runtime using ONNX Runtime for modular model execution (separate encoder, fusion, detection head components)
- Implement track management (track birth/death, M-of-N initiation, scored track deletion)
- Integrate Stone Soup (Python) as a dependency via PyO3/FFI for advanced algorithms (JPDA, MHT, IMM) until Rust-native versions exist
- Provide a synthetic data generation pipeline for radar/EO-IR sensor signatures
- Support ONNX model loading for transformer detection components (BEVFusion/TransFusion-style architectures)
- Implement evaluation metrics (MOTA, IDF1, HOTA, AMOTA)

## Capabilities

### New Capabilities
- `state-estimation`: Kalman filter family (KF, EKF, UKF) with configurable motion models (constant velocity, CTRV, coordinated turn) and Joseph-form numerically stable covariance updates
- `data-association`: Hungarian algorithm for optimal linear assignment, Mahalanobis-gated cost matrices, IoU-based and fused motion-appearance cost support
- `sensor-fusion`: Multi-sensor measurement fusion — centralized stacked measurement, information filter form for decentralized architectures, covariance intersection for unknown correlations
- `transformer-inference`: ONNX Runtime integration for loading and executing modular transformer pipeline components (encoders, BEV pooling, fusion modules, detection heads) with dynamic shape support and TensorRT backend
- `track-management`: Track lifecycle management — birth (M-of-N initiation), maintenance (state propagation + association), death (scored deletion), with class-specific track heads for heterogeneous target dynamics
- `stonesoup-bridge`: PyO3-based bridge to Stone Soup Python library for JPDA, MHT, IMM, and other advanced algorithms not yet implemented in Rust
- `synthetic-data`: Synthetic sensor data generation for radar returns, EO/IR signatures, and ADS-B feeds with configurable target dynamics, RCS profiles, and noise characteristics
- `evaluation-metrics`: MOT evaluation metrics — MOTA, MOTP, IDF1, HOTA (with DetA/AssA decomposition), AMOTA for 3D tracking benchmarks

### Modified Capabilities

(none — greenfield project)

## Impact

- **New crate**: `thresh` Rust workspace with sub-crates for each capability module
- **Dependencies**: `ort` (ONNX Runtime Rust bindings), `nalgebra` (linear algebra), `ndarray`, `pyo3` (Stone Soup bridge), `lapjv` or `pathfinding` (Hungarian algorithm)
- **Python dependency**: Stone Soup (`stonesoup`) for advanced tracking algorithms via PyO3 interop
- **Build requirements**: ONNX Runtime C library, Python 3.10+ with Stone Soup installed for bridge functionality
- **External model assets**: Pre-trained ONNX model components (not included in repo — loaded at runtime)
- **Deployment**: Modular pipeline components, each independently testable and certifiable
