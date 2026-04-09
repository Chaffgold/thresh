# thresh

Multi-sensor fusion multi-object tracking framework in Rust.

Hybrid architecture: transformer-based detection (via ONNX Runtime) + classical Bayesian state estimation (Kalman filter family). Designed for heterogeneous aerospace targets spanning UAVs through ballistic missiles.

## Workspace

| Crate | Description |
|---|---|
| `thresh-core` | Common types: state vectors, measurements, covariance matrices, coordinates |
| `thresh-filter` | KF, EKF, UKF with CV, CA, CTRV, Coordinated Turn motion models |
| `thresh-association` | Hungarian algorithm, Mahalanobis gating, 2D/3D IoU, cascaded association |
| `thresh-fusion` | Centralized fusion, information filter, covariance intersection |
| `thresh-inference` | ONNX Runtime pipeline orchestration (feature-gated) |
| `thresh-tracker` | Track lifecycle, M-of-N confirmation, class-specific heads |
| `thresh-bridge` | PyO3 bridge to Stone Soup (feature-gated) |
| `thresh-synth` | Synthetic trajectory + sensor data generation |
| `thresh-eval` | MOT metrics: MOTA, MOTP, IDF1, HOTA, AMOTA |

## Quick start

```rust
use thresh_tracker::tracker::MultiObjectTracker;
use nalgebra::DVector;

// Create a tracker with 10m measurement noise, 100 chi-squared gate
let mut tracker = MultiObjectTracker::new_cv_position(10.0, 100.0);

// Feed detections each frame
let detections = vec![
    DVector::from_column_slice(&[1000.0, 2000.0, 5000.0]),
];
tracker.step(&detections, 1.0); // dt = 1 second
```

## Building

```sh
cargo build --workspace
cargo test --workspace
```

Optional features:
- `onnx` on `thresh-inference`: enables ONNX Runtime (requires runtime binaries)
- `stonesoup` on `thresh-bridge`: enables PyO3 Stone Soup integration

## License

Apache-2.0
