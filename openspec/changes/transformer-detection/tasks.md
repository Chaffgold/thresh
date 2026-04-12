# Transformer-Based Detection Pipeline — Tasks

## 1. Core detection types

- [x] 1.1 Add `Detection3D` struct to `crates/thresh-core/src/detection.rs` wrapping `BoundingBox3D` with optional `embedding: Vec<f64>` and a `frame_id: u64` field
- [x] 1.2 Implement `Detection3D::to_measurement() -> DVector<f64>` converting position (and optionally velocity) to a Kalman-filter-compatible measurement vector
- [x] 1.3 Add `SensorInputType` enum (`PointCloud`, `ImageTensor`) to `crates/thresh-core/src/detection.rs`
- [x] 1.4 Add `DetectionError` enum to thresh-core covering model load, inference, and shape mismatch errors

## 2. Detection pipeline trait

- [x] 2.1 Define `DetectionPipeline` trait in `crates/thresh-inference/src/detection.rs` with `detect(&self, input: &SensorInput) -> Vec<Detection3D>`
- [x] 2.2 Add `OnnxDetectorConfig` struct with `model_path`, `confidence_threshold`, `nms_iou_threshold`, `voxel_size`, `max_points_per_voxel`

## 3. Pre-processing

- [x] 3.1 Implement point cloud voxelization: `voxelize(points, voxel_size, max_per_voxel)` in `crates/thresh-inference/src/preprocess.rs`
- [x] 3.2 Implement image normalization: `normalize_image(data, mean, std, channels)` in `crates/thresh-inference/src/preprocess.rs`
- [x] 3.3 Unit test voxelization with known point distributions (uniform grid, single voxel, empty input)

## 4. ONNX inference

- [x] 4.1 Implement `OnnxDetector` struct wrapping `ort::Session`, constructed via `OnnxDetector::load()`, gated behind `#[cfg(feature = "onnx")]`
- [x] 4.2 Implement `DetectionPipeline` for `OnnxDetector`: placeholder implementation (returns empty detections)
- [ ] 4.3 Create a tiny synthetic ONNX test model (< 1 MB, random weights, correct RT-DETR input/output shapes) under `test-data/models/`

## 5. Post-processing

- [ ] 5.1 Implement confidence thresholding: filter raw model outputs below `confidence_threshold`
- [x] 5.2 Implement axis-aligned 3D IoU computation between two `Detection3D` instances
- [x] 5.3 Implement greedy NMS using 3D IoU in `crates/thresh-inference/src/detection.rs`
- [x] 5.4 Unit test NMS with overlapping and non-overlapping box sets

## 6. Tracker integration

- [ ] 6.1 Add `MultiObjectTracker::step_detections(&mut self, detections: &[Detection3D], dt: f64)` to `crates/thresh-tracker/src/tracker.rs`
- [ ] 6.2 Integration test: synthetic detections through `step_detections` produce confirmed tracks

## 7. CI and documentation

- [ ] 7.1 Add `onnx` feature gate to `crates/thresh-inference/Cargo.toml` controlling `ort` dependency
- [ ] 7.2 CI job: run `cargo test -p thresh-inference --features onnx` on CPU with the synthetic test model
- [ ] 7.3 Add doc comments and module-level documentation for the detection pipeline
