# Transformer-Based Detection Pipeline — Design

## Context

thresh-inference has ONNX Runtime plumbing (session management, tensor I/O, provider selection) but no detection pipeline. The hybrid architecture goal — transformer detection feeding classical Bayesian tracking — is unimplemented. thresh-core already has a `BoundingBox3D` type in `detection.rs` with position, dimensions, yaw, confidence, class_id, and optional velocity. The tracker (`MultiObjectTracker::step`) currently accepts `&[DVector<f64>]` as detections.

## Goals / Non-Goals

**Goals:**
- Define a `DetectionPipeline` trait in thresh-inference with `detect(&self, input: &SensorInput) -> Vec<Detection3D>`
- Implement `OnnxDetector` that loads a DETR/RT-DETR `.onnx` model via `ort::Session`
- Pre-processing: sensor input (point cloud or image) to voxelized/normalized tensor
- Post-processing: raw model logits to `Detection3D` structs with confidence thresholding and NMS
- Add `Detection3D` type to thresh-core bridging `BoundingBox3D` with tracker-compatible measurement conversion
- Allow `MultiObjectTracker` to accept `Vec<Detection3D>` as an alternative input path
- Ship a small synthetic test model (< 1 MB) for CI; real models are user-supplied
- Feature-gated behind `onnx` on thresh-inference

**Non-Goals:**
- Model training or fine-tuning
- Custom transformer architectures beyond DETR/RT-DETR
- Real-time video capture or camera frame management
- GPU-only paths — CPU fallback is mandatory for CI
- End-to-end learned tracking (detection only; tracking stays classical)

## Decisions

### Detection3D in thresh-core

Extend the existing `detection.rs` module. `Detection3D` wraps `BoundingBox3D` and adds an `embedding: Option<Vec<f64>>` field for re-identification features. Provide `Detection3D::to_measurement() -> DVector<f64>` that extracts `[x, y, z]` (or `[x, y, z, vx, vy]` if velocity is present) for direct use in the tracker's Kalman filter update. This avoids forcing callers to manually convert bounding boxes to state-space measurements.

### DetectionPipeline trait in thresh-inference

```rust
pub trait DetectionPipeline: Send + Sync {
    fn detect(&self, input: &SensorInput) -> Result<Vec<Detection3D>, DetectionError>;
}
```

`SensorInput` is an enum covering `PointCloud(Vec<[f64; 4]>)` (x, y, z, intensity) and `ImageTensor(ndarray::Array4<f32>)` (NCHW). The trait is object-safe so it can be stored as `Box<dyn DetectionPipeline>`.

### OnnxDetector struct

Wraps `ort::Session`. Configuration via `OnnxDetectorConfig`:
- `model_path: PathBuf`
- `confidence_threshold: f64` (default 0.3)
- `nms_iou_threshold: f64` (default 0.5)
- `voxel_size: [f64; 3]` (for point cloud voxelization)
- `max_points_per_voxel: usize`

Construction is `OnnxDetector::from_config(config) -> Result<Self>`. The session is created once and reused. Pre-processing and post-processing are separate private methods for testability.

### Pre-processing pipeline

Point cloud path: raw points are voxelized into a fixed-size 3D grid. Each voxel aggregates points (mean position + count). The grid is flattened to an `ort::Value` tensor. Image path: normalize to model-expected range, resize to model input dimensions. Both paths produce `Vec<ort::Value>` for session input.

### Post-processing pipeline

Model output (class logits + bbox regression) is decoded to `BoundingBox3D` candidates. Apply confidence threshold first (cheap filter), then axis-aligned NMS using IoU on the bird's-eye-view projection. NMS uses the existing `BoundingBox3D` geometry. Output: `Vec<Detection3D>` sorted by descending confidence.

### Tracker integration

Add `MultiObjectTracker::step_detections(&mut self, detections: &[Detection3D], dt: f64)` that calls `Detection3D::to_measurement()` on each detection, then delegates to the existing `step()`. This keeps the internal tracker logic unchanged while providing a typed API for detection pipeline users.

### Test model for CI

Ship a tiny ONNX model (random weights, correct input/output shapes) under `test-data/models/`. CI tests verify the pipeline loads the model, runs inference, and produces structurally valid `Detection3D` outputs. Numerical correctness is not tested with the dummy model — that requires real weights.

## Risks / Trade-offs

- **ort version coupling**: The `ort` crate has frequent breaking changes. Pin to a specific version range and document upgrade procedure.
- **Voxelization performance**: Naive voxelization is O(n) in points but can be memory-heavy for large grids. Start simple; optimize if profiling shows it matters.
- **Model portability**: Different DETR variants have different input/output tensor shapes. The initial implementation targets RT-DETR's specific schema; other variants will need adapter implementations of `DetectionPipeline`.
- **Test model size**: Even a tiny model adds binary weight to the repo. Use Git LFS or generate the test model at build time via a build script.

## Open Questions

1. Should `SensorInput` live in thresh-core (so other crates can reference it) or thresh-inference (since only inference uses it)?
2. Should we support batched inference (multiple frames per `detect` call) in the initial API, or add it later?
3. What is the minimum ONNX opset version we should target for model compatibility?
