//! Detection pipeline trait, NMS, confidence filtering, and concrete detector
//! implementations.
//!
//! # Overview
//!
//! The [`DetectionPipeline`] trait defines the interface for all detection
//! backends. A pipeline consumes a [`SensorInput`] (point cloud, image tensor,
//! or other sensor frame) and produces a `Vec<Detection3D>`.
//!
//! Two concrete implementations are provided:
//!
//! * **[`MockDetector`]** — returns pre-configured detections; useful for unit
//!   and integration tests without a real model.
//! * **`OnnxDetector`** (behind the `onnx` feature gate) — loads a pre-trained
//!   ONNX model via `ort::Session` and runs inference on each frame. It decodes
//!   the three-output detector contract (`boxes` + `scores` + optional
//!   `classes`; see design Decision 9) into [`Detection3D`]s, then applies
//!   confidence filtering and NMS. A real deployment supplies a trained 3DETR
//!   (or similar) `.onnx` file; the checked-in `test_detector.onnx` is a
//!   random-weight stub for the shape contract.
//!
//! # Post-processing
//!
//! Raw model outputs go through two filtering stages before reaching the
//! tracker:
//!
//! 1. **Confidence thresholding** ([`filter_by_confidence`]) — drop detections
//!    below a score threshold.
//! 2. **Non-maximum suppression** ([`nms_3d`]) — remove duplicate boxes using
//!    axis-aligned 3D IoU.
//!
//! # Tracker bridge
//!
//! [`detections_to_tracker_input`] converts `Detection3D` positions into
//! `DVector<f64>` measurement vectors consumed by the Kalman-filter-based
//! `MultiObjectTracker::step` method.

use thresh_core::detection::Detection3D;

// ---------------------------------------------------------------------------
// ONNX detector configuration
// ---------------------------------------------------------------------------

/// Configuration for the ONNX-based 3D object detector.
#[derive(Debug, Clone)]
pub struct OnnxDetectorConfig {
    /// Path to the `.onnx` model file.
    pub model_path: String,
    /// Minimum confidence to keep a detection.
    pub confidence_threshold: f64,
    /// IoU threshold for non-maximum suppression.
    pub nms_iou_threshold: f64,
    /// Voxel dimensions `[x, y, z]` in metres (for point-cloud inputs).
    pub voxel_size: [f64; 3],
    /// Maximum number of points per voxel.
    pub max_points_per_voxel: usize,
}

impl Default for OnnxDetectorConfig {
    fn default() -> Self {
        Self {
            model_path: String::new(),
            confidence_threshold: 0.5,
            nms_iou_threshold: 0.4,
            voxel_size: [0.1, 0.1, 0.2],
            max_points_per_voxel: 35,
        }
    }
}

// ---------------------------------------------------------------------------
// Sensor input
// ---------------------------------------------------------------------------

/// Input to a detection pipeline, abstracting over sensor modalities.
#[derive(Debug, Clone)]
pub struct SensorInput {
    /// 3D point cloud — each element is `[x, y, z]`.
    pub points: Vec<[f64; 3]>,
    /// Optional per-point intensity values.
    pub intensities: Option<Vec<f64>>,
    /// Timestamp of the sensor frame (seconds since epoch or mission time).
    pub timestamp: f64,
}

// ---------------------------------------------------------------------------
// Detection pipeline trait
// ---------------------------------------------------------------------------

/// Trait for detection pipelines that produce 3D detections from sensor input.
pub trait DetectionPipeline {
    /// Run detection on the given sensor input.
    fn detect(&self, input: &SensorInput) -> Vec<Detection3D>;
    /// Human-readable name of this detector.
    fn name(&self) -> &str;
}

// ---------------------------------------------------------------------------
// 3D IoU and NMS
// ---------------------------------------------------------------------------

/// Compute 3D IoU between two axis-aligned bounding boxes.
///
/// This uses an axis-aligned approximation (ignores yaw). Intersection is the
/// overlap volume along each axis; union is `vol_a + vol_b - intersection`.
fn iou_3d(a: &Detection3D, b: &Detection3D) -> f64 {
    let overlap = |a_center: f64, a_extent: f64, b_center: f64, b_extent: f64| -> f64 {
        let a_min = a_center - a_extent / 2.0;
        let a_max = a_center + a_extent / 2.0;
        let b_min = b_center - b_extent / 2.0;
        let b_max = b_center + b_extent / 2.0;
        (a_max.min(b_max) - a_min.max(b_min)).max(0.0)
    };

    let ix = overlap(
        a.position[0],
        a.dimensions[0],
        b.position[0],
        b.dimensions[0],
    );
    let iy = overlap(
        a.position[1],
        a.dimensions[1],
        b.position[1],
        b.dimensions[1],
    );
    let iz = overlap(
        a.position[2],
        a.dimensions[2],
        b.position[2],
        b.dimensions[2],
    );

    let intersection = ix * iy * iz;
    let vol_a = a.volume();
    let vol_b = b.volume();
    let union = vol_a + vol_b - intersection;

    if union <= 0.0 {
        0.0
    } else {
        intersection / union
    }
}

/// Apply 3D non-maximum suppression to a set of detections.
///
/// Detections are sorted by confidence (descending). For each kept detection,
/// all subsequent detections whose IoU with it exceeds `iou_threshold` are
/// removed. The suppression uses an axis-aligned IoU approximation.
pub fn nms_3d(detections: &mut Vec<Detection3D>, iou_threshold: f64) {
    // Sort descending by confidence.
    detections.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap());

    let mut keep = vec![true; detections.len()];
    for i in 0..detections.len() {
        if !keep[i] {
            continue;
        }
        for j in (i + 1)..detections.len() {
            if !keep[j] {
                continue;
            }
            if iou_3d(&detections[i], &detections[j]) > iou_threshold {
                keep[j] = false;
            }
        }
    }

    let mut idx = 0;
    detections.retain(|_| {
        let k = keep[idx];
        idx += 1;
        k
    });
}

// ---------------------------------------------------------------------------
// Confidence filtering
// ---------------------------------------------------------------------------

/// Remove detections whose confidence score is below the given threshold.
///
/// This is the standard first post-processing step after raw model output:
/// discard anything the model is not sufficiently sure about before running
/// NMS or feeding into the tracker.
pub fn filter_by_confidence(detections: &mut Vec<Detection3D>, threshold: f64) {
    detections.retain(|d| d.confidence >= threshold);
}

// ---------------------------------------------------------------------------
// Tracker integration
// ---------------------------------------------------------------------------

/// Convert [`Detection3D`] positions to `DVector<f64>` for the tracker's
/// Kalman-filter measurement interface.
pub fn detections_to_tracker_input(detections: &[Detection3D]) -> Vec<nalgebra::DVector<f64>> {
    detections.iter().map(|d| d.to_measurement()).collect()
}

// ---------------------------------------------------------------------------
// MockDetector
// ---------------------------------------------------------------------------

/// A mock detector that returns pre-configured detections for testing.
pub struct MockDetector {
    /// The detections this mock will always return.
    pub detections: Vec<Detection3D>,
}

impl DetectionPipeline for MockDetector {
    fn detect(&self, _input: &SensorInput) -> Vec<Detection3D> {
        self.detections.clone()
    }

    fn name(&self) -> &str {
        "MockDetector"
    }
}

// ---------------------------------------------------------------------------
// OnnxDetector (feature-gated)
// ---------------------------------------------------------------------------

/// An ONNX Runtime-backed detector that loads a pre-trained model.
///
/// The model is loaded once and the session is reused for each `detect` call.
/// In this initial implementation the actual inference is a placeholder that
/// returns an empty detection list — a real deployment would supply a trained
/// RT-DETR (or similar) `.onnx` file at `model_path`.
#[cfg(feature = "onnx")]
pub struct OnnxDetector {
    // `ort::Session::run` takes `&mut self`, but `DetectionPipeline::detect`
    // is `&self`, so the session lives behind a `Mutex` for interior
    // mutability (also keeps `OnnxDetector` `Sync` for use as a trait object).
    session: std::sync::Mutex<ort::session::Session>,
    confidence_threshold: f64,
    nms_iou_threshold: f64,
}

#[cfg(feature = "onnx")]
impl OnnxDetector {
    /// Load a detector from an ONNX model file.
    ///
    /// # Errors
    /// Returns an `ort::Error` if the model cannot be loaded.
    pub fn load(
        model_path: &str,
        confidence_threshold: f64,
        nms_iou_threshold: f64,
    ) -> Result<Self, ort::Error> {
        let session = ort::session::Session::builder()?.commit_from_file(model_path)?;
        Ok(Self {
            session: std::sync::Mutex::new(session),
            confidence_threshold,
            nms_iou_threshold,
        })
    }

    /// Re-create the ONNX session with weights overridden from a SafeTensors file.
    ///
    /// This is **not** a hot-swap — the old session is dropped and a new one is
    /// created. ORT does not support direct weight injection into an existing
    /// session.
    ///
    /// # Current status
    ///
    /// This is a documented stub. A real implementation would:
    /// 1. Load SafeTensors weights via [`crate::weights::SafeTensorsLoader`].
    /// 2. Modify the ONNX model's protobuf initializers in-memory.
    /// 3. Create a new `ort::Session` from the modified model bytes.
    ///
    /// For production use, prefer [`crate::native_detector::NativeDetector`]
    /// with SafeTensors weights — it supports direct weight injection without
    /// protobuf manipulation.
    pub fn reload_with_weights(
        _config: &OnnxDetectorConfig,
        _weight_path: &str,
    ) -> Result<Self, crate::weights::WeightError> {
        // ORT doesn't support direct weight injection into an existing session.
        // Real implementation would require onnx-protobuf manipulation.
        Err(crate::weights::WeightError::Io(
            "ONNX weight replacement not yet implemented — use NativeDetector with SafeTensors instead".into(),
        ))
    }
}

// ONNX detector contract dimensions (Design Decision 9).
#[cfg(feature = "onnx")]
const DET_NUM_POINTS: usize = 1000;
#[cfg(feature = "onnx")]
const DET_POINT_DIM: usize = 4; // x, y, z, intensity
#[cfg(feature = "onnx")]
const DET_MAX_BOXES: usize = 100;
#[cfg(feature = "onnx")]
const DET_BOX_DIM: usize = 7; // x, y, z, L, W, H, yaw

#[cfg(feature = "onnx")]
impl OnnxDetector {
    /// Flatten a sensor frame into the detector's `(1, 1000, 4)`
    /// `[x, y, z, intensity]` input tensor (zero-padded or truncated to 1000
    /// points).
    fn build_input(input: &SensorInput) -> Vec<f32> {
        let mut data = vec![0.0_f32; DET_NUM_POINTS * DET_POINT_DIM];
        for (i, p) in input.points.iter().take(DET_NUM_POINTS).enumerate() {
            let intensity = input
                .intensities
                .as_ref()
                .and_then(|v| v.get(i))
                .copied()
                .unwrap_or(0.0);
            let base = i * DET_POINT_DIM;
            data[base] = p[0] as f32;
            data[base + 1] = p[1] as f32;
            data[base + 2] = p[2] as f32;
            data[base + 3] = intensity as f32;
        }
        data
    }

    /// Run the model and decode raw outputs into detections (before filtering).
    ///
    /// Reads `boxes` (1,100,7) + `scores` (1,100,1); the optional third
    /// `classes` output (1,100,1, int64) sets `class_id`, defaulting to 0 for
    /// backward-compatible 2-output models (task 7.8).
    fn run_and_decode(&self, input: &SensorInput) -> Result<Vec<Detection3D>, String> {
        let data = Self::build_input(input);
        let tensor = ort::value::Tensor::from_array((
            vec![1_i64, DET_NUM_POINTS as i64, DET_POINT_DIM as i64],
            data,
        ))
        .map_err(|e| format!("input tensor build failed: {e}"))?;
        let mut session = self
            .session
            .lock()
            .map_err(|_| "session mutex poisoned".to_string())?;
        let outputs = session
            .run(ort::inputs![tensor])
            .map_err(|e| format!("inference failed: {e}"))?;
        let (_, boxes) = outputs[0]
            .try_extract_tensor::<f32>()
            .map_err(|e| format!("boxes extract failed: {e}"))?;
        let (_, scores) = outputs[1]
            .try_extract_tensor::<f32>()
            .map_err(|e| format!("scores extract failed: {e}"))?;
        let classes: Option<Vec<i64>> = if outputs.len() >= 3 {
            outputs[2]
                .try_extract_tensor::<i64>()
                .ok()
                .map(|(_, c)| c.to_vec())
        } else {
            None
        };
        Ok(decode_detections(boxes, scores, classes.as_deref()))
    }
}

/// Decode flat `boxes`/`scores`/`classes` tensors into [`Detection3D`]s.
///
/// `boxes` is `100 × 7` row-major `[x, y, z, L, W, H, yaw]`; `scores` is the
/// per-detection confidence; `classes` (when present) is the per-detection
/// class index. A missing `classes` tensor yields `class_id = 0`.
#[cfg(feature = "onnx")]
fn decode_detections(boxes: &[f32], scores: &[f32], classes: Option<&[i64]>) -> Vec<Detection3D> {
    let n = (boxes.len() / DET_BOX_DIM)
        .min(scores.len())
        .min(DET_MAX_BOXES);
    let mut dets = Vec::with_capacity(n);
    for i in 0..n {
        let b = &boxes[i * DET_BOX_DIM..i * DET_BOX_DIM + DET_BOX_DIM];
        let class_id = classes
            .and_then(|c| c.get(i))
            .map(|&c| c.max(0) as u32)
            .unwrap_or(0);
        dets.push(Detection3D {
            position: [b[0] as f64, b[1] as f64, b[2] as f64],
            dimensions: [b[3] as f64, b[4] as f64, b[5] as f64],
            yaw: b[6] as f64,
            class_id,
            confidence: scores[i] as f64,
        });
    }
    dets
}

#[cfg(feature = "onnx")]
impl DetectionPipeline for OnnxDetector {
    fn detect(&self, input: &SensorInput) -> Vec<Detection3D> {
        let mut dets = match self.run_and_decode(input) {
            Ok(d) => d,
            Err(e) => {
                eprintln!("OnnxDetector inference failed: {e}");
                return Vec::new();
            }
        };
        filter_by_confidence(&mut dets, self.confidence_threshold);
        nms_3d(&mut dets, self.nms_iou_threshold);
        dets
    }

    fn name(&self) -> &str {
        "OnnxDetector"
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_detection(x: f64, y: f64, z: f64, size: f64, confidence: f64) -> Detection3D {
        Detection3D {
            position: [x, y, z],
            dimensions: [size, size, size],
            yaw: 0.0,
            class_id: 0,
            confidence,
        }
    }

    #[test]
    fn test_detection3d_construction() {
        let d = Detection3D {
            position: [1.0, 2.0, 3.0],
            dimensions: [4.0, 5.0, 6.0],
            yaw: 0.5,
            class_id: 7,
            confidence: 0.95,
        };
        assert_eq!(d.position, [1.0, 2.0, 3.0]);
        assert_eq!(d.dimensions, [4.0, 5.0, 6.0]);
        assert!((d.yaw - 0.5).abs() < f64::EPSILON);
        assert_eq!(d.class_id, 7);
        assert!((d.confidence - 0.95).abs() < f64::EPSILON);
    }

    #[test]
    fn test_sensor_input_construction() {
        let input = SensorInput {
            points: vec![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            intensities: Some(vec![0.8, 0.9]),
            timestamp: 100.0,
        };
        assert_eq!(input.points.len(), 2);
        assert_eq!(input.intensities.as_ref().unwrap().len(), 2);
        assert!((input.timestamp - 100.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_iou_3d_identical_boxes() {
        let a = make_detection(0.0, 0.0, 0.0, 2.0, 0.9);
        let b = a.clone();
        let iou = iou_3d(&a, &b);
        assert!(
            (iou - 1.0).abs() < 1e-10,
            "IoU of identical boxes should be 1.0, got {iou}"
        );
    }

    #[test]
    fn test_iou_3d_no_overlap() {
        let a = make_detection(0.0, 0.0, 0.0, 1.0, 0.9);
        let b = make_detection(100.0, 100.0, 100.0, 1.0, 0.9);
        let iou = iou_3d(&a, &b);
        assert!(
            (iou).abs() < 1e-10,
            "IoU of non-overlapping boxes should be 0.0, got {iou}"
        );
    }

    #[test]
    fn test_iou_3d_partial_overlap() {
        // Two unit cubes shifted by 0.5 in x only
        let a = make_detection(0.0, 0.0, 0.0, 1.0, 0.9);
        let b = make_detection(0.5, 0.0, 0.0, 1.0, 0.9);
        let iou = iou_3d(&a, &b);
        // Overlap in x: from 0.0 to 0.5 = 0.5, y: 1.0, z: 1.0 => intersection = 0.5
        // Union = 1.0 + 1.0 - 0.5 = 1.5
        // IoU = 0.5 / 1.5 = 1/3
        assert!(iou > 0.0, "IoU should be positive for overlapping boxes");
        assert!(
            iou < 1.0,
            "IoU should be less than 1.0 for non-identical boxes"
        );
        assert!(
            (iou - 1.0 / 3.0).abs() < 1e-10,
            "Expected IoU ~0.333, got {iou}"
        );
    }

    #[test]
    fn test_nms_removes_overlapping() {
        // Three detections: two overlapping at origin, one far away
        let mut dets = vec![
            make_detection(0.0, 0.0, 0.0, 2.0, 0.9),
            make_detection(0.1, 0.0, 0.0, 2.0, 0.7), // overlaps heavily with first
            make_detection(100.0, 100.0, 100.0, 2.0, 0.5), // far away
        ];
        nms_3d(&mut dets, 0.3);
        assert_eq!(dets.len(), 2, "NMS should remove 1 overlapping detection");
        // Highest confidence kept, plus the distant one
        assert!((dets[0].confidence - 0.9).abs() < f64::EPSILON);
        assert!((dets[1].confidence - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_nms_keeps_non_overlapping() {
        let mut dets = vec![
            make_detection(0.0, 0.0, 0.0, 1.0, 0.9),
            make_detection(100.0, 0.0, 0.0, 1.0, 0.8),
            make_detection(0.0, 100.0, 0.0, 1.0, 0.7),
        ];
        nms_3d(&mut dets, 0.5);
        assert_eq!(
            dets.len(),
            3,
            "NMS should keep all non-overlapping detections"
        );
    }

    #[test]
    fn test_mock_detector_returns_configured() {
        let expected = vec![
            make_detection(1.0, 2.0, 3.0, 1.0, 0.9),
            make_detection(4.0, 5.0, 6.0, 2.0, 0.8),
        ];
        let detector = MockDetector {
            detections: expected.clone(),
        };
        let input = SensorInput {
            points: vec![],
            intensities: None,
            timestamp: 0.0,
        };
        let result = detector.detect(&input);
        assert_eq!(result.len(), expected.len());
        assert_eq!(result[0].position, expected[0].position);
        assert_eq!(result[1].position, expected[1].position);
        assert_eq!(detector.name(), "MockDetector");
    }

    #[test]
    fn test_onnx_detector_config_default() {
        let config = OnnxDetectorConfig::default();
        assert!(config.model_path.is_empty());
        assert!((config.confidence_threshold - 0.5).abs() < f64::EPSILON);
        assert!((config.nms_iou_threshold - 0.4).abs() < f64::EPSILON);
        assert_eq!(config.voxel_size, [0.1, 0.1, 0.2]);
        assert_eq!(config.max_points_per_voxel, 35);
    }

    #[cfg(feature = "onnx")]
    fn model_path(name: &str) -> std::path::PathBuf {
        std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../test-data/models")
            .join(name)
    }

    /// Task 7.8: decoding defaults `class_id` to 0 when no classes tensor is
    /// present (backward-compatible 2-output models), and reads it otherwise.
    #[cfg(feature = "onnx")]
    #[test]
    fn decode_detections_defaults_class_to_zero() {
        let boxes = vec![
            1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 0.5, // det 0
            10.0, 11.0, 12.0, 2.0, 2.0, 2.0, 0.1, // det 1
        ];
        let scores = vec![0.9_f32, 0.8];
        let none = decode_detections(&boxes, &scores, None);
        assert_eq!(none.len(), 2);
        assert!(none.iter().all(|d| d.class_id == 0));
        assert_eq!(none[0].position, [1.0, 2.0, 3.0]);
        assert_eq!(none[0].dimensions, [4.0, 5.0, 6.0]);
        let some = decode_detections(&boxes, &scores, Some(&[3, 1]));
        assert_eq!(some[0].class_id, 3);
        assert_eq!(some[1].class_id, 1);
    }

    /// Task 8.1: the committed three-output detector stub loads and decodes
    /// into `Detection3D`s with class indices in `[0, 5)`.
    #[cfg(feature = "onnx")]
    #[test]
    fn detector_stub_three_output_contract() {
        let detector =
            OnnxDetector::load(model_path("test_detector.onnx").to_str().unwrap(), 0.0, 1.0)
                .expect("load detector stub");
        let input = SensorInput {
            points: (0..1000).map(|i| [i as f64 * 0.01, 0.0, 0.0]).collect(),
            intensities: Some(vec![0.5; 1000]),
            timestamp: 0.0,
        };
        let dets = detector.detect(&input);
        assert!(!dets.is_empty(), "stub should decode some detections");
        assert!(
            dets.len() <= 100,
            "at most 100 detections, got {}",
            dets.len()
        );
        assert!(
            dets.iter().all(|d| d.class_id < 5),
            "class index must be in [0, 5)"
        );
    }

    /// Task 8.2: the IMM-classifier stub maps `(1, 10, 12)` to 4 probabilities.
    #[cfg(feature = "onnx")]
    #[test]
    fn imm_classifier_stub_contract() {
        use crate::session::OnnxModel;
        let mut model =
            OnnxModel::load(model_path("imm_mode_classifier.onnx")).expect("load imm stub");
        let out = model
            .run_f32(&vec![0.0_f32; 10 * 12], &[1, 10, 12])
            .expect("run imm classifier");
        assert_eq!(out.len(), 4, "4 mode probabilities");
        let sum: f32 = out.iter().sum();
        assert!((sum - 1.0).abs() < 1e-4, "softmax sums to 1, got {sum}");
    }

    #[test]
    fn test_onnx_detector_config_custom() {
        let config = OnnxDetectorConfig {
            model_path: "/tmp/model.onnx".into(),
            confidence_threshold: 0.8,
            nms_iou_threshold: 0.5,
            voxel_size: [0.2, 0.2, 0.4],
            max_points_per_voxel: 64,
        };
        assert_eq!(config.model_path, "/tmp/model.onnx");
        assert!((config.confidence_threshold - 0.8).abs() < f64::EPSILON);
    }

    #[test]
    fn test_filter_by_confidence() {
        let mut dets = vec![
            make_detection(0.0, 0.0, 0.0, 1.0, 0.3),
            make_detection(1.0, 0.0, 0.0, 1.0, 0.5),
            make_detection(2.0, 0.0, 0.0, 1.0, 0.7),
            make_detection(3.0, 0.0, 0.0, 1.0, 0.9),
            make_detection(4.0, 0.0, 0.0, 1.0, 0.1),
        ];
        filter_by_confidence(&mut dets, 0.5);
        assert_eq!(
            dets.len(),
            3,
            "should keep 3 detections with confidence >= 0.5"
        );
        assert!(dets.iter().all(|d| d.confidence >= 0.5));
    }

    /// §5.6 Verify the SafeTensorsLoader can load test weights and that
    /// the WeightSet API works for detector-style access patterns.
    /// This test is non-feature-gated and exercises the SafeTensors path
    /// without requiring the `onnx` feature.
    #[test]
    fn test_safetensors_weight_loading_for_detector_integration() {
        use crate::weights::{SafeTensorsLoader, WeightLoader};
        let manifest_dir = env!("CARGO_MANIFEST_DIR");
        let path = format!("{manifest_dir}/../../test-data/models/test_weights.safetensors");
        let weights = SafeTensorsLoader.load(&path).unwrap();
        assert!(
            weights.get("encoder.layer.0.weight").is_some(),
            "encoder weight tensor should be present"
        );
        assert!(
            weights.get("decoder.head.weight").is_some(),
            "decoder head weight tensor should be present"
        );
        // Verify shapes are correct for detector use
        let enc = weights.get("encoder.layer.0.weight").unwrap();
        assert_eq!(enc.nrows(), 256);
        assert_eq!(enc.ncols(), 4);
        let dec = weights.get("decoder.head.weight").unwrap();
        assert_eq!(dec.nrows(), 7);
        assert_eq!(dec.ncols(), 256);
    }

    /// §4.1/4.2 OnnxDetector::reload_with_weights returns a documented error
    /// since ORT doesn't support runtime weight replacement.
    #[test]
    #[cfg(feature = "onnx")]
    #[ignore] // Requires ORT binary which is not available in CI
    fn test_onnx_reload_with_weights_returns_stub_error() {
        let config = OnnxDetectorConfig::default();
        let result = OnnxDetector::reload_with_weights(&config, "dummy.safetensors");
        assert!(result.is_err(), "should return an error (stub)");
    }

    #[test]
    fn test_detections_to_tracker_input() {
        let dets = vec![
            make_detection(1.0, 2.0, 3.0, 1.0, 0.9),
            make_detection(4.0, 5.0, 6.0, 1.0, 0.8),
        ];
        let vecs = detections_to_tracker_input(&dets);
        assert_eq!(vecs.len(), 2);
        assert_eq!(vecs[0].len(), 3);
        assert!((vecs[0][0] - 1.0).abs() < f64::EPSILON);
        assert!((vecs[0][1] - 2.0).abs() < f64::EPSILON);
        assert!((vecs[0][2] - 3.0).abs() < f64::EPSILON);
        assert!((vecs[1][0] - 4.0).abs() < f64::EPSILON);
    }
}
