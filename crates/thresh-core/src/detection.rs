//! 3D detection output types.

use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Sensor input
// ---------------------------------------------------------------------------

/// Multi-modal sensor input for detection pipelines.
#[derive(Debug, Clone)]
pub enum SensorInputType {
    /// 3D point cloud (e.g. from LiDAR or radar).
    PointCloud {
        /// Points as `[x, y, z]` in sensor coordinates.
        points: Vec<[f64; 3]>,
        /// Optional per-point intensity values.
        intensities: Option<Vec<f64>>,
    },
    /// Image tensor in NCHW layout.
    ImageTensor {
        /// Flattened pixel data (typically normalized).
        data: Vec<f32>,
        /// Shape `[batch, channels, height, width]`.
        shape: [usize; 4],
    },
}

// ---------------------------------------------------------------------------
// Detection errors
// ---------------------------------------------------------------------------

/// Errors produced by detection pipelines.
#[derive(Debug)]
pub enum DetectionError {
    /// Failed to load the detection model.
    ModelLoad(String),
    /// Inference execution failed.
    Inference(String),
    /// Tensor shape mismatch between expected and actual.
    ShapeMismatch {
        /// Expected dimensions.
        expected: Vec<usize>,
        /// Actual dimensions.
        got: Vec<usize>,
    },
}

impl std::fmt::Display for DetectionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DetectionError::ModelLoad(msg) => write!(f, "model load error: {msg}"),
            DetectionError::Inference(msg) => write!(f, "inference error: {msg}"),
            DetectionError::ShapeMismatch { expected, got } => {
                write!(f, "shape mismatch: expected {expected:?}, got {got:?}")
            }
        }
    }
}

impl std::error::Error for DetectionError {}

/// 3D oriented bounding box from a detection pipeline.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BoundingBox3D {
    /// Center x position.
    pub x: f64,
    /// Center y position.
    pub y: f64,
    /// Center z position.
    pub z: f64,
    /// Length (extent along heading direction).
    pub length: f64,
    /// Width (extent perpendicular to heading).
    pub width: f64,
    /// Height (vertical extent).
    pub height: f64,
    /// Yaw angle in radians.
    pub yaw: f64,
    /// Detection confidence score [0, 1].
    pub score: f64,
    /// Predicted class index.
    pub class_id: u32,
    /// Predicted velocity [vx, vy] if available.
    pub velocity: Option<[f64; 2]>,
}

/// High-level detection output combining a 3D bounding box with optional
/// embedding features for re-identification and a frame identifier.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Detection3D {
    /// Center position `[x, y, z]`.
    pub position: [f64; 3],
    /// Dimensions `[length, width, height]`.
    pub dimensions: [f64; 3],
    /// Heading angle in radians.
    pub yaw: f64,
    /// Predicted class index.
    pub class_id: u32,
    /// Detection confidence score in `[0, 1]`.
    pub confidence: f64,
}

impl Detection3D {
    /// Create a `Detection3D` from a [`BoundingBox3D`].
    pub fn from_bbox(bbox: &BoundingBox3D) -> Self {
        Self {
            position: bbox.center(),
            dimensions: [bbox.length, bbox.width, bbox.height],
            yaw: bbox.yaw,
            class_id: bbox.class_id,
            confidence: bbox.score,
        }
    }

    /// Convert to a [`nalgebra::DVector<f64>`] containing `[x, y, z]` for
    /// use as a Kalman-filter measurement.
    pub fn to_measurement(&self) -> nalgebra::DVector<f64> {
        nalgebra::DVector::from_column_slice(&self.position)
    }

    /// Volume of the detection bounding box.
    pub fn volume(&self) -> f64 {
        self.dimensions[0] * self.dimensions[1] * self.dimensions[2]
    }
}

impl BoundingBox3D {
    /// Volume of the bounding box.
    pub fn volume(&self) -> f64 {
        self.length * self.width * self.height
    }

    /// Center position as an array.
    pub fn center(&self) -> [f64; 3] {
        [self.x, self.y, self.z]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sensor_input_type_point_cloud() {
        let input = SensorInputType::PointCloud {
            points: vec![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            intensities: Some(vec![0.8, 0.9]),
        };
        match &input {
            SensorInputType::PointCloud {
                points,
                intensities,
            } => {
                assert_eq!(points.len(), 2);
                assert_eq!(intensities.as_ref().unwrap().len(), 2);
            }
            _ => panic!("expected PointCloud variant"),
        }
    }

    #[test]
    fn test_sensor_input_type_image_tensor() {
        let input = SensorInputType::ImageTensor {
            data: vec![0.0; 3 * 224 * 224],
            shape: [1, 3, 224, 224],
        };
        match &input {
            SensorInputType::ImageTensor { data, shape } => {
                assert_eq!(data.len(), 3 * 224 * 224);
                assert_eq!(*shape, [1, 3, 224, 224]);
            }
            _ => panic!("expected ImageTensor variant"),
        }
    }

    #[test]
    fn test_detection_error_display() {
        let e = DetectionError::ModelLoad("file not found".into());
        assert!(e.to_string().contains("model load error"));

        let e = DetectionError::Inference("timeout".into());
        assert!(e.to_string().contains("inference error"));

        let e = DetectionError::ShapeMismatch {
            expected: vec![1, 3, 640, 640],
            got: vec![1, 3, 320, 320],
        };
        let msg = e.to_string();
        assert!(msg.contains("shape mismatch"));
        assert!(msg.contains("640"));
        assert!(msg.contains("320"));
    }

    #[test]
    fn test_detection_error_is_error() {
        let e = DetectionError::Inference("test".into());
        // Verify it implements std::error::Error
        let _: &dyn std::error::Error = &e;
    }

    #[test]
    fn bounding_box_volume() {
        let bb = BoundingBox3D {
            x: 0.0,
            y: 0.0,
            z: 0.0,
            length: 5.0,
            width: 2.0,
            height: 1.5,
            yaw: 0.0,
            score: 0.9,
            class_id: 0,
            velocity: None,
        };
        assert!((bb.volume() - 15.0).abs() < 1e-10);
    }

    #[test]
    fn bounding_box_center() {
        let bb = BoundingBox3D {
            x: 10.0,
            y: 20.0,
            z: 30.0,
            length: 1.0,
            width: 1.0,
            height: 1.0,
            yaw: 0.0,
            score: 0.5,
            class_id: 1,
            velocity: Some([100.0, 50.0]),
        };
        assert_eq!(bb.center(), [10.0, 20.0, 30.0]);
    }
}
