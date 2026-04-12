//! 3D detection output types.

use serde::{Deserialize, Serialize};

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
