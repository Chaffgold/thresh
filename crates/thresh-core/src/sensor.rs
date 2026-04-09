//! Sensor registration and coordinate frame management.

use nalgebra::{Matrix3, Vector3};
use serde::{Deserialize, Serialize};

/// Sensor position, orientation, and coordinate transform.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SensorRegistration {
    /// Sensor position in the common frame [x, y, z] in meters.
    pub position: Vector3<f64>,
    /// Rotation matrix from sensor frame to common frame.
    pub rotation: Matrix3<f64>,
    /// Unique sensor identifier.
    pub sensor_id: u32,
}

impl SensorRegistration {
    /// Create a sensor at the given position with an identity (aligned) orientation.
    pub fn new(position: Vector3<f64>, sensor_id: u32) -> Self {
        Self {
            position,
            rotation: Matrix3::identity(),
            sensor_id,
        }
    }

    /// Create a sensor with explicit rotation.
    pub fn with_rotation(position: Vector3<f64>, rotation: Matrix3<f64>, sensor_id: u32) -> Self {
        Self {
            position,
            rotation,
            sensor_id,
        }
    }

    /// Transform a point from sensor frame to common frame.
    pub fn to_common_frame(&self, sensor_point: &Vector3<f64>) -> Vector3<f64> {
        self.rotation * sensor_point + self.position
    }

    /// Transform a point from common frame to sensor frame.
    pub fn to_sensor_frame(&self, common_point: &Vector3<f64>) -> Vector3<f64> {
        self.rotation.transpose() * (common_point - self.position)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_sensor_roundtrip() {
        let sensor = SensorRegistration::new(Vector3::new(100.0, 200.0, 50.0), 0);
        let common = Vector3::new(500.0, 600.0, 100.0);
        let sensor_pt = sensor.to_sensor_frame(&common);
        let back = sensor.to_common_frame(&sensor_pt);
        assert!((back - common).norm() < 1e-10);
    }

    #[test]
    fn offset_sensor() {
        let sensor = SensorRegistration::new(Vector3::new(1000.0, 0.0, 0.0), 1);
        let origin_in_sensor = sensor.to_sensor_frame(&Vector3::zeros());
        assert!((origin_in_sensor.x - (-1000.0)).abs() < 1e-10);
    }
}
