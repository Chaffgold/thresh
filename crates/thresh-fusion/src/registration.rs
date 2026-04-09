//! Sensor registration: transform measurements to common frame.

use nalgebra::{DVector, Vector3};
use thresh_core::coords::spherical_to_cartesian;
use thresh_core::sensor::SensorRegistration;

/// Convert a radar measurement (range, azimuth, elevation) to Cartesian in the common frame.
pub fn radar_to_common_frame(
    range: f64,
    azimuth: f64,
    elevation: f64,
    sensor: &SensorRegistration,
) -> Vector3<f64> {
    let local_cart = spherical_to_cartesian(range, azimuth, elevation);
    sensor.to_common_frame(&local_cart)
}

/// Convert a radar measurement to a Cartesian position measurement vector.
pub fn radar_to_cartesian_measurement(
    range: f64,
    azimuth: f64,
    elevation: f64,
    sensor: &SensorRegistration,
) -> DVector<f64> {
    let pos = radar_to_common_frame(range, azimuth, elevation, sensor);
    DVector::from_column_slice(&[pos.x, pos.y, pos.z])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn radar_at_origin_along_x() {
        let sensor = SensorRegistration::new(Vector3::zeros(), 0);
        let pos = radar_to_common_frame(1000.0, 0.0, 0.0, &sensor);
        assert!((pos.x - 1000.0).abs() < 1e-8);
        assert!(pos.y.abs() < 1e-8);
        assert!(pos.z.abs() < 1e-8);
    }

    #[test]
    fn radar_with_offset_sensor() {
        let sensor = SensorRegistration::new(Vector3::new(500.0, 0.0, 0.0), 1);
        let pos = radar_to_common_frame(1000.0, 0.0, 0.0, &sensor);
        assert!((pos.x - 1500.0).abs() < 1e-8);
    }
}
