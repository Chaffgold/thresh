//! Convert `thresh_core::measurement::Measurement` into Stone Soup `Detection` objects.

use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::convert::dvector_to_numpy;
use crate::error::BridgeResult;
use thresh_core::measurement::Measurement;

/// Convert a `Measurement` to a Stone Soup `Detection` Python object.
///
/// The resulting object is a `stonesoup.types.detection.Detection` with:
/// - `state_vector`: numpy column vector built from `Measurement::to_vector()`
/// - `metadata`: dict containing sensor type and sensor-specific fields
pub fn measurement_to_detection(
    py: Python<'_>,
    measurement: &Measurement,
) -> BridgeResult<Py<PyAny>> {
    let ss_detection = py.import("stonesoup.types.detection")?;
    let np = py.import("numpy")?;

    // Build state vector as a numpy column vector.
    let z = measurement.to_vector();
    let arr = dvector_to_numpy(py, &z)?;
    let state_vector = np.call_method1("reshape", (arr.bind(py), (-1i32, 1i32)))?;

    // Build metadata dict.
    let metadata = PyDict::new(py);
    match measurement {
        Measurement::Radar {
            range,
            azimuth,
            elevation,
            range_rate,
            time,
            sensor_id,
        } => {
            metadata.set_item("sensor_type", "radar")?;
            metadata.set_item("range", *range)?;
            metadata.set_item("azimuth", *azimuth)?;
            metadata.set_item("elevation", *elevation)?;
            metadata.set_item("range_rate", *range_rate)?;
            metadata.set_item("time", *time)?;
            metadata.set_item("sensor_id", *sensor_id)?;
        }
        Measurement::EoIr {
            azimuth,
            elevation,
            time,
            sensor_id,
        } => {
            metadata.set_item("sensor_type", "eoir")?;
            metadata.set_item("azimuth", *azimuth)?;
            metadata.set_item("elevation", *elevation)?;
            metadata.set_item("time", *time)?;
            metadata.set_item("sensor_id", *sensor_id)?;
        }
        Measurement::AdsB {
            lat,
            lon,
            alt,
            velocity,
            time,
        } => {
            metadata.set_item("sensor_type", "adsb")?;
            metadata.set_item("lat", *lat)?;
            metadata.set_item("lon", *lon)?;
            metadata.set_item("alt", *alt)?;
            metadata.set_item("velocity", *velocity)?;
            metadata.set_item("time", *time)?;
        }
    }

    let kwargs = PyDict::new(py);
    kwargs.set_item("state_vector", state_vector)?;
    kwargs.set_item("metadata", metadata)?;

    let detection_cls = ss_detection.getattr("Detection")?;
    let det = detection_cls.call((), Some(&kwargs))?;
    Ok(det.unbind())
}
