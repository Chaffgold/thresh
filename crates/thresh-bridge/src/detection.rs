//! Context-aware conversion into complete Stone Soup detections.
#![doc = include_str!("../../../docs/bridge-detections.md")]

use nalgebra::DVector;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use thresh_core::measurement::Measurement;

use crate::context::{CommonFrame, DetectionContext};
use crate::convert::{dmatrix_to_numpy, dvector_to_numpy, numpy_to_dmatrix, numpy_to_dvector};
use crate::error::{BridgeError, BridgeResult};
use crate::observation::{self, Model, Observation};
use crate::runtime::{import_module, required_attribute, with_python};
use crate::time::ObservationTime;

/// Convert a measurement with explicit time, geometry, datum, and noise context.
///
/// The host must initialize Python first (for a Rust executable, call
/// `pyo3::Python::initialize()` at startup). This function never initializes it.
/// Unsupported variants and invalid Rust-side input are rejected before Python
/// access. Successful output has a float64 column vector, UTC observation time,
/// and a measurement model carrying the transformed covariance.
///
/// This replaces the old token-taking, context-free converter. Boundary success
/// does not imply that the JPDA or GM-PHD reference wrappers are operational.
/// The bridge provides no MHT or IMM reference; those contracts are native.
pub fn measurement_to_detection(
    measurement: &Measurement,
    context: &DetectionContext,
) -> BridgeResult<Py<PyAny>> {
    reject_unsupported(measurement)?;
    let time = context.validate(measurement)?;
    let observation = observation::normalize(measurement, context)?;
    with_python("measurement conversion", |py| {
        construct_detection(py, measurement, context, &time, &observation)
    })
}

fn reject_unsupported(measurement: &Measurement) -> BridgeResult<()> {
    match measurement {
        Measurement::Radar { .. } | Measurement::EoIr { .. } | Measurement::AdsB { .. } => Ok(()),
        _ => Err(BridgeError::ConversionError(
            "unsupported measurement: only Radar, EoIr, and AdsB have bridge profiles".into(),
        )),
    }
}

fn construct_detection(
    py: Python<'_>,
    measurement: &Measurement,
    context: &DetectionContext,
    time: &ObservationTime,
    observation: &Observation,
) -> BridgeResult<Py<PyAny>> {
    import_module(py, "stonesoup")?;
    import_module(py, "numpy")?;
    let detection_module = import_module(py, "stonesoup.types.detection")?;
    let detection_class = required_attribute(&detection_module, "Detection", "load Detection")?;
    let model = construct_model(py, observation)?;
    let kwargs = detection_kwargs(py, measurement, context, time, observation, &model)
        .map_err(|e| BridgeError::from_python(py, e, "build detection arguments"))?;
    let detection = detection_class
        .call((), Some(&kwargs))
        .map_err(|e| BridgeError::from_python(py, e, "construct Detection"))?;
    validate_detection(py, &detection, observation, time)?;
    Ok(detection.unbind())
}

fn construct_model<'py>(
    py: Python<'py>,
    observation: &Observation,
) -> BridgeResult<Bound<'py, PyAny>> {
    let (module, class) = match &observation.model {
        Model::Spherical { class, .. } => ("stonesoup.models.measurement.nonlinear", *class),
        Model::Cartesian { .. } => ("stonesoup.models.measurement.linear", "LinearGaussian"),
    };
    let module = import_module(py, module)?;
    let class = required_attribute(&module, class, "load observation model")?;
    let kwargs = model_kwargs(py, observation)
        .map_err(|e| BridgeError::from_python(py, e, "build model arguments"))?;
    class
        .call((), Some(&kwargs))
        .map_err(|e| BridgeError::from_python(py, e, "construct observation model"))
}

fn model_kwargs<'py>(py: Python<'py>, observation: &Observation) -> PyResult<Bound<'py, PyDict>> {
    let kwargs = PyDict::new(py);
    kwargs.set_item("ndim_state", 6)?;
    kwargs.set_item(
        "noise_covar",
        dmatrix_to_numpy(py, &observation.covariance)?,
    )?;
    match &observation.model {
        Model::Spherical {
            position,
            rotation,
            doppler,
            ..
        } => {
            kwargs.set_item("mapping", (0, 2, 4))?;
            kwargs.set_item(
                "translation_offset",
                column(py, &DVector::from_column_slice(position.as_slice()))?,
            )?;
            kwargs.set_item(
                "rotation_offset",
                column(py, &DVector::from_column_slice(rotation.as_slice()))?,
            )?;
            if *doppler {
                kwargs.set_item("velocity_mapping", (1, 3, 5))?;
                kwargs.set_item("velocity", column(py, &DVector::zeros(3))?)?;
            }
        }
        Model::Cartesian { mapping } => kwargs.set_item("mapping", mapping.clone())?,
    }
    Ok(kwargs)
}

fn column<'py>(py: Python<'py>, values: &DVector<f64>) -> PyResult<Bound<'py, PyAny>> {
    dvector_to_numpy(py, values)?
        .into_bound(py)
        .call_method1("reshape", ((values.len(), 1),))
}

fn detection_kwargs<'py>(
    py: Python<'py>,
    measurement: &Measurement,
    context: &DetectionContext,
    time: &ObservationTime,
    observation: &Observation,
    model: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyDict>> {
    let kwargs = PyDict::new(py);
    kwargs.set_item("state_vector", column(py, &observation.values)?)?;
    kwargs.set_item("timestamp", python_timestamp(py, time)?)?;
    kwargs.set_item("measurement_model", model)?;
    kwargs.set_item("metadata", metadata(py, measurement, context, time)?)?;
    Ok(kwargs)
}

fn python_timestamp<'py>(py: Python<'py>, time: &ObservationTime) -> PyResult<Bound<'py, PyAny>> {
    let datetime = py.import("datetime")?;
    let utc = datetime.getattr("timezone")?.getattr("utc")?;
    let c = &time.calendar;
    datetime.getattr("datetime")?.call1((
        c.year,
        c.month,
        c.day,
        c.hour,
        c.minute,
        c.second,
        c.microsecond,
        utc,
    ))
}

fn metadata<'py>(
    py: Python<'py>,
    measurement: &Measurement,
    context: &DetectionContext,
    time: &ObservationTime,
) -> PyResult<Bound<'py, PyDict>> {
    let metadata = PyDict::new(py);
    metadata.set_item("time", time.relative_seconds)?;
    metadata.set_item("origin_utc", &time.origin_utc)?;
    metadata.set_item("observation_utc_unrounded", &time.observation_utc)?;
    metadata.set_item("arrival_utc", &time.arrival_utc)?;
    metadata.set_item("common_frame", "WGS84 local ENU")?;
    if let CommonFrame::Wgs84Enu(origin) = &context.frame {
        metadata.set_item(
            "frame_origin",
            (
                origin.latitude_rad,
                origin.longitude_rad,
                origin.ellipsoid_height_m,
            ),
        )?;
    }
    set_source_metadata(&metadata, measurement)?;
    Ok(metadata)
}

fn set_source_metadata(metadata: &Bound<'_, PyDict>, measurement: &Measurement) -> PyResult<()> {
    match measurement {
        Measurement::Radar {
            range,
            azimuth,
            elevation,
            range_rate,
            sensor_id,
            ..
        } => {
            metadata.set_item("sensor_type", "radar")?;
            metadata.set_item("range", range)?;
            metadata.set_item("azimuth", azimuth)?;
            metadata.set_item("elevation", elevation)?;
            metadata.set_item("range_rate", range_rate)?;
            metadata.set_item("sensor_id", sensor_id)?;
        }
        Measurement::EoIr {
            azimuth,
            elevation,
            sensor_id,
            ..
        } => {
            metadata.set_item("sensor_type", "eoir")?;
            metadata.set_item("azimuth", azimuth)?;
            metadata.set_item("elevation", elevation)?;
            metadata.set_item("sensor_id", sensor_id)?;
        }
        Measurement::AdsB {
            lat,
            lon,
            alt,
            velocity,
            ..
        } => {
            metadata.set_item("sensor_type", "adsb")?;
            metadata.set_item("provenance", "cooperative_report")?;
            metadata.set_item("lat", lat)?;
            metadata.set_item("lon", lon)?;
            metadata.set_item("alt_msl", alt)?;
            metadata.set_item("velocity", velocity)?;
        }
        _ => unreachable!("unsupported variants rejected before Python access"),
    }
    Ok(())
}

fn validate_detection(
    py: Python<'_>,
    detection: &Bound<'_, PyAny>,
    observation: &Observation,
    time: &ObservationTime,
) -> BridgeResult<()> {
    let vector = required_attribute(detection, "state_vector", "check Detection")?;
    let shape: (usize, usize) = vector
        .getattr("shape")
        .and_then(|s| s.extract())
        .map_err(|e| BridgeError::from_python(py, e, "check column shape"))?;
    if shape != (observation.values.len(), 1)
        || numpy_to_dvector(py, &vector)? != observation.values
    {
        return Err(BridgeError::ConversionError(
            "Detection changed observation shape/values".into(),
        ));
    }
    validate_model(py, detection, observation)?;
    let actual = required_attribute(detection, "timestamp", "check Detection")?;
    let equal = python_timestamp(py, time)
        .and_then(|expected| actual.eq(expected))
        .map_err(|e| BridgeError::from_python(py, e, "check observation timestamp"))?;
    if !equal {
        return Err(BridgeError::ConversionError(
            "Detection changed observation timestamp".into(),
        ));
    }
    Ok(())
}

fn validate_model(
    py: Python<'_>,
    detection: &Bound<'_, PyAny>,
    observation: &Observation,
) -> BridgeResult<()> {
    let model = required_attribute(detection, "measurement_model", "check Detection")?;
    let state = required_attribute(&model, "ndim_state", "check model dimensions")?;
    let measurement = required_attribute(&model, "ndim_meas", "check model dimensions")?;
    let mapping_attribute = required_attribute(&model, "mapping", "check model mapping")?;
    let shape = (|| -> PyResult<(usize, usize, Vec<usize>)> {
        Ok((
            state.extract()?,
            measurement.extract()?,
            mapping_attribute.extract()?,
        ))
    })()
    .map_err(|e| BridgeError::from_python(py, e, "read model dimensions/mapping"))?;
    let mapping = match &observation.model {
        Model::Spherical { .. } => vec![0, 2, 4],
        Model::Cartesian { mapping } => mapping.clone(),
    };
    if shape != (6, observation.values.len(), mapping) {
        return Err(BridgeError::ConversionError(
            "observation model mapping/dimensions changed".into(),
        ));
    }
    let covariance = required_attribute(&model, "noise_covar", "check model covariance")?;
    if numpy_to_dmatrix(py, &covariance)? != observation.covariance {
        return Err(BridgeError::ConversionError(
            "observation model covariance changed".into(),
        ));
    }
    validate_model_pose(py, &model, &observation.model)?;
    Ok(())
}

fn validate_model_pose(
    py: Python<'_>,
    model: &Bound<'_, PyAny>,
    expected: &Model,
) -> BridgeResult<()> {
    let Model::Spherical {
        position,
        rotation,
        doppler,
        ..
    } = expected
    else {
        return Ok(());
    };
    let translation = required_attribute(model, "translation_offset", "check sensor pose")?;
    let rotation_offset = required_attribute(model, "rotation_offset", "check sensor pose")?;
    if numpy_to_dvector(py, &translation)?.as_slice() != position.as_slice()
        || numpy_to_dvector(py, &rotation_offset)?.as_slice() != rotation.as_slice()
    {
        return Err(BridgeError::ConversionError(
            "observation model sensor pose changed".into(),
        ));
    }
    if *doppler {
        validate_doppler_model(py, model)?;
    }
    Ok(())
}

fn validate_doppler_model(py: Python<'_>, model: &Bound<'_, PyAny>) -> BridgeResult<()> {
    let mapping = required_attribute(model, "velocity_mapping", "check Doppler mapping")?;
    let mapping = mapping
        .extract::<Vec<usize>>()
        .map_err(|e| BridgeError::from_python(py, e, "read Doppler mapping"))?;
    let velocity = required_attribute(model, "velocity", "check stationary sensor")?;
    if mapping != [1, 3, 5] || numpy_to_dvector(py, &velocity)? != DVector::zeros(3) {
        return Err(BridgeError::ConversionError(
            "observation model stationary velocity/mapping changed".into(),
        ));
    }
    Ok(())
}
