//! Attached input checks; contract violations never reach hypothesis generation.

use pyo3::exceptions::{PyAttributeError, PyTypeError};
use pyo3::prelude::*;
use pyo3::types::{PyIterator, PyList};

use super::invalid;
use crate::convert::{numpy_to_dmatrix, numpy_to_dvector};
use crate::error::{BridgeError, BridgeResult};
use crate::runtime::{import_module, required_attribute};

fn python_error(error: PyErr, py: Python<'_>) -> BridgeError {
    BridgeError::from_python(py, error, "PHD input validation")
}

fn input_attribute<'py>(object: &Bound<'py, PyAny>, name: &str) -> BridgeResult<Bound<'py, PyAny>> {
    object.getattr(name).map_err(|error| {
        if error.is_instance_of::<PyAttributeError>(object.py()) {
            BridgeError::AlgorithmError(format!("PHD: missing input {name}"))
        } else {
            python_error(error, object.py())
        }
    })
}

fn input_iterator<'py>(object: &Bound<'py, PyAny>) -> BridgeResult<Bound<'py, PyIterator>> {
    let iterator = object.try_iter().map_err(|error| {
        if error.is_instance_of::<PyTypeError>(object.py()) {
            BridgeError::AlgorithmError("PHD: inputs must be iterable collections".into())
        } else {
            python_error(error, object.py())
        }
    })?;
    if iterator.as_any().is(object) {
        return invalid("inputs must be repeatable collections, not one-shot iterators");
    }
    Ok(iterator)
}

fn array_contract<T>(result: BridgeResult<T>) -> BridgeResult<T> {
    result.map_err(|error| match error {
        BridgeError::ConversionError(message) => {
            BridgeError::AlgorithmError(format!("PHD: {message}"))
        }
        other => other,
    })
}

struct ScanValidator<'py> {
    component_type: Bound<'py, PyAny>,
    detection_type: Bound<'py, PyAny>,
    datetime_type: Bound<'py, PyAny>,
    zero_offset: Bound<'py, PyAny>,
    timestamp: Bound<'py, PyAny>,
    state_dimension: Option<usize>,
    detection_dimension: Option<usize>,
    prior_time: Option<Bound<'py, PyAny>>,
}

impl<'py> ScanValidator<'py> {
    fn new(py: Python<'py>, timestamp: &Bound<'py, PyAny>) -> BridgeResult<Self> {
        let states = import_module(py, "stonesoup.types.state")?;
        let detections = import_module(py, "stonesoup.types.detection")?;
        let datetime = import_module(py, "datetime")?;
        let component_type = required_attribute(
            states.as_any(),
            "TaggedWeightedGaussianState",
            "PHD input type lookup",
        )?;
        let detection_type =
            required_attribute(detections.as_any(), "Detection", "PHD input type lookup")?;
        let datetime_type =
            required_attribute(datetime.as_any(), "datetime", "PHD datetime lookup")?;
        let zero_offset =
            required_attribute(datetime.as_any(), "timedelta", "PHD datetime lookup")?
                .call0()
                .map_err(|error| python_error(error, py))?;
        let validator = Self {
            component_type,
            detection_type,
            datetime_type,
            zero_offset,
            timestamp: timestamp.clone(),
            state_dimension: None,
            detection_dimension: None,
            prior_time: None,
        };
        validator.validate_time(timestamp)?;
        Ok(validator)
    }

    fn validate_time(&self, time: &Bound<'py, PyAny>) -> BridgeResult<()> {
        let error = |error| python_error(error, time.py());
        if !time.is_instance(&self.datetime_type).map_err(error)? {
            return invalid("required times must be timezone-aware UTC datetimes");
        }
        let offset = time.call_method0("utcoffset").map_err(error)?;
        if offset.is_none() || !offset.eq(&self.zero_offset).map_err(error)? {
            return invalid("required times must be timezone-aware UTC datetimes");
        }
        Ok(())
    }

    fn validate_component(
        &mut self,
        component: &Bound<'py, PyAny>,
        birth: bool,
    ) -> BridgeResult<()> {
        let error = |error| python_error(error, component.py());
        if !component.is_instance(&self.component_type).map_err(error)? {
            return invalid("components must be TaggedWeightedGaussianState instances");
        }
        validate_tag_and_weight(component, birth)?;
        let state = input_attribute(component, "state_vector")?;
        let state = array_contract(numpy_to_dvector(component.py(), &state))?;
        let covar = input_attribute(component, "covar")?;
        let covar = array_contract(numpy_to_dmatrix(component.py(), &covar))?;
        if covar.nrows() != state.len() || covar.ncols() != state.len() {
            return invalid("component covariance must match its state dimension");
        }
        compatible_dimension(&mut self.state_dimension, state.len(), "component state")?;
        let timestamp = input_attribute(component, "timestamp")?;
        self.validate_component_time(&timestamp, birth)
    }

    fn validate_component_time(
        &mut self,
        time: &Bound<'py, PyAny>,
        birth: bool,
    ) -> BridgeResult<()> {
        self.validate_time(time)?;
        let error = |error| python_error(error, time.py());
        if birth {
            return self.validate_scan_time(time);
        }
        if time.gt(&self.timestamp).map_err(error)? {
            return invalid("prior time must not be later than the scan");
        }
        if let Some(prior_time) = &self.prior_time
            && !time.eq(prior_time).map_err(error)?
        {
            return invalid("prior components must share a timestamp");
        }
        self.prior_time = Some(time.clone());
        Ok(())
    }

    fn validate_scan_time(&self, time: &Bound<'py, PyAny>) -> BridgeResult<()> {
        self.validate_time(time)?;
        if !time
            .eq(&self.timestamp)
            .map_err(|error| python_error(error, time.py()))?
        {
            return invalid("birth and detection times must equal the scan timestamp");
        }
        Ok(())
    }

    fn validate_detection(&mut self, detection: &Bound<'py, PyAny>) -> BridgeResult<()> {
        let error = |error| python_error(error, detection.py());
        if !detection.is_instance(&self.detection_type).map_err(error)? {
            return invalid("detections must be Detection instances");
        }
        let timestamp = input_attribute(detection, "timestamp")?;
        self.validate_scan_time(&timestamp)?;
        let state = input_attribute(detection, "state_vector")?;
        let state = array_contract(numpy_to_dvector(detection.py(), &state))?;
        compatible_dimension(&mut self.detection_dimension, state.len(), "detection")?;
        let model = input_attribute(detection, "measurement_model")?;
        self.validate_measurement_dimensions(&model, state.len())
    }

    fn validate_measurement_dimensions(
        &self,
        model: &Bound<'py, PyAny>,
        dimension: usize,
    ) -> BridgeResult<()> {
        // A missing per-detection model is supported: the caller's updater owns it.
        if model.is_none() {
            return Ok(());
        }
        let error = |error| python_error(error, model.py());
        let measurement_dimension: usize = input_attribute(model, "ndim_meas")?
            .extract()
            .map_err(error)?;
        let state_dimension: usize = input_attribute(model, "ndim_state")?
            .extract()
            .map_err(error)?;
        if measurement_dimension != dimension
            || self
                .state_dimension
                .is_some_and(|size| size != state_dimension)
        {
            return invalid("measurement model dimensions must match detections and components");
        }
        Ok(())
    }
}

fn validate_tag_and_weight(component: &Bound<'_, PyAny>, birth: bool) -> BridgeResult<()> {
    let tag = input_attribute(component, "tag")?;
    let tag: String = tag
        .extract()
        .map_err(|_| BridgeError::AlgorithmError("PHD: component tag must be a string".into()))?;
    if (tag == "birth") != birth {
        return invalid("only explicit birth components must carry the reserved birth tag");
    }
    let weight = input_attribute(component, "weight")?;
    let weight: f64 = weight.extract::<f64>().map_err(|error| {
        if error.is_instance_of::<PyTypeError>(component.py()) {
            BridgeError::AlgorithmError("PHD: component weight must be numeric".into())
        } else {
            python_error(error, component.py())
        }
    })?;
    if !weight.is_finite() || weight < 0.0 {
        return invalid("component weights must be finite and nonnegative");
    }
    Ok(())
}

fn compatible_dimension(
    expected: &mut Option<usize>,
    actual: usize,
    label: &str,
) -> BridgeResult<()> {
    if expected.is_some_and(|size| size != actual) {
        return invalid(&format!("incompatible {label} dimensions"));
    }
    *expected = Some(actual);
    Ok(())
}

fn append_components<'py>(
    validator: &mut ScanValidator<'py>,
    input: &Bound<'py, PyIterator>,
    components: &Bound<'py, PyList>,
    birth: bool,
) -> BridgeResult<()> {
    for component in input.clone() {
        let component = component.map_err(|error| python_error(error, input.py()))?;
        validator.validate_component(&component, birth)?;
        components
            .append(component)
            .map_err(|error| python_error(error, input.py()))?;
    }
    Ok(())
}

pub(super) fn validate_scan<'py>(
    py: Python<'py>,
    prior: &Bound<'py, PyAny>,
    births: &Bound<'py, PyAny>,
    detections: &Bound<'py, PyAny>,
    timestamp: &Bound<'py, PyAny>,
) -> BridgeResult<Bound<'py, PyList>> {
    // Check every collection before advancing any iterator. In particular, a
    // late invalid collection must not consume an earlier one-shot input.
    let prior = input_iterator(prior)?;
    let births = input_iterator(births)?;
    let detections = input_iterator(detections)?;
    let mut validator = ScanValidator::new(py, timestamp)?;
    let components = PyList::empty(py);
    append_components(&mut validator, &prior, &components, false)?;
    append_components(&mut validator, &births, &components, true)?;
    for detection in detections {
        let detection = detection.map_err(|error| python_error(error, py))?;
        validator.validate_detection(&detection)?;
    }
    Ok(components)
}
