//! Explicit Stone Soup Gaussian-mixture PHD scan composition.

use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::error::{BridgeError, BridgeResult};
use crate::runtime::{import_module, required_attribute, with_python};

mod validation;

/// Configuration for the pinned Gaussian Mixture PHD reference.
#[derive(Debug, Clone)]
pub struct PhdConfig {
    /// Finite probability of survival in [0, 1]. Also applies to detected births.
    pub survival_probability: f64,
    /// Finite probability of detection in [0, 1].
    pub detection_probability: f64,
    /// Positive clutter spatial density in inverse measurement-volume units.
    pub clutter_intensity: f64,
    /// Nonnegative pruning weight threshold. Removed weight is redistributed.
    pub prune_threshold: f64,
    /// Nonnegative squared Mahalanobis merge threshold (strict comparison).
    /// Pairwise merged weights are capped at one by the pinned reducer.
    pub merge_threshold: f64,
    /// Positive component cap. Truncated weight is redistributed to survivors.
    pub max_components: usize,
    /// Optional positive squared measurement-distance gate; `None` is unbounded.
    /// This gate is independent of the mixture merge threshold.
    pub gating_threshold_squared: Option<f64>,
}

impl Default for PhdConfig {
    fn default() -> Self {
        Self {
            survival_probability: 0.99,
            detection_probability: 0.9,
            clutter_intensity: 1e-5,
            prune_threshold: 1e-5,
            merge_threshold: 4.0,
            max_components: 100,
            gating_threshold_squared: None,
        }
    }
}

impl PhdConfig {
    fn validate(&self) -> BridgeResult<()> {
        validate_scalar("survival_probability", self.survival_probability, |x| {
            (0.0..=1.0).contains(&x)
        })?;
        validate_scalar("detection_probability", self.detection_probability, |x| {
            (0.0..=1.0).contains(&x)
        })?;
        validate_scalar("clutter_intensity", self.clutter_intensity, |x| x > 0.0)?;
        validate_scalar("prune_threshold", self.prune_threshold, |x| x >= 0.0)?;
        validate_scalar("merge_threshold", self.merge_threshold, |x| x >= 0.0)?;
        if let Some(threshold) = self.gating_threshold_squared {
            validate_scalar("gating_threshold_squared", threshold, |x| x > 0.0)?;
        }
        if self.max_components == 0 {
            return invalid("max_components must be positive");
        }
        Ok(())
    }
}

fn validate_scalar(name: &str, value: f64, domain: impl FnOnce(f64) -> bool) -> BridgeResult<()> {
    if !value.is_finite() || !domain(value) {
        return invalid(&format!("invalid {name}: {value}"));
    }
    Ok(())
}

fn invalid<T>(message: &str) -> BridgeResult<T> {
    Err(BridgeError::AlgorithmError(format!("PHD: {message}")))
}

/// A stateless scan composition using caller-selected predictor/updater models.
///
/// This is a bounded linear-Gaussian reference, not a tracker. The pinned runtime
/// omits missed births, applies survival to detected births, replaces exact zero
/// updater weights with float64 epsilon, and can lose intensity through merging.
pub struct PhdFilter {
    py_hypothesiser: Py<PyAny>,
    py_updater: Py<PyAny>,
    py_reducer: Py<PyAny>,
    config: PhdConfig,
}

impl PhdFilter {
    /// Compose the distance/mixture hypothesiser, intensity updater and reducer.
    ///
    /// The host owns Python initialization. Supplied single-target components
    /// retain their identity and must use compatible dynamics/measurement models.
    /// Invalid numeric configuration fails before importing any Python modules.
    pub fn new(
        py: Python<'_>,
        config: PhdConfig,
        predictor: &Bound<'_, PyAny>,
        updater: &Bound<'_, PyAny>,
    ) -> BridgeResult<Self> {
        config.validate()?;
        let py_hypothesiser = build_hypothesiser(py, &config, predictor, updater)?;
        let py_updater = build_updater(py, &config, updater)?;
        let py_reducer = build_reducer(py, &config)?;
        Ok(Self {
            py_hypothesiser,
            py_updater,
            py_reducer,
            config,
        })
    }

    /// Predict, update and reduce one scan with explicit birth intensity.
    ///
    /// Priors/births are repeatable collections of tagged Gaussian states with finite
    /// nonnegative weights and compatible finite float64 state/covariance arrays.
    /// Birth tags must be `birth`; prior tags must not. All times must be aware
    /// UTC datetimes. Priors share a time no later than the scan; birth/detection
    /// times equal the scan. Detections are also a repeatable collection and share
    /// one compatible measurement space. One-shot iterators are rejected.
    /// Per-detection model dimensions are checked when supplied; compatibility
    /// of an updater-owned fallback measurement model is the caller's responsibility.
    ///
    /// One deep copy of the combined component graph protects caller arrays and
    /// reachable ancestry before prediction. Predictor/updater objects are not
    /// copied. The returned GaussianMixtureUpdate retains hypothesis provenance;
    /// an empty result has component-derived timestamp `None`.
    pub fn filter(
        &self,
        prior: &Py<PyAny>,
        birth_components: &Py<PyAny>,
        detections: &Py<PyAny>,
        timestamp: &Py<PyAny>,
    ) -> BridgeResult<Py<PyAny>> {
        with_python("PHD scan", |py| {
            let components = validation::validate_scan(
                py,
                prior.bind(py),
                birth_components.bind(py),
                detections.bind(py),
                timestamp.bind(py),
            )?;
            let working = copy_components(py, components.as_any())?;
            let hypotheses = self.hypothesise(py, &working, detections, timestamp)?;
            let update = call_step(
                self.py_updater.bind(py),
                "update",
                &hypotheses,
                "PHD update",
            )?;
            self.reduce_update(py, &update)?;
            Ok(update.unbind())
        })
    }

    fn hypothesise<'py>(
        &self,
        py: Python<'py>,
        components: &Bound<'py, PyAny>,
        detections: &Py<PyAny>,
        timestamp: &Py<PyAny>,
    ) -> BridgeResult<Bound<'py, PyAny>> {
        let method = required_attribute(
            self.py_hypothesiser.bind(py),
            "hypothesise",
            "PHD hypothesis lookup",
        )?;
        method
            .call1((components, detections.bind(py), timestamp.bind(py)))
            .map_err(|error| BridgeError::from_python(py, error, "PHD hypothesis"))
    }

    fn reduce_update(&self, py: Python<'_>, update: &Bound<'_, PyAny>) -> BridgeResult<()> {
        let components = required_attribute(update, "components", "PHD components lookup")?;
        let reduced = call_step(
            self.py_reducer.bind(py),
            "reduce",
            &components,
            "PHD reduction",
        )?;
        update
            .setattr("components", reduced)
            .map_err(|error| BridgeError::from_python(py, error, "PHD result"))
    }

    /// Get the validated configuration.
    pub fn config(&self) -> &PhdConfig {
        &self.config
    }
}

fn construct<'py>(
    py: Python<'py>,
    module: &str,
    class: &str,
    kwargs: &Bound<'py, PyDict>,
    operation: &str,
) -> BridgeResult<Bound<'py, PyAny>> {
    let module = import_module(py, module)?;
    required_attribute(module.as_any(), class, &format!("{operation} lookup"))?
        .call((), Some(kwargs))
        .map_err(|error| BridgeError::from_python(py, error, operation))
}

fn build_hypothesiser(
    py: Python<'_>,
    config: &PhdConfig,
    predictor: &Bound<'_, PyAny>,
    updater: &Bound<'_, PyAny>,
) -> BridgeResult<Py<PyAny>> {
    let operation = "PHD hypothesiser construction";
    let error = |error| BridgeError::from_python(py, error, operation);
    let measure = construct(
        py,
        "stonesoup.measures",
        "SquaredMahalanobis",
        &PyDict::new(py),
        operation,
    )?;
    let kwargs = PyDict::new(py);
    kwargs.set_item("predictor", predictor).map_err(error)?;
    kwargs.set_item("updater", updater).map_err(error)?;
    kwargs.set_item("measure", measure).map_err(error)?;
    kwargs
        .set_item(
            "missed_distance",
            config.gating_threshold_squared.unwrap_or(f64::INFINITY),
        )
        .map_err(error)?;
    let distance = construct(
        py,
        "stonesoup.hypothesiser.distance",
        "DistanceHypothesiser",
        &kwargs,
        operation,
    )?;
    let kwargs = PyDict::new(py);
    kwargs.set_item("hypothesiser", distance).map_err(error)?;
    kwargs.set_item("order_by_detection", true).map_err(error)?;
    construct(
        py,
        "stonesoup.hypothesiser.gaussianmixture",
        "GaussianMixtureHypothesiser",
        &kwargs,
        operation,
    )
    .map(Bound::unbind)
}

fn build_updater(
    py: Python<'_>,
    config: &PhdConfig,
    updater: &Bound<'_, PyAny>,
) -> BridgeResult<Py<PyAny>> {
    let operation = "PHD updater construction";
    let error = |error| BridgeError::from_python(py, error, operation);
    let kwargs = PyDict::new(py);
    kwargs.set_item("updater", updater).map_err(error)?;
    kwargs
        .set_item("clutter_spatial_density", config.clutter_intensity)
        .map_err(error)?;
    kwargs
        .set_item("prob_detection", config.detection_probability)
        .map_err(error)?;
    kwargs
        .set_item("prob_survival", config.survival_probability)
        .map_err(error)?;
    kwargs.set_item("normalisation", true).map_err(error)?;
    construct(
        py,
        "stonesoup.updater.pointprocess",
        "PHDUpdater",
        &kwargs,
        operation,
    )
    .map(Bound::unbind)
}

fn build_reducer(py: Python<'_>, config: &PhdConfig) -> BridgeResult<Py<PyAny>> {
    let operation = "PHD reducer construction";
    let error = |error| BridgeError::from_python(py, error, operation);
    let kwargs = PyDict::new(py);
    kwargs
        .set_item("prune_threshold", config.prune_threshold)
        .map_err(error)?;
    kwargs
        .set_item("merge_threshold", config.merge_threshold)
        .map_err(error)?;
    kwargs
        .set_item("max_number_components", config.max_components)
        .map_err(error)?;
    construct(
        py,
        "stonesoup.mixturereducer.gaussianmixture",
        "GaussianMixtureReducer",
        &kwargs,
        operation,
    )
    .map(Bound::unbind)
}

fn copy_components<'py>(
    py: Python<'py>,
    components: &Bound<'py, PyAny>,
) -> BridgeResult<Bound<'py, PyAny>> {
    let copy = import_module(py, "copy")?;
    required_attribute(copy.as_any(), "deepcopy", "PHD component copy lookup")?
        .call1((components,))
        .map_err(|error| BridgeError::from_python(py, error, "PHD component copy"))
}

fn call_step<'py>(
    object: &Bound<'py, PyAny>,
    method: &str,
    input: &Bound<'py, PyAny>,
    operation: &str,
) -> BridgeResult<Bound<'py, PyAny>> {
    required_attribute(object, method, &format!("{operation} lookup"))?
        .call1((input,))
        .map_err(|error| BridgeError::from_python(object.py(), error, operation))
}

#[cfg(test)]
mod tests;
