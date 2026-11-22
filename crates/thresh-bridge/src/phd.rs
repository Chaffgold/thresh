//! Wrapper around Stone Soup's Gaussian Mixture PHD (Probability Hypothesis Density) filter.

use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::error::{BridgeError, BridgeResult};
use crate::runtime::{import_module, required_attribute, with_python};

/// Configuration for the Gaussian Mixture PHD filter.
#[derive(Debug, Clone)]
pub struct PhdConfig {
    /// Probability of survival between time steps.
    pub survival_probability: f64,
    /// Probability of detection by the sensor.
    pub detection_probability: f64,
    /// Clutter intensity (expected number of false alarms per unit volume).
    pub clutter_intensity: f64,
    /// Pruning weight threshold for Gaussian components.
    pub prune_threshold: f64,
    /// Merging distance threshold (Mahalanobis).
    pub merge_threshold: f64,
    /// Maximum number of Gaussian components to retain.
    pub max_components: usize,
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
        }
    }
}

/// Gaussian Mixture PHD filter backed by Stone Soup.
pub struct PhdFilter {
    /// The Python GM-PHD updater object.
    py_updater: Py<PyAny>,
    /// Configuration stored for reference.
    config: PhdConfig,
}

impl PhdFilter {
    /// Create a new GM-PHD filter.
    ///
    /// `predictor` and `updater` are Stone Soup predictor/updater Python objects
    /// that define the underlying single-target dynamics and measurement model.
    pub fn new(
        py: Python<'_>,
        config: PhdConfig,
        predictor: &Bound<'_, PyAny>,
        updater: &Bound<'_, PyAny>,
    ) -> BridgeResult<Self> {
        let error = |error| BridgeError::from_python(py, error, "PHD construction");
        let ss_phd = import_module(py, "stonesoup.updater.pointprocess")?;

        let kwargs = PyDict::new(py);
        kwargs.set_item("updater", updater).map_err(error)?;
        kwargs.set_item("predictor", predictor).map_err(error)?;
        kwargs
            .set_item("prob_survival", config.survival_probability)
            .map_err(error)?;
        kwargs
            .set_item("prob_detect", config.detection_probability)
            .map_err(error)?;
        kwargs
            .set_item("clutter_intensity", config.clutter_intensity)
            .map_err(error)?;
        kwargs
            .set_item("prune_threshold", config.prune_threshold)
            .map_err(error)?;
        kwargs
            .set_item("merge_threshold", config.merge_threshold)
            .map_err(error)?;
        kwargs
            .set_item("max_components", config.max_components)
            .map_err(error)?;

        let phd_cls = required_attribute(ss_phd.as_any(), "PHDUpdater", "PHD updater lookup")?;
        let phd_updater = phd_cls.call((), Some(&kwargs)).map_err(error)?;

        Ok(Self {
            py_updater: phd_updater.unbind(),
            config,
        })
    }

    /// Run one PHD update step with a set of detections.
    ///
    /// `prior` is the prior Gaussian mixture intensity (Stone Soup type).
    /// `detections` is a Python set of `Detection` objects.
    /// Returns the posterior Gaussian mixture intensity.
    pub fn filter(&self, prior: &Py<PyAny>, detections: &Py<PyAny>) -> BridgeResult<Py<PyAny>> {
        with_python("PHD update", |py| {
            let method =
                required_attribute(self.py_updater.bind(py), "update", "PHD update lookup")?;
            method
                .call1((prior.bind(py), detections.bind(py)))
                .map(Bound::unbind)
                .map_err(|error| BridgeError::from_python(py, error, "PHD update"))
        })
    }

    /// Get the current configuration.
    pub fn config(&self) -> &PhdConfig {
        &self.config
    }
}
