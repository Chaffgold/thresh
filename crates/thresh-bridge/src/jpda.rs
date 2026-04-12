//! Wrapper around Stone Soup's JPDA (Joint Probabilistic Data Association) data associator.

use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::error::{BridgeError, BridgeResult};

/// Configuration for the JPDA data associator.
#[derive(Debug, Clone)]
pub struct JpdaConfig {
    /// Gate probability (default 0.99).
    pub gate_probability: f64,
    /// Clutter spatial density (default 1e-6).
    pub clutter_spatial_density: f64,
}

impl Default for JpdaConfig {
    fn default() -> Self {
        Self {
            gate_probability: 0.99,
            clutter_spatial_density: 1e-6,
        }
    }
}

/// JPDA data associator backed by Stone Soup.
pub struct JpdaAssociator {
    /// The Python JPDA object, held across calls.
    py_associator: Py<PyAny>,
}

impl JpdaAssociator {
    /// Create a new JPDA associator.
    ///
    /// Initialises the underlying Stone Soup `JPDAHypothesiser` and
    /// `GNNWith2DAssignment` data associator.
    pub fn new(config: &JpdaConfig) -> BridgeResult<Self> {
        Python::with_gil(|py| {
            let ss_hyp = py
                .import("stonesoup.hypothesiser.probability")
                .map_err(BridgeError::from)?;
            let ss_assoc = py
                .import("stonesoup.dataassociator.probability")
                .map_err(BridgeError::from)?;

            // Build hypothesiser.
            let hyp_kwargs = PyDict::new(py);
            hyp_kwargs.set_item("gate_probability", config.gate_probability)?;
            hyp_kwargs.set_item("clutter_spatial_density", config.clutter_spatial_density)?;
            let hyp_cls = ss_hyp.getattr("PDAHypothesiser")?;
            let hypothesiser = hyp_cls.call((), Some(&hyp_kwargs))?;

            // Build JPDA associator.
            let assoc_kwargs = PyDict::new(py);
            assoc_kwargs.set_item("hypothesiser", hypothesiser)?;
            let assoc_cls = ss_assoc.getattr("JPDA")?;
            let associator = assoc_cls.call((), Some(&assoc_kwargs))?;

            Ok(Self {
                py_associator: associator.unbind(),
            })
        })
    }

    /// Run the JPDA association on a set of tracks and detections.
    ///
    /// `tracks` and `detections` are Python objects (Stone Soup types).
    /// Returns the association mapping as a Python dict.
    pub fn associate(
        &self,
        tracks: &Py<PyAny>,
        detections: &Py<PyAny>,
        timestamp: &Py<PyAny>,
    ) -> BridgeResult<Py<PyAny>> {
        Python::with_gil(|py| {
            let result = self
                .py_associator
                .call_method1(
                    py,
                    "associate",
                    (tracks.bind(py), detections.bind(py), timestamp.bind(py)),
                )
                .map_err(BridgeError::from)?;
            Ok(result)
        })
    }
}
