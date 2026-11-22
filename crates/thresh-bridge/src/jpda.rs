//! Wrapper around Stone Soup's JPDA (Joint Probabilistic Data Association) data associator.

use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::error::{BridgeError, BridgeResult};
use crate::runtime::{import_module, required_attribute, with_python};

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
        with_python("JPDA construction", |py| {
            let error = |error| BridgeError::from_python(py, error, "JPDA construction");
            let ss_hyp = import_module(py, "stonesoup.hypothesiser.probability")?;
            let ss_assoc = import_module(py, "stonesoup.dataassociator.probability")?;

            // Build hypothesiser.
            let hyp_kwargs = PyDict::new(py);
            hyp_kwargs
                .set_item("gate_probability", config.gate_probability)
                .map_err(error)?;
            hyp_kwargs
                .set_item("clutter_spatial_density", config.clutter_spatial_density)
                .map_err(error)?;
            let hyp_cls = required_attribute(
                ss_hyp.as_any(),
                "PDAHypothesiser",
                "JPDA hypothesiser lookup",
            )?;
            let hypothesiser = hyp_cls.call((), Some(&hyp_kwargs)).map_err(error)?;

            // Build JPDA associator.
            let assoc_kwargs = PyDict::new(py);
            assoc_kwargs
                .set_item("hypothesiser", hypothesiser)
                .map_err(error)?;
            let assoc_cls =
                required_attribute(ss_assoc.as_any(), "JPDA", "JPDA associator lookup")?;
            let associator = assoc_cls.call((), Some(&assoc_kwargs)).map_err(error)?;

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
        with_python("JPDA association", |py| {
            let method = required_attribute(
                self.py_associator.bind(py),
                "associate",
                "JPDA association lookup",
            )?;
            method
                .call1((tracks.bind(py), detections.bind(py), timestamp.bind(py)))
                .map(Bound::unbind)
                .map_err(|error| BridgeError::from_python(py, error, "JPDA association"))
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::step_test_support::assert_step_contract;

    #[test]
    fn initialized_associate_preserves_dispatch_and_errors() {
        assert_step_contract(
            "associate",
            "JPDA association",
            "JPDA association lookup",
            3,
            |py_associator| JpdaAssociator { py_associator },
            |wrapper, arguments| wrapper.associate(&arguments[0], &arguments[1], &arguments[2]),
        );
    }
}
