//! Wrapper around Stone Soup's MHT (Multiple Hypothesis Tracking) tracker.

use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::error::{BridgeError, BridgeResult};
use crate::runtime::{import_module, required_attribute, with_python};

/// Configuration for the MHT tracker.
#[derive(Debug, Clone)]
pub struct MhtConfig {
    /// Maximum number of hypotheses to maintain.
    pub max_hypotheses: usize,
    /// Pruning threshold (log-likelihood ratio).
    pub prune_threshold: f64,
}

impl Default for MhtConfig {
    fn default() -> Self {
        Self {
            max_hypotheses: 100,
            prune_threshold: -10.0,
        }
    }
}

/// MHT tracker backed by Stone Soup.
pub struct MhtTracker {
    /// The Python MHT tracker object.
    py_tracker: Py<PyAny>,
}

impl MhtTracker {
    /// Create a new MHT tracker.
    pub fn new(config: &MhtConfig) -> BridgeResult<Self> {
        with_python("MHT construction", |py| {
            let error = |error| BridgeError::from_python(py, error, "MHT construction");
            let ss_tracker = import_module(py, "stonesoup.tracker.simple")?;

            let kwargs = PyDict::new(py);
            kwargs
                .set_item("max_num_hypotheses", config.max_hypotheses)
                .map_err(error)?;
            kwargs
                .set_item("prune_threshold", config.prune_threshold)
                .map_err(error)?;

            let tracker_cls = required_attribute(
                ss_tracker.as_any(),
                "MultiTargetMixtureTracker",
                "MHT tracker lookup",
            )?;
            let tracker = tracker_cls.call((), Some(&kwargs)).map_err(error)?;

            Ok(Self {
                py_tracker: tracker.unbind(),
            })
        })
    }

    /// Run one step of the MHT tracker with new detections.
    ///
    /// `detections` is a Python set of Stone Soup `Detection` objects.
    /// `timestamp` is a Python datetime.
    /// Returns the updated set of tracks.
    pub fn run(&self, detections: &Py<PyAny>, timestamp: &Py<PyAny>) -> BridgeResult<Py<PyAny>> {
        with_python("MHT tracking", |py| {
            let method = required_attribute(self.py_tracker.bind(py), "track", "MHT track lookup")?;
            method
                .call1((detections.bind(py), timestamp.bind(py)))
                .map(Bound::unbind)
                .map_err(|error| BridgeError::from_python(py, error, "MHT tracking"))
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::step_test_support::assert_step_contract;

    #[test]
    fn initialized_run_preserves_dispatch_and_errors() {
        assert_step_contract(
            "track",
            "MHT tracking",
            "MHT track lookup",
            2,
            |py_tracker| MhtTracker { py_tracker },
            |wrapper, arguments| wrapper.run(&arguments[0], &arguments[1]),
        );
    }
}
