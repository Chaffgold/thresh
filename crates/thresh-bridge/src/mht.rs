//! Wrapper around Stone Soup's MHT (Multiple Hypothesis Tracking) tracker.

use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::error::{BridgeError, BridgeResult};

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
        Python::with_gil(|py| {
            let ss_tracker = py
                .import("stonesoup.tracker.simple")
                .map_err(BridgeError::from)?;

            let kwargs = PyDict::new(py);
            kwargs.set_item("max_num_hypotheses", config.max_hypotheses)?;
            kwargs.set_item("prune_threshold", config.prune_threshold)?;

            let tracker_cls = ss_tracker.getattr("MultiTargetMixtureTracker")?;
            let tracker = tracker_cls.call((), Some(&kwargs))?;

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
        Python::with_gil(|py| {
            let result = self
                .py_tracker
                .call_method1(py, "track", (detections.bind(py), timestamp.bind(py)))
                .map_err(BridgeError::from)?;
            Ok(result)
        })
    }
}
