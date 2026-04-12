//! Placeholder for scenario execution bindings.
//!
//! The full scenario API will accept a configuration dictionary and return
//! track results from a synthetic scenario run. For now, use the `thresh-data`
//! CLI for scenario execution and load results via `thresh_py.compute_mot_metrics`.

use pyo3::prelude::*;

use crate::errors::thresh_err;

/// Run a synthetic tracking scenario.
///
/// **Not yet implemented.** Use the `thresh-data` CLI for scenario execution
/// and load results via `thresh_py.compute_mot_metrics`.
#[pyfunction]
pub fn run_scenario(_config: &Bound<'_, PyAny>) -> PyResult<Vec<PyObject>> {
    Err(thresh_err(
        "not yet implemented: use thresh-data CLI for scenario execution",
    ))
}
