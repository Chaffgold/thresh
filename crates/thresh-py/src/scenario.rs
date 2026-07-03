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
pub fn run_scenario(_config: &Bound<'_, PyAny>) -> PyResult<Vec<Py<PyAny>>> {
    Err(thresh_err(
        "not yet implemented: use thresh-data CLI for scenario execution",
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use pyo3::types::PyDict;

    /// The stub must keep rejecting with a pointer at the supported path.
    /// Embeds the interpreter directly (no maturin needed), so it runs in
    /// the workspace test lane.
    #[test]
    fn test_run_scenario_not_implemented() {
        Python::initialize();
        Python::attach(|py| {
            let config = PyDict::new(py);
            let err = run_scenario(config.as_any()).expect_err("stub must error");
            assert!(
                err.to_string().contains("not yet implemented"),
                "unexpected error message: {err}"
            );
        });
    }
}
