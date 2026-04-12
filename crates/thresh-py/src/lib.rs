//! Python bindings for the thresh tracking framework.

use pyo3::prelude::*;

pub mod conversions;
pub mod errors;
mod eval;
mod filter;
mod scenario;
mod tracker;

/// The `thresh_py` Python module.
#[pymodule]
fn thresh_py(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Tracker
    m.add_class::<tracker::PyMultiObjectTracker>()?;
    m.add_class::<tracker::PyTrackState>()?;

    // Filter
    m.add_class::<filter::PyKalmanFilter>()?;

    // Eval
    m.add_function(wrap_pyfunction!(eval::compute_mot_metrics_py, m)?)?;

    // Scenario (placeholder)
    m.add_function(wrap_pyfunction!(scenario::run_scenario, m)?)?;

    // Custom exception
    m.add("ThreshError", m.py().get_type::<errors::ThreshError>())?;

    Ok(())
}
