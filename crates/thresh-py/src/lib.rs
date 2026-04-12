//! Python bindings for the thresh tracking framework.

use pyo3::prelude::*;

mod eval;
mod tracker;

/// The `thresh_py` Python module.
#[pymodule]
fn thresh_py(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<tracker::PyMultiObjectTracker>()?;
    m.add_class::<tracker::PyTrackState>()?;
    m.add_function(wrap_pyfunction!(eval::compute_mot_metrics_py, m)?)?;
    Ok(())
}
