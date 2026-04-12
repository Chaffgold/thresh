//! Conversion utilities between nalgebra types and numpy arrays.

use nalgebra::{DMatrix, DVector};
use pyo3::prelude::*;
use pyo3::types::PyList;

use crate::error::{BridgeError, BridgeResult};

/// Convert a nalgebra `DVector<f64>` to a 1-D numpy array.
pub fn dvector_to_numpy(py: Python<'_>, v: &DVector<f64>) -> PyResult<Py<PyAny>> {
    let np = py.import("numpy")?;
    let list = PyList::new(py, v.iter().copied())?;
    let arr = np.call_method1("array", (list,))?;
    Ok(arr.unbind())
}

/// Convert a nalgebra `DMatrix<f64>` to a 2-D numpy array (row-major).
pub fn dmatrix_to_numpy(py: Python<'_>, m: &DMatrix<f64>) -> PyResult<Py<PyAny>> {
    let np = py.import("numpy")?;
    let rows = m.nrows();
    let cols = m.ncols();
    // Build as list-of-lists then convert.
    let outer = PyList::empty(py);
    for i in 0..rows {
        let row: Vec<f64> = (0..cols).map(|j| m[(i, j)]).collect();
        let py_row = PyList::new(py, &row)?;
        outer.append(py_row)?;
    }
    let arr = np.call_method1("array", (outer,))?;
    Ok(arr.unbind())
}

/// Convert a 1-D numpy array to a nalgebra `DVector<f64>`.
pub fn numpy_to_dvector(_py: Python<'_>, arr: &Bound<'_, PyAny>) -> BridgeResult<DVector<f64>> {
    let flat = arr
        .call_method0("flatten")
        .map_err(|e| BridgeError::ConversionError(format!("failed to flatten array: {e}")))?;
    let list = flat
        .call_method0("tolist")
        .map_err(|e| BridgeError::ConversionError(format!("failed to convert to list: {e}")))?;
    let values: Vec<f64> = list
        .extract()
        .map_err(|e| BridgeError::ConversionError(format!("failed to extract f64 list: {e}")))?;
    Ok(DVector::from_column_slice(&values))
}

/// Convert a 2-D numpy array to a nalgebra `DMatrix<f64>` (row-major).
pub fn numpy_to_dmatrix(_py: Python<'_>, arr: &Bound<'_, PyAny>) -> BridgeResult<DMatrix<f64>> {
    let shape = arr
        .getattr("shape")
        .map_err(|e| BridgeError::ConversionError(format!("array has no shape: {e}")))?;
    let dims: Vec<usize> = shape
        .extract()
        .map_err(|e| BridgeError::ConversionError(format!("failed to extract shape: {e}")))?;
    if dims.len() != 2 {
        return Err(BridgeError::ConversionError(format!(
            "expected 2-D array, got {}-D",
            dims.len()
        )));
    }
    let rows = dims[0];
    let cols = dims[1];
    let flat = arr
        .call_method0("flatten")
        .and_then(|f| f.call_method0("tolist"))
        .map_err(|e| BridgeError::ConversionError(format!("flatten/tolist failed: {e}")))?;
    let values: Vec<f64> = flat
        .extract()
        .map_err(|e| BridgeError::ConversionError(format!("failed to extract f64 list: {e}")))?;
    // numpy flatten is row-major; DMatrix::from_row_slice expects the same.
    Ok(DMatrix::from_row_slice(rows, cols, &values))
}
