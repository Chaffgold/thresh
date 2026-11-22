//! Strict conversion between nalgebra and NumPy, preserving logical element order.
//!
//! Only finite, nonempty float64 arrays are supported. Generic vectors are emitted
//! as `(n,)`; detection construction is responsible for making column StateVectors.
//! Validated ndarray subclasses are viewed as base arrays without dtype coercion.
//! Observation covariance has additional checks: its dimension must match the
//! profile, and symmetry and positive semidefiniteness use
//! `1e-12 * max(1, max_row_sum(abs(R)))`, evaluated with overflow-safe scaling.
//! Accepted covariance is never symmetrized or given jitter. Singular PSD noise
//! can pass boundary validation but need not be invertible by a later updater.

use nalgebra::{DMatrix, DVector};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use crate::error::{BridgeError, BridgeResult};

/// Convert a finite, nonempty vector to a one-dimensional float64 NumPy array.
///
/// The host must already own an initialized interpreter. Invalid values or an
/// empty vector produce `ValueError`; import/execution errors remain Python errors.
pub fn dvector_to_numpy(py: Python<'_>, v: &DVector<f64>) -> PyResult<Py<PyAny>> {
    validate_output(v.len(), 1, v.iter().copied())?;
    let values = PyList::new(py, v.iter().copied())?;
    array_float64(py, &values)
}

/// Convert a finite, nonempty matrix to a two-dimensional float64 NumPy array.
///
/// Elements are explicitly serialized in row order, independent of nalgebra's
/// column-major storage. Empty dimensions and nonfinite entries produce `ValueError`.
pub fn dmatrix_to_numpy(py: Python<'_>, m: &DMatrix<f64>) -> PyResult<Py<PyAny>> {
    validate_output(m.nrows(), m.ncols(), m.iter().copied())?;
    let rows = PyList::empty(py);
    for row in 0..m.nrows() {
        rows.append(PyList::new(py, m.row(row).iter().copied())?)?;
    }
    array_float64(py, &rows)
}

fn array_float64(py: Python<'_>, values: &Bound<'_, PyList>) -> PyResult<Py<PyAny>> {
    let np = py.import("numpy")?;
    let options = PyDict::new(py);
    options.set_item("dtype", np.getattr("float64")?)?;
    options.set_item("order", "C")?;
    Ok(np
        .getattr("array")?
        .call((values,), Some(&options))?
        .unbind())
}

fn validate_output(rows: usize, cols: usize, values: impl Iterator<Item = f64>) -> PyResult<()> {
    checked_count(rows, cols).map_err(|e| PyValueError::new_err(e.to_string()))?;
    finite_values(values).map_err(|e| PyValueError::new_err(e.to_string()))
}

/// Convert a nonempty float64 NumPy vector `(n,)` or column `(n,1)` to nalgebra.
///
/// Strides and negative strides preserve logical order. Row vectors with more
/// than one element, general matrices, other ranks/dtypes, and nonfinite entries
/// are rejected rather than flattened or silently coerced.
pub fn numpy_to_dvector(py: Python<'_>, arr: &Bound<'_, PyAny>) -> BridgeResult<DVector<f64>> {
    let shape = array_shape(py, arr)?;
    let len = match shape.as_slice() {
        [len] | [len, 1] => checked_count(*len, 1)?,
        _ => return conversion_error("expected vector shape (n,) or (n,1)"),
    };
    let values = logical_values(py, arr, len)?;
    Ok(DVector::from_vec(values))
}

/// Convert a finite, nonempty two-dimensional float64 NumPy matrix to nalgebra.
///
/// C-order extraction preserves every logical row/column for strided, transposed,
/// and Fortran-order arrays. Shape and allocation size are checked before copying.
pub fn numpy_to_dmatrix(py: Python<'_>, arr: &Bound<'_, PyAny>) -> BridgeResult<DMatrix<f64>> {
    let shape = array_shape(py, arr)?;
    let [rows, cols] = shape.as_slice() else {
        return conversion_error("expected two-dimensional matrix");
    };
    let count = checked_count(*rows, *cols)?;
    let values = logical_values(py, arr, count)?;
    Ok(DMatrix::from_row_slice(*rows, *cols, &values))
}

fn array_shape(py: Python<'_>, arr: &Bound<'_, PyAny>) -> BridgeResult<Vec<usize>> {
    let np = py
        .import("numpy")
        .map_err(|e| BridgeError::from_python(py, e, "import NumPy for array validation"))?;
    let ndarray = np
        .getattr("ndarray")
        .map_err(|e| BridgeError::from_python(py, e, "resolve NumPy ndarray"))?;
    let is_array = arr
        .is_instance(&ndarray)
        .map_err(|e| BridgeError::from_python(py, e, "check NumPy array type"))?;
    if !is_array {
        return conversion_error("expected a NumPy ndarray, not an array-like value");
    }
    validate_dtype(py, arr)?;
    arr.getattr("shape")
        .and_then(|shape| shape.extract())
        .map_err(|e| BridgeError::from_python(py, e, "extract NumPy array shape"))
}

fn validate_dtype(py: Python<'_>, arr: &Bound<'_, PyAny>) -> BridgeResult<()> {
    let dtype = arr
        .getattr("dtype")
        .map_err(|e| BridgeError::from_python(py, e, "inspect NumPy dtype"))?;
    let kind: String = dtype
        .getattr("kind")
        .and_then(|value| value.extract())
        .map_err(|e| BridgeError::from_python(py, e, "inspect NumPy dtype kind"))?;
    let itemsize: usize = dtype
        .getattr("itemsize")
        .and_then(|value| value.extract())
        .map_err(|e| BridgeError::from_python(py, e, "inspect NumPy dtype size"))?;
    if kind != "f" || itemsize != size_of::<f64>() {
        return conversion_error("expected float64 dtype (no implicit coercion)");
    }
    Ok(())
}

fn logical_values(
    py: Python<'_>,
    arr: &Bound<'_, PyAny>,
    expected: usize,
) -> BridgeResult<Vec<f64>> {
    // Subclasses such as np.matrix override ravel to retain two dimensions.
    // Shape/dtype validation has already run; omit dtype here to avoid coercion.
    let base = py
        .import("numpy")
        .and_then(|np| np.call_method1("asarray", (arr,)))
        .map_err(|e| BridgeError::from_python(py, e, "normalize validated ndarray subclass"))?;
    validate_dtype(py, &base)?;
    let values: Vec<f64> = base
        .call_method1("ravel", ("C",))
        .and_then(|flat| flat.call_method0("tolist"))
        .and_then(|list| list.extract())
        .map_err(|e| BridgeError::from_python(py, e, "extract logical C-order float64 values"))?;
    if values.len() != expected {
        return conversion_error("NumPy shape and extracted element count disagree");
    }
    finite_values(values.iter().copied())?;
    Ok(values)
}

fn checked_count(rows: usize, cols: usize) -> BridgeResult<usize> {
    if rows == 0 || cols == 0 {
        return conversion_error("array dimensions must be nonempty");
    }
    let count = rows
        .checked_mul(cols)
        .ok_or_else(|| BridgeError::ConversionError("array element count overflow".into()))?;
    let bytes = count
        .checked_mul(size_of::<f64>())
        .ok_or_else(|| BridgeError::ConversionError("array allocation size overflow".into()))?;
    if bytes > isize::MAX as usize {
        return conversion_error("array allocation exceeds the Rust addressable limit");
    }
    Ok(count)
}

fn finite_values(mut values: impl Iterator<Item = f64>) -> BridgeResult<()> {
    if values.any(|value| !value.is_finite()) {
        return conversion_error("array entries must be finite");
    }
    Ok(())
}

fn conversion_error<T>(message: &str) -> BridgeResult<T> {
    Err(BridgeError::ConversionError(message.into()))
}

#[cfg(test)]
mod tests {
    use super::checked_count;

    #[test]
    fn shape_size_checks_reject_element_byte_and_address_overflow() {
        assert!(checked_count(usize::MAX, 2).is_err());
        assert!(checked_count(usize::MAX / 4, 1).is_err());
        assert!(checked_count(isize::MAX as usize / 8 + 1, 1).is_err());
        assert_eq!(checked_count(2, 3).unwrap(), 6);
    }
}
