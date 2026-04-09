//! Integration tests for the Stone Soup bridge.
//!
//! These tests require a working Python installation with `stonesoup` and
//! `numpy` installed.  They are `#[ignore]`d by default so that CI (which
//! may not have these dependencies) still passes.

#![cfg(feature = "stonesoup")]

use thresh_bridge::error::BridgeError;

/// Verify that we can import `stonesoup` from Python.
#[test]
#[ignore]
fn stonesoup_importable() {
    pyo3::Python::with_gil(|py| {
        let result = py.import("stonesoup");
        match result {
            Ok(_) => {} // success
            Err(e) => {
                let err = BridgeError::from(e);
                panic!("Failed to import stonesoup: {err}");
            }
        }
    });
}

/// Verify that nalgebra-to-numpy round-trips work.
#[test]
#[ignore]
fn nalgebra_numpy_roundtrip() {
    use nalgebra::{DMatrix, DVector};
    use thresh_bridge::convert;

    pyo3::Python::with_gil(|py| {
        // DVector round-trip
        let v = DVector::from_column_slice(&[1.0, 2.0, 3.0, 4.0]);
        let np_arr = convert::dvector_to_numpy(py, &v).expect("dvector_to_numpy");
        let v2 = convert::numpy_to_dvector(py, np_arr.bind(py)).expect("numpy_to_dvector");
        assert_eq!(v, v2);

        // DMatrix round-trip
        let m = DMatrix::from_row_slice(2, 3, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let np_mat = convert::dmatrix_to_numpy(py, &m).expect("dmatrix_to_numpy");
        let m2 = convert::numpy_to_dmatrix(py, np_mat.bind(py)).expect("numpy_to_dmatrix");
        assert_eq!(m, m2);
    });
}
