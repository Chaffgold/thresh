//! Synthetic executable contracts for float64 array and shape conversion.

#![cfg(feature = "stonesoup")]

use std::ffi::CString;

use nalgebra::{DMatrix, DVector};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use thresh_bridge::convert;

fn with_python(test: impl for<'py> FnOnce(Python<'py>)) {
    Python::initialize();
    Python::attach(test);
}

fn array<'py>(py: Python<'py>, expression: &str) -> Bound<'py, PyAny> {
    let locals = PyDict::new(py);
    locals.set_item("np", py.import("numpy").unwrap()).unwrap();
    py.eval(&CString::new(expression).unwrap(), None, Some(&locals))
        .unwrap()
}

fn stonesoup_array<'py>(
    py: Python<'py>,
    class: &str,
    values: &Bound<'py, PyAny>,
) -> Bound<'py, PyAny> {
    py.import("stonesoup.types.array")
        .unwrap()
        .getattr(class)
        .unwrap()
        .call1((values,))
        .unwrap()
}

fn assert_array_layout(arr: &Bound<'_, PyAny>, shape: &[usize]) {
    assert_eq!(
        arr.getattr("shape")
            .unwrap()
            .extract::<Vec<usize>>()
            .unwrap(),
        shape
    );
    assert_eq!(
        arr.getattr("dtype")
            .unwrap()
            .getattr("name")
            .unwrap()
            .extract::<String>()
            .unwrap(),
        "float64"
    );
}

#[test]
fn outbound_vectors_are_1d_float64_and_nonsquare_matrices_preserve_every_element() {
    with_python(|py| {
        let vector = DVector::from_vec(vec![1.25, -2.0, 0.0]);
        let output = convert::dvector_to_numpy(py, &vector).unwrap();
        assert_array_layout(output.bind(py), &[3]);
        assert_eq!(
            convert::numpy_to_dvector(py, output.bind(py)).unwrap(),
            vector
        );
        let matrix = DMatrix::from_row_slice(2, 3, &[1.0, 2.0, 3.0, 11.0, 12.0, 13.0]);
        let output = convert::dmatrix_to_numpy(py, &matrix).unwrap();
        assert_array_layout(output.bind(py), &[2, 3]);
        let rows = output
            .bind(py)
            .call_method0("tolist")
            .unwrap()
            .extract::<Vec<Vec<f64>>>()
            .unwrap();
        assert_eq!(rows, vec![vec![1.0, 2.0, 3.0], vec![11.0, 12.0, 13.0]]);
        assert_eq!(
            convert::numpy_to_dmatrix(py, output.bind(py)).unwrap(),
            matrix
        );
    });
}

#[test]
fn strided_vectors_and_columns_preserve_logical_order() {
    with_python(|py| {
        for expression in [
            "np.arange(8, dtype='float64')[::-2]",
            "np.arange(8, dtype='float64')[::-2,None]",
            "np.matrix(np.arange(8, dtype='float64')[::-2,None])",
        ] {
            let input = array(py, expression);
            let vector = convert::numpy_to_dvector(py, &input).unwrap();
            assert_eq!(vector.as_slice(), &[7.0, 5.0, 3.0, 1.0]);
            let output = convert::dvector_to_numpy(py, &vector).unwrap();
            assert_array_layout(output.bind(py), &[4]);
            assert_eq!(
                convert::numpy_to_dvector(py, output.bind(py)).unwrap(),
                vector
            );
        }
        let state = stonesoup_array(
            py,
            "StateVector",
            &array(py, "np.arange(8, dtype='float64')[::-2]"),
        );
        assert_eq!(
            convert::numpy_to_dvector(py, &state).unwrap().as_slice(),
            &[7.0, 5.0, 3.0, 1.0]
        );
        let scalar_column = array(py, "np.array([[2.5]], dtype='float64')");
        assert_eq!(
            convert::numpy_to_dvector(py, &scalar_column)
                .unwrap()
                .as_slice(),
            &[2.5]
        );
    });
}

#[test]
fn strided_transposed_and_fortran_matrices_preserve_logical_order() {
    with_python(|py| {
        let cases = [
            (
                "np.arange(24, dtype='float64').reshape(4,6)[::2,::-2]",
                vec![5.0, 3.0, 1.0, 17.0, 15.0, 13.0],
            ),
            (
                "np.arange(6, dtype='float64').reshape(3,2).T",
                vec![0.0, 2.0, 4.0, 1.0, 3.0, 5.0],
            ),
            (
                "np.asfortranarray(np.arange(6, dtype='float64').reshape(2,3))",
                vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            ),
            (
                "np.matrix(np.arange(6, dtype='float64').reshape(2,3))",
                vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            ),
        ];
        for (expression, expected) in cases {
            let input = array(py, expression);
            let matrix = convert::numpy_to_dmatrix(py, &input).unwrap();
            assert_eq!(matrix, DMatrix::from_row_slice(2, 3, &expected));
            let output = convert::dmatrix_to_numpy(py, &matrix).unwrap();
            assert_eq!(
                convert::numpy_to_dmatrix(py, output.bind(py)).unwrap(),
                matrix
            );
        }
        let covariance = stonesoup_array(
            py,
            "CovarianceMatrix",
            &array(py, "np.array([[4.,1.],[1.,9.]])"),
        );
        assert_eq!(
            convert::numpy_to_dmatrix(py, &covariance).unwrap(),
            DMatrix::from_row_slice(2, 2, &[4.0, 1.0, 1.0, 9.0])
        );
    });
}

#[test]
fn unsupported_dtypes_are_never_coerced() {
    with_python(|py| {
        for dtype in ["int64", "float32", "complex128", "bool", "object", "U4"] {
            let input = array(py, &format!("np.ones((2,1), dtype='{dtype}')"));
            assert!(convert::numpy_to_dvector(py, &input).is_err(), "{dtype}");
            assert!(convert::numpy_to_dmatrix(py, &input).is_err(), "{dtype}");
            let matrix_subclass = array(py, &format!("np.matrix(np.ones((2,1), dtype='{dtype}'))"));
            assert!(
                convert::numpy_to_dvector(py, &matrix_subclass).is_err(),
                "{dtype}"
            );
            assert!(
                convert::numpy_to_dmatrix(py, &matrix_subclass).is_err(),
                "{dtype}"
            );
        }
        // Endianness is storage metadata, not a change in the float64 value type.
        let input = array(py, "np.array([1.25, -2.5], dtype='>f8')");
        assert_eq!(
            convert::numpy_to_dvector(py, &input).unwrap().as_slice(),
            &[1.25, -2.5]
        );
    });
}

#[test]
fn invalid_vector_shapes_do_not_flatten() {
    with_python(|py| {
        for expression in [
            "np.zeros((1,3))",
            "np.matrix(np.zeros((1,3)))",
            "np.zeros((2,2))",
            "np.zeros((2,1,1))",
            "np.array(1.0)",
            "np.zeros((0,))",
            "np.zeros((0,1))",
            "[1.0, 2.0]",
        ] {
            assert!(
                convert::numpy_to_dvector(py, &array(py, expression)).is_err(),
                "{expression}"
            );
        }
    });
}

#[test]
fn invalid_matrix_shapes_are_rejected() {
    with_python(|py| {
        for expression in [
            "np.array(1.0)",
            "np.zeros(2)",
            "np.zeros((2,2,2))",
            "np.zeros((0,2))",
            "np.zeros((2,0))",
            "[[1.0, 2.0]]",
        ] {
            assert!(
                convert::numpy_to_dmatrix(py, &array(py, expression)).is_err(),
                "{expression}"
            );
        }
    });
}

#[test]
fn nonfinite_inputs_are_rejected_in_both_directions() {
    with_python(|py| {
        for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            assert!(convert::dvector_to_numpy(py, &DVector::from_vec(vec![value])).is_err());
            assert!(convert::dmatrix_to_numpy(py, &DMatrix::from_element(1, 1, value)).is_err());
        }
        for expression in [
            "np.array([[np.nan]])",
            "np.array([[np.inf]])",
            "np.array([[-np.inf]])",
            "np.matrix([[np.nan]])",
            "np.matrix([[np.inf]])",
        ] {
            let input = array(py, expression);
            assert!(convert::numpy_to_dvector(py, &input).is_err());
            assert!(convert::numpy_to_dmatrix(py, &input).is_err());
        }
    });
}

#[test]
fn empty_nalgebra_arrays_are_rejected() {
    with_python(|py| {
        assert!(convert::dvector_to_numpy(py, &DVector::zeros(0)).is_err());
        assert!(convert::dmatrix_to_numpy(py, &DMatrix::zeros(0, 2)).is_err());
        assert!(convert::dmatrix_to_numpy(py, &DMatrix::zeros(2, 0)).is_err());
    });
}

#[test]
fn malformed_shape_and_checked_size_overflow_fail_before_matrix_construction() {
    with_python(|py| {
        let module = PyModule::from_code(
            py,
            c"
import numpy as np
class Malformed(np.ndarray):
    @property
    def shape(self):
        return self.reported
def malformed(shape):
    value = np.array([[1.0], [2.0]]).view(Malformed)
    value.reported = shape
    return value
",
            c"numeric_fixture.py",
            c"numeric_fixture",
        )
        .unwrap();
        let make = module.getattr("malformed").unwrap();
        for shape in [(3, 1), (usize::MAX, 2), (usize::MAX / 4, 1)] {
            let input = make.call1((shape,)).unwrap();
            assert!(convert::numpy_to_dmatrix(py, &input).is_err());
            assert!(convert::numpy_to_dvector(py, &input).is_err());
        }
        let input = make.call1(((-1, 1),)).unwrap();
        assert!(convert::numpy_to_dmatrix(py, &input).is_err());
        assert!(convert::numpy_to_dvector(py, &input).is_err());
    });
}
