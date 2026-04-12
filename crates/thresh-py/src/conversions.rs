//! Conversion utilities between Python-friendly types and nalgebra types.

use nalgebra::{DMatrix, DVector};

/// Convert a slice of `Vec<f64>` (list of lists) into a `Vec<DVector<f64>>`.
pub fn lists_to_dvectors(data: &[Vec<f64>]) -> Vec<DVector<f64>> {
    data.iter().map(|v| DVector::from_vec(v.clone())).collect()
}

/// Convert a `DVector<f64>` to a `Vec<f64>`.
pub fn dvector_to_list(v: &DVector<f64>) -> Vec<f64> {
    v.as_slice().to_vec()
}

/// Convert a `DMatrix<f64>` to a `Vec<Vec<f64>>` (row-major list of lists).
pub fn dmatrix_to_lists(m: &DMatrix<f64>) -> Vec<Vec<f64>> {
    (0..m.nrows())
        .map(|i| (0..m.ncols()).map(|j| m[(i, j)]).collect())
        .collect()
}

/// Convert a `Vec<Vec<f64>>` (list of lists) to a `DMatrix<f64>`.
///
/// Returns `None` if the input is empty or ragged (rows have different lengths).
pub fn lists_to_dmatrix(data: &[Vec<f64>]) -> Option<DMatrix<f64>> {
    if data.is_empty() {
        return None;
    }
    let nrows = data.len();
    let ncols = data[0].len();
    if ncols == 0 {
        return None;
    }
    // Check for ragged rows.
    if data.iter().any(|row| row.len() != ncols) {
        return None;
    }
    // nalgebra DMatrix stores column-major, so we build from row-major data.
    let mut m = DMatrix::zeros(nrows, ncols);
    for (i, row) in data.iter().enumerate() {
        for (j, &val) in row.iter().enumerate() {
            m[(i, j)] = val;
        }
    }
    Some(m)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lists_to_dvectors() {
        let data = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0]];
        let dvecs = lists_to_dvectors(&data);
        assert_eq!(dvecs.len(), 2);
        assert_eq!(dvecs[0].len(), 3);
        assert_eq!(dvecs[1].len(), 2);
        assert!((dvecs[0][0] - 1.0).abs() < 1e-12);
        assert!((dvecs[1][1] - 5.0).abs() < 1e-12);
    }

    #[test]
    fn test_lists_to_dvectors_empty() {
        let data: Vec<Vec<f64>> = vec![];
        let dvecs = lists_to_dvectors(&data);
        assert!(dvecs.is_empty());
    }

    #[test]
    fn test_dvector_to_list() {
        let v = DVector::from_column_slice(&[1.0, 2.0, 3.0]);
        let list = dvector_to_list(&v);
        assert_eq!(list, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_dvector_to_list_empty() {
        let v = DVector::from_column_slice(&[] as &[f64]);
        let list = dvector_to_list(&v);
        assert!(list.is_empty());
    }

    #[test]
    fn test_dmatrix_to_lists() {
        let m = DMatrix::from_row_slice(2, 3, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let lists = dmatrix_to_lists(&m);
        assert_eq!(lists.len(), 2);
        assert_eq!(lists[0], vec![1.0, 2.0, 3.0]);
        assert_eq!(lists[1], vec![4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_lists_to_dmatrix() {
        let data = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let m = lists_to_dmatrix(&data).unwrap();
        assert_eq!(m.nrows(), 2);
        assert_eq!(m.ncols(), 2);
        assert!((m[(0, 0)] - 1.0).abs() < 1e-12);
        assert!((m[(0, 1)] - 2.0).abs() < 1e-12);
        assert!((m[(1, 0)] - 3.0).abs() < 1e-12);
        assert!((m[(1, 1)] - 4.0).abs() < 1e-12);
    }

    #[test]
    fn test_lists_to_dmatrix_ragged() {
        let data = vec![vec![1.0, 2.0], vec![3.0]];
        assert!(lists_to_dmatrix(&data).is_none());
    }

    #[test]
    fn test_lists_to_dmatrix_empty() {
        let data: Vec<Vec<f64>> = vec![];
        assert!(lists_to_dmatrix(&data).is_none());
    }

    #[test]
    fn test_lists_to_dmatrix_empty_rows() {
        let data = vec![vec![], vec![]];
        assert!(lists_to_dmatrix(&data).is_none());
    }

    #[test]
    fn test_roundtrip_dmatrix() {
        let original = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]];
        let m = lists_to_dmatrix(&original).unwrap();
        let back = dmatrix_to_lists(&m);
        assert_eq!(original, back);
    }

    #[test]
    fn test_roundtrip_dvector() {
        let original = vec![1.0, 2.0, 3.0, 4.0];
        let v = DVector::from_vec(original.clone());
        let back = dvector_to_list(&v);
        assert_eq!(original, back);
    }
}
