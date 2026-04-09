//! State vector and covariance matrix types for tracking.

use nalgebra::{DMatrix, DVector};
use serde::{Deserialize, Serialize};

/// State vector wrapping a dynamic-size nalgebra vector.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StateVector {
    /// The underlying state values.
    pub data: DVector<f64>,
    /// Dimension of the state space.
    pub dim: usize,
}

impl StateVector {
    /// Create a new state vector from a slice.
    pub fn new(values: &[f64]) -> Self {
        Self {
            dim: values.len(),
            data: DVector::from_column_slice(values),
        }
    }

    /// Create a zero state vector with the given dimension.
    pub fn zeros(dim: usize) -> Self {
        Self {
            dim,
            data: DVector::zeros(dim),
        }
    }

    /// Number of elements.
    pub fn len(&self) -> usize {
        self.dim
    }

    /// Whether the state vector is empty.
    pub fn is_empty(&self) -> bool {
        self.dim == 0
    }
}

impl std::ops::Index<usize> for StateVector {
    type Output = f64;

    fn index(&self, index: usize) -> &f64 {
        &self.data[index]
    }
}

impl std::ops::IndexMut<usize> for StateVector {
    fn index_mut(&mut self, index: usize) -> &mut f64 {
        &mut self.data[index]
    }
}

/// Covariance matrix with symmetry enforcement.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CovarianceMatrix {
    /// The underlying matrix data.
    pub data: DMatrix<f64>,
    /// Dimension (rows = cols).
    pub dim: usize,
}

impl CovarianceMatrix {
    /// Create a covariance matrix from a square matrix.
    ///
    /// Enforces symmetry by averaging with its transpose.
    pub fn new(mat: DMatrix<f64>) -> Self {
        assert_eq!(mat.nrows(), mat.ncols(), "Covariance must be square");
        let dim = mat.nrows();
        let symmetric = (&mat + mat.transpose()) * 0.5;
        Self {
            dim,
            data: symmetric,
        }
    }

    /// Create a diagonal covariance from a slice of variances.
    pub fn from_diagonal(variances: &[f64]) -> Self {
        let dim = variances.len();
        let diag = DVector::from_column_slice(variances);
        Self {
            dim,
            data: DMatrix::from_diagonal(&diag),
        }
    }

    /// Create a zero covariance matrix.
    pub fn zeros(dim: usize) -> Self {
        Self {
            dim,
            data: DMatrix::zeros(dim, dim),
        }
    }

    /// Create an identity-scaled covariance.
    pub fn identity(dim: usize, scale: f64) -> Self {
        Self {
            dim,
            data: DMatrix::identity(dim, dim) * scale,
        }
    }

    /// Re-enforce symmetry (useful after numerical operations).
    pub fn enforce_symmetry(&mut self) {
        self.data = (&self.data + self.data.transpose()) * 0.5;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn state_vector_new() {
        let sv = StateVector::new(&[1.0, 2.0, 3.0]);
        assert_eq!(sv.len(), 3);
        assert_eq!(sv[0], 1.0);
        assert_eq!(sv[1], 2.0);
        assert_eq!(sv[2], 3.0);
    }

    #[test]
    fn state_vector_zeros() {
        let sv = StateVector::zeros(6);
        assert_eq!(sv.len(), 6);
        assert_eq!(sv[0], 0.0);
    }

    #[test]
    fn state_vector_index_mut() {
        let mut sv = StateVector::zeros(3);
        sv[1] = 42.0;
        assert_eq!(sv[1], 42.0);
    }

    #[test]
    fn covariance_symmetry_enforcement() {
        let mat = DMatrix::from_row_slice(2, 2, &[1.0, 0.5, 0.3, 2.0]);
        let cov = CovarianceMatrix::new(mat);
        assert_eq!(cov.data[(0, 1)], cov.data[(1, 0)]);
        assert_eq!(cov.data[(0, 1)], 0.4); // (0.5 + 0.3) / 2
    }

    #[test]
    fn covariance_diagonal() {
        let cov = CovarianceMatrix::from_diagonal(&[1.0, 4.0, 9.0]);
        assert_eq!(cov.dim, 3);
        assert_eq!(cov.data[(0, 0)], 1.0);
        assert_eq!(cov.data[(1, 1)], 4.0);
        assert_eq!(cov.data[(2, 2)], 9.0);
        assert_eq!(cov.data[(0, 1)], 0.0);
    }

    #[test]
    fn covariance_identity() {
        let cov = CovarianceMatrix::identity(3, 5.0);
        assert_eq!(cov.data[(0, 0)], 5.0);
        assert_eq!(cov.data[(1, 1)], 5.0);
        assert_eq!(cov.data[(0, 1)], 0.0);
    }
}
