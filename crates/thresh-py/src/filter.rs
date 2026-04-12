//! Python wrapper around the linear Kalman filter.

use nalgebra::{DMatrix, DVector};
use pyo3::prelude::*;
use thresh_filter::kf::KalmanFilter;

use crate::conversions::{dmatrix_to_lists, dvector_to_list, lists_to_dmatrix};
use crate::errors::thresh_err;

/// A linear Kalman filter exposed to Python.
#[pyclass]
pub struct PyKalmanFilter {
    inner: KalmanFilter,
}

#[pymethods]
impl PyKalmanFilter {
    /// Create a new Kalman filter with the given state dimension.
    ///
    /// The initial state is zero and the initial covariance is the identity
    /// matrix.
    #[new]
    fn new(state_dim: usize) -> PyResult<Self> {
        if state_dim == 0 {
            return Err(thresh_err("state_dim must be > 0"));
        }
        let x = DVector::zeros(state_dim);
        let p = DMatrix::identity(state_dim, state_dim);
        Ok(Self {
            inner: KalmanFilter::new(x, p),
        })
    }

    /// Run the predict step with a transition matrix F and process noise Q.
    ///
    /// Both are given as list-of-lists (row-major).
    fn predict(&mut self, f: Vec<Vec<f64>>, q: Vec<Vec<f64>>) -> PyResult<()> {
        let f_mat = lists_to_dmatrix(&f)
            .ok_or_else(|| thresh_err("F must be a non-empty rectangular matrix"))?;
        let q_mat = lists_to_dmatrix(&q)
            .ok_or_else(|| thresh_err("Q must be a non-empty rectangular matrix"))?;

        let n = self.inner.x.len();
        if f_mat.nrows() != n || f_mat.ncols() != n {
            return Err(thresh_err(format!(
                "F must be {n}x{n}, got {}x{}",
                f_mat.nrows(),
                f_mat.ncols()
            )));
        }
        if q_mat.nrows() != n || q_mat.ncols() != n {
            return Err(thresh_err(format!(
                "Q must be {n}x{n}, got {}x{}",
                q_mat.nrows(),
                q_mat.ncols()
            )));
        }

        self.inner.x = &f_mat * &self.inner.x;
        self.inner.p = &f_mat * &self.inner.p * f_mat.transpose() + q_mat;
        Ok(())
    }

    /// Run the update step given measurement z, observation matrix H, and
    /// measurement noise R.
    ///
    /// `z` is a flat list; `h` and `r` are list-of-lists (row-major).
    fn update(&mut self, z: Vec<f64>, h: Vec<Vec<f64>>, r: Vec<Vec<f64>>) -> PyResult<()> {
        let z_vec = DVector::from_vec(z);
        let h_mat = lists_to_dmatrix(&h)
            .ok_or_else(|| thresh_err("H must be a non-empty rectangular matrix"))?;
        let r_mat = lists_to_dmatrix(&r)
            .ok_or_else(|| thresh_err("R must be a non-empty rectangular matrix"))?;

        let n = self.inner.x.len();
        let m = z_vec.len();
        if h_mat.nrows() != m || h_mat.ncols() != n {
            return Err(thresh_err(format!(
                "H must be {m}x{n}, got {}x{}",
                h_mat.nrows(),
                h_mat.ncols()
            )));
        }
        if r_mat.nrows() != m || r_mat.ncols() != m {
            return Err(thresh_err(format!(
                "R must be {m}x{m}, got {}x{}",
                r_mat.nrows(),
                r_mat.ncols()
            )));
        }

        self.inner.update(&z_vec, &h_mat, &r_mat);
        Ok(())
    }

    /// Get the current state estimate as a list.
    #[getter]
    fn state(&self) -> Vec<f64> {
        dvector_to_list(&self.inner.x)
    }

    /// Get the current covariance matrix as a list of lists.
    #[getter]
    fn covariance(&self) -> Vec<Vec<f64>> {
        dmatrix_to_lists(&self.inner.p)
    }
}

// ── Rust-only unit tests (no Python runtime needed) ─────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::conversions::dmatrix_to_lists;

    /// Helper: build an identity matrix as list-of-lists.
    fn identity_lists(n: usize) -> Vec<Vec<f64>> {
        dmatrix_to_lists(&DMatrix::identity(n, n))
    }

    /// Helper: build a scaled identity matrix as list-of-lists.
    fn scaled_identity_lists(n: usize, scale: f64) -> Vec<Vec<f64>> {
        dmatrix_to_lists(&(DMatrix::identity(n, n) * scale))
    }

    #[test]
    fn test_new_filter_state_and_covariance() {
        let kf = PyKalmanFilter {
            inner: KalmanFilter::new(DVector::zeros(4), DMatrix::identity(4, 4)),
        };
        assert_eq!(kf.state(), vec![0.0, 0.0, 0.0, 0.0]);
        assert_eq!(kf.covariance().len(), 4);
        assert_eq!(kf.covariance()[0].len(), 4);
        // Diagonal should be 1.0
        for i in 0..4 {
            assert!((kf.covariance()[i][i] - 1.0).abs() < 1e-12);
        }
    }

    #[test]
    fn test_predict_changes_state() {
        // 2D state: [position, velocity]
        let mut kf = PyKalmanFilter {
            inner: KalmanFilter::new(
                DVector::from_column_slice(&[0.0, 1.0]),
                DMatrix::identity(2, 2),
            ),
        };

        // Transition: position += velocity * dt (dt=1)
        let f = vec![vec![1.0, 1.0], vec![0.0, 1.0]];
        let q = scaled_identity_lists(2, 0.01);

        kf.predict(f, q).unwrap();

        let state = kf.state();
        // New position = 0 + 1*1 = 1.0
        assert!((state[0] - 1.0).abs() < 1e-12);
        // Velocity unchanged
        assert!((state[1] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_update_changes_state() {
        let mut kf = PyKalmanFilter {
            inner: KalmanFilter::new(
                DVector::from_column_slice(&[0.0, 0.0]),
                DMatrix::identity(2, 2) * 100.0,
            ),
        };

        // Observe only position (first component)
        let z = vec![10.0];
        let h = vec![vec![1.0, 0.0]];
        let r = vec![vec![1.0]];

        kf.update(z, h, r).unwrap();

        let state = kf.state();
        // State should move toward measurement
        assert!(
            state[0] > 5.0,
            "Position should be pulled toward 10.0, got {}",
            state[0]
        );
    }

    #[test]
    fn test_predict_then_update_cycle() {
        let mut kf = PyKalmanFilter {
            inner: KalmanFilter::new(
                DVector::from_column_slice(&[0.0, 0.0]),
                DMatrix::identity(2, 2) * 1000.0,
            ),
        };

        let f = vec![vec![1.0, 1.0], vec![0.0, 1.0]];
        let q = scaled_identity_lists(2, 0.1);
        let h = vec![vec![1.0, 0.0]];
        let r = vec![vec![1.0]];

        // Target at position 0 moving at velocity 5.
        for step in 1..=20 {
            kf.predict(f.clone(), q.clone()).unwrap();
            let z = vec![5.0 * step as f64];
            kf.update(z, h.clone(), r.clone()).unwrap();
        }

        let state = kf.state();
        // Should converge near true velocity of 5.0
        assert!(
            (state[1] - 5.0).abs() < 1.0,
            "Velocity should converge to ~5.0, got {}",
            state[1]
        );
    }

    #[test]
    fn test_predict_dimension_mismatch() {
        let mut kf = PyKalmanFilter {
            inner: KalmanFilter::new(DVector::zeros(2), DMatrix::identity(2, 2)),
        };
        // Wrong-sized F (3x3 instead of 2x2)
        let f = identity_lists(3);
        let q = identity_lists(2);
        assert!(kf.predict(f, q).is_err());
    }

    #[test]
    fn test_update_dimension_mismatch() {
        let mut kf = PyKalmanFilter {
            inner: KalmanFilter::new(DVector::zeros(2), DMatrix::identity(2, 2)),
        };
        // z has 1 element but H is 2x2 (should be 1x2)
        let z = vec![1.0];
        let h = identity_lists(2);
        let r = vec![vec![1.0]];
        assert!(kf.update(z, h, r).is_err());
    }

    #[test]
    fn test_ragged_matrix_rejected() {
        let mut kf = PyKalmanFilter {
            inner: KalmanFilter::new(DVector::zeros(2), DMatrix::identity(2, 2)),
        };
        let f = vec![vec![1.0, 0.0], vec![0.0]]; // ragged
        let q = identity_lists(2);
        assert!(kf.predict(f, q).is_err());
    }
}
