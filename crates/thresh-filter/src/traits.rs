//! Motion model traits for filter generics.

use nalgebra::{DMatrix, DVector};

/// A nonlinear motion model for EKF/UKF use.
pub trait MotionModel {
    /// State dimension.
    fn state_dim(&self) -> usize;

    /// Propagate the state forward by dt seconds.
    fn predict(&self, state: &DVector<f64>, dt: f64) -> DVector<f64>;

    /// Jacobian of the state transition (df/dx) evaluated at the given state.
    fn jacobian(&self, state: &DVector<f64>, dt: f64) -> DMatrix<f64>;

    /// Process noise covariance Q for the given dt.
    fn process_noise(&self, dt: f64) -> DMatrix<f64>;
}

/// A linear motion model with constant F matrix (for standard KF).
pub trait LinearModel: MotionModel {
    /// State transition matrix F for the given dt.
    fn transition_matrix(&self, dt: f64) -> DMatrix<f64>;

    /// Control input matrix G (optional, defaults to zero).
    fn control_matrix(&self, _dt: f64) -> Option<DMatrix<f64>> {
        None
    }
}
