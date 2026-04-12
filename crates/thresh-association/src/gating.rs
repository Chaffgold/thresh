//! Mahalanobis distance and chi-squared gating.

use nalgebra::{DMatrix, DVector};

/// Compute the squared Mahalanobis distance: d² = (z - Hx)ᵀ S⁻¹ (z - Hx).
pub fn mahalanobis_squared(z: &DVector<f64>, predicted_z: &DVector<f64>, s: &DMatrix<f64>) -> f64 {
    let innovation = z - predicted_z;
    let s_inv = s
        .clone()
        .try_inverse()
        .expect("Innovation covariance S is singular in Mahalanobis");
    (innovation.transpose() * s_inv * &innovation)[(0, 0)]
}

/// Chi-squared gating thresholds for given measurement dimension and significance level.
///
/// Returns the chi-squared critical value for p = 0.99 (1% false rejection).
pub fn chi2_threshold(measurement_dim: usize) -> f64 {
    // Pre-computed chi-squared critical values at p=0.99
    match measurement_dim {
        1 => 6.635,
        2 => 9.210,
        3 => 11.345,
        4 => 13.277,
        5 => 15.086,
        6 => 16.812,
        _ => {
            // Approximation for higher dimensions: dim + 3*sqrt(2*dim)
            let d = measurement_dim as f64;
            d + 3.0 * (2.0 * d).sqrt()
        }
    }
}

/// Check if a measurement passes the Mahalanobis gate.
pub fn passes_gate(
    z: &DVector<f64>,
    predicted_z: &DVector<f64>,
    s: &DMatrix<f64>,
    threshold: Option<f64>,
) -> bool {
    let d2 = mahalanobis_squared(z, predicted_z, s);
    let gate = threshold.unwrap_or_else(|| chi2_threshold(z.len()));
    d2 <= gate
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_innovation_zero_distance() {
        let z = DVector::from_column_slice(&[1.0, 2.0]);
        let s = DMatrix::identity(2, 2);
        let d2 = mahalanobis_squared(&z, &z, &s);
        assert!(d2.abs() < 1e-10);
    }

    #[test]
    fn mahalanobis_identity_is_euclidean() {
        let z = DVector::from_column_slice(&[3.0, 4.0]);
        let pred = DVector::from_column_slice(&[0.0, 0.0]);
        let s = DMatrix::identity(2, 2);
        let d2 = mahalanobis_squared(&z, &pred, &s);
        assert!((d2 - 25.0).abs() < 1e-10);
    }

    #[test]
    fn gating_accepts_close() {
        let z = DVector::from_column_slice(&[1.0, 0.0]);
        let pred = DVector::from_column_slice(&[0.0, 0.0]);
        let s = DMatrix::identity(2, 2) * 10.0;
        assert!(passes_gate(&z, &pred, &s, None));
    }

    #[test]
    fn gating_rejects_far() {
        let z = DVector::from_column_slice(&[100.0, 100.0]);
        let pred = DVector::from_column_slice(&[0.0, 0.0]);
        let s = DMatrix::identity(2, 2);
        assert!(!passes_gate(&z, &pred, &s, None));
    }
}
