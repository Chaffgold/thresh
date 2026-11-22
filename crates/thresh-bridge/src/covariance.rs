//! Pure-Rust observation covariance validation; never modifies supplied noise.

use nalgebra::{DMatrix, SymmetricEigen};

use crate::error::{BridgeError, BridgeResult};

const RELATIVE_TOLERANCE: f64 = 1e-12;

/// Check a finite square covariance at the declared observation dimension.
///
/// Symmetry and minimum eigenvalue use `1e-12 * max(1, ||R||_infinity)`.
/// Scaling makes both the bound and eigensolver safe for extreme finite entries.
/// PSD matrices (including singular ones) are accepted unchanged, within this
/// tolerance; this does not guarantee suitability for an updater requiring an
/// inverse. No jitter or symmetrization is performed.
pub(crate) fn validate_covariance(matrix: &DMatrix<f64>, dimension: usize) -> BridgeResult<()> {
    validate_dimensions_and_values(matrix, dimension)?;
    let (normalized, tolerance) = normalized_covariance(matrix)?;
    validate_symmetry(&normalized, tolerance)?;
    validate_eigenvalues(normalized, tolerance)
}

fn validate_dimensions_and_values(matrix: &DMatrix<f64>, dimension: usize) -> BridgeResult<()> {
    if dimension == 0 || matrix.shape() != (dimension, dimension) {
        return invalid("covariance must be square at the observation dimension");
    }
    if matrix.iter().any(|value| !value.is_finite()) {
        return invalid("covariance entries must be finite");
    }
    Ok(())
}

fn normalized_covariance(matrix: &DMatrix<f64>) -> BridgeResult<(DMatrix<f64>, f64)> {
    let scale = matrix
        .iter()
        .fold(1.0_f64, |max, value| max.max(value.abs()));
    let normalized = matrix / scale;
    let norm = normalized
        .row_iter()
        .map(|row| row.iter().map(|value| value.abs()).sum::<f64>())
        .fold(0.0_f64, f64::max);
    let normalized_bound = norm.max(1.0 / scale);
    let tolerance = RELATIVE_TOLERANCE * normalized_bound;
    // Multiply by the small factor first: ||R|| may exceed f64::MAX even when
    // the specified validation tolerance is finite and faithfully representable.
    let absolute_tolerance = (RELATIVE_TOLERANCE * scale) * normalized_bound;
    if !tolerance.is_finite() || !absolute_tolerance.is_finite() {
        return invalid("covariance validation bound is not finite");
    }
    Ok((normalized, tolerance))
}

fn validate_symmetry(matrix: &DMatrix<f64>, tolerance: f64) -> BridgeResult<()> {
    for row in 0..matrix.nrows() {
        for col in 0..row {
            if (matrix[(row, col)] - matrix[(col, row)]).abs() > tolerance {
                return invalid("covariance is asymmetric beyond tolerance");
            }
        }
    }
    Ok(())
}

fn validate_eigenvalues(matrix: DMatrix<f64>, tolerance: f64) -> BridgeResult<()> {
    let iterations = matrix
        .nrows()
        .checked_mul(matrix.ncols())
        .and_then(|count| count.checked_mul(100))
        .ok_or_else(|| BridgeError::ConversionError("covariance dimension overflow".into()))?;
    // Only the lower triangle is read after the explicit tolerance check above.
    // The original matrix is untouched, including any tolerated asymmetry.
    let eigen = SymmetricEigen::try_new(matrix, f64::EPSILON, iterations).ok_or_else(|| {
        BridgeError::ConversionError("covariance eigensolver did not converge".into())
    })?;
    if eigen.eigenvalues.iter().any(|value| !value.is_finite()) {
        return invalid("covariance eigenvalue result is not finite");
    }
    if eigen.eigenvalues.iter().any(|value| *value < -tolerance) {
        return invalid("covariance is not positive semidefinite within tolerance");
    }
    Ok(())
}

fn invalid<T>(message: &str) -> BridgeResult<T> {
    Err(BridgeError::ConversionError(message.into()))
}

#[cfg(test)]
mod tests {
    use super::{normalized_covariance, validate_covariance};
    use nalgebra::DMatrix;

    #[test]
    fn correlated_and_singular_covariance_remain_unchanged() {
        let cases = [
            DMatrix::from_row_slice(2, 2, &[4.0, 1.0, 1.0, 9.0]),
            DMatrix::from_row_slice(2, 2, &[1.0, 2.0, 2.0, 4.0]),
            DMatrix::zeros(3, 3),
        ];
        for covariance in cases {
            let original = covariance.clone();
            validate_covariance(&covariance, covariance.nrows()).unwrap();
            assert_eq!(covariance, original);
        }
    }

    #[test]
    fn malformed_nonfinite_and_indefinite_covariance_rejected() {
        let cases = [
            DMatrix::zeros(0, 0),
            DMatrix::zeros(2, 3),
            DMatrix::from_row_slice(2, 2, &[1.0, 2.0, 2.0, 1.0]),
            DMatrix::from_row_slice(2, 2, &[1.0, 0.0, 0.1, 1.0]),
            DMatrix::from_row_slice(2, 2, &[f64::NAN, 0.0, 0.0, 1.0]),
            DMatrix::from_row_slice(2, 2, &[1.0, 0.0, 0.0, f64::INFINITY]),
        ];
        for covariance in cases {
            assert!(validate_covariance(&covariance, 2).is_err());
        }
        assert!(validate_covariance(&DMatrix::identity(3, 3), 2).is_err());
    }

    #[test]
    fn covariance_tolerance_is_scale_aware_without_repairs() {
        let tolerated = DMatrix::from_row_slice(2, 2, &[1.0, 1e-13, 0.0, -5e-13]);
        let original = tolerated.clone();
        validate_covariance(&tolerated, 2).unwrap();
        assert_eq!(tolerated, original);
        assert!(validate_covariance(&(tolerated * 1e100), 2).is_ok());
        let rejected = DMatrix::from_row_slice(2, 2, &[1.0, 0.0, 0.0, -2e-12]);
        assert!(validate_covariance(&rejected, 2).is_err());
        assert!(validate_covariance(&(rejected * 1e100), 2).is_err());
    }

    #[test]
    fn extreme_finite_covariance_never_gets_infinite_tolerance() {
        let huge = f64::MAX;
        let positive = DMatrix::from_row_slice(2, 2, &[huge, huge, huge, huge]);
        let indefinite = DMatrix::from_row_slice(2, 2, &[huge / 2.0, huge, huge, huge / 2.0]);
        let asymmetric = DMatrix::from_row_slice(2, 2, &[huge, huge, -huge, huge]);
        assert!((huge + huge).is_infinite());
        let (_, bound) = normalized_covariance(&positive).unwrap();
        assert!(bound.is_finite());
        assert!(validate_covariance(&positive, 2).is_ok());
        assert!(validate_covariance(&indefinite, 2).is_err());
        assert!(validate_covariance(&asymmetric, 2).is_err());
        assert_eq!(positive[(0, 0)], huge);
    }
}
