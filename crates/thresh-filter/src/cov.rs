//! Shared covariance numerical-stability helpers used by the nonlinear
//! filters (`ukf`, `ckf`). Centralised here so the repair logic is defined
//! once rather than copied per filter.

use nalgebra::DMatrix;

/// Covariance eigenvalue floor used when repairing an indefinite matrix.
const MIN_EIGENVALUE: f64 = 1e-10;

/// Clamp any eigenvalues below `MIN_EIGENVALUE` back up to it, restoring
/// positive-definiteness. A no-op when `p` is already positive-definite.
pub fn ensure_psd(p: &mut DMatrix<f64>) {
    let eigen = p.clone().symmetric_eigen();
    let mut clamped = eigen.eigenvalues.clone();
    let mut needs_repair = false;
    for i in 0..clamped.len() {
        if clamped[i] < MIN_EIGENVALUE {
            clamped[i] = MIN_EIGENVALUE;
            needs_repair = true;
        }
    }
    if needs_repair {
        let d = DMatrix::from_diagonal(&clamped);
        let repaired = &eigen.eigenvectors * d * eigen.eigenvectors.transpose();
        *p = (&repaired + repaired.transpose()) * 0.5;
    }
}

/// Return the symmetric part of `p`: `(P + Pᵀ) / 2`. Used to scrub the
/// small asymmetry that accumulates through filter covariance updates.
pub fn symmetrize(p: &DMatrix<f64>) -> DMatrix<f64> {
    (p + p.transpose()) * 0.5
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ensure_psd_is_noop_when_already_pd() {
        let mut p = DMatrix::<f64>::identity(3, 3) * 2.0;
        let before = p.clone();
        ensure_psd(&mut p);
        assert!((p - before).norm() < 1e-12, "PD input must be unchanged");
    }

    #[test]
    fn ensure_psd_repairs_indefinite_matrix() {
        // Symmetric but indefinite: eigenvalues 3 and -1.
        let mut p = DMatrix::from_row_slice(2, 2, &[1.0, 2.0, 2.0, 1.0]);
        ensure_psd(&mut p);
        let min_eig = p
            .clone()
            .symmetric_eigen()
            .eigenvalues
            .iter()
            .copied()
            .fold(f64::INFINITY, f64::min);
        assert!(
            min_eig > 0.0,
            "expected PD after repair, min eig = {min_eig}"
        );
        assert!(
            (&p - p.transpose()).norm() < 1e-12,
            "repaired matrix must stay symmetric"
        );
    }

    #[test]
    fn symmetrize_averages_off_diagonal_entries() {
        let m = DMatrix::from_row_slice(2, 2, &[1.0, 3.0, 1.0, 5.0]);
        let s = symmetrize(&m);
        assert!((s[(0, 1)] - 2.0).abs() < 1e-12);
        assert!((s[(1, 0)] - 2.0).abs() < 1e-12);
        assert!((&s - s.transpose()).norm() < 1e-12);
    }
}
