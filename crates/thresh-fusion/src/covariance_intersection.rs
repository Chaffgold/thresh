//! Covariance Intersection for fusion with unknown cross-correlations.

use nalgebra::{DMatrix, DVector};

/// Fuse two estimates with unknown cross-correlations using Covariance Intersection.
///
/// Returns (x_fused, P_fused) that conservatively bounds the true MSE.
/// Optimizes omega in [0, 1] to minimize tr(P_fused).
pub fn covariance_intersection(
    x_a: &DVector<f64>,
    p_a: &DMatrix<f64>,
    x_b: &DVector<f64>,
    p_b: &DMatrix<f64>,
) -> (DVector<f64>, DMatrix<f64>) {
    // 1D line search for optimal omega
    let steps = 100;
    let mut best_omega = 0.5;
    let mut best_trace = f64::INFINITY;

    for i in 0..=steps {
        let omega = i as f64 / steps as f64;
        let omega = omega.clamp(0.01, 0.99); // avoid singularities

        let p_a_inv = p_a.clone().try_inverse().unwrap();
        let p_b_inv = p_b.clone().try_inverse().unwrap();
        let p_fused_inv = &p_a_inv * omega + &p_b_inv * (1.0 - omega);
        if let Some(p_fused) = p_fused_inv.try_inverse() {
            let tr = p_fused.trace();
            if tr < best_trace {
                best_trace = tr;
                best_omega = omega;
            }
        }
    }

    let p_a_inv = p_a.clone().try_inverse().unwrap();
    let p_b_inv = p_b.clone().try_inverse().unwrap();
    let p_fused_inv = &p_a_inv * best_omega + &p_b_inv * (1.0 - best_omega);
    let p_fused = p_fused_inv.try_inverse().expect("CI fused P singular");
    let x_fused = &p_fused * (&p_a_inv * x_a * best_omega + &p_b_inv * x_b * (1.0 - best_omega));

    (x_fused, p_fused)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ci_fused_covariance_bounds_both() {
        let x_a = DVector::from_column_slice(&[10.0, 20.0]);
        let p_a = DMatrix::from_diagonal(&DVector::from_column_slice(&[5.0, 10.0]));
        let x_b = DVector::from_column_slice(&[12.0, 18.0]);
        let p_b = DMatrix::from_diagonal(&DVector::from_column_slice(&[8.0, 3.0]));

        let (x_f, p_f) = covariance_intersection(&x_a, &p_a, &x_b, &p_b);

        // Fused estimate should be between the two
        assert!(x_f[0] >= 9.0 && x_f[0] <= 13.0);
        assert!(x_f[1] >= 17.0 && x_f[1] <= 21.0);

        // P_fused should be PSD
        let eigenvalues = p_f.clone().symmetric_eigen().eigenvalues;
        for i in 0..eigenvalues.len() {
            assert!(
                eigenvalues[i] > 0.0,
                "Non-positive eigenvalue: {}",
                eigenvalues[i]
            );
        }

        // Trace of fused should be <= trace of each input (CI is conservative
        // but still reduces uncertainty vs either alone)
        let tr_f = p_f.trace();
        let tr_min = p_a.trace().min(p_b.trace());
        assert!(
            tr_f <= tr_min + 1e-6,
            "CI fused trace {tr_f} should be <= min input trace {tr_min}"
        );
    }
}
