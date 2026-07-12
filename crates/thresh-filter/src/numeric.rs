//! Shared numeric-Jacobian helper for nonlinear motion models.
//!
//! Central differences of the exact discretized `predict` map (design
//! Decision 3 of the `orbital-ballistic-filter-models` change): EKF
//! consistency requires the Jacobian of the map actually used, and
//! differencing the sub-stepped RK4 delivers that automatically for any
//! force composition. See the "How the numeric Jacobian is cross-checked"
//! section of `docs/reference/orbital-ballistic-dynamics-reference.md` for
//! the step-size analysis.

use nalgebra::{DMatrix, DVector};

/// Central-difference step factor `ε = ∛ε_mach ≈ 6.06e-6`.
///
/// The cube root is the standard optimum for central differences:
/// truncation error `O(h²)` balances rounding error `O(ε_mach/h)` at
/// `h ~ ε_mach^(1/3)`.
fn step_epsilon() -> f64 {
    f64::EPSILON.cbrt()
}

/// Per-column central-difference step `h_i = ε·max(|x_i|, s_i)`.
///
/// The floor scale `s_i` keeps the step meaningful when the component is
/// near zero (e.g. an ECI coordinate crossing a coordinate plane).
fn column_step(x_i: f64, scale_i: f64) -> f64 {
    step_epsilon() * x_i.abs().max(scale_i)
}

/// One central-difference column: `(f(x + h·e_i, dt) − f(x − h·e_i, dt)) / (2h)`.
fn central_difference_column<F>(f: &F, x: &DVector<f64>, dt: f64, i: usize, h: f64) -> DVector<f64>
where
    F: Fn(&DVector<f64>, f64) -> DVector<f64>,
{
    let mut x_plus = x.clone();
    let mut x_minus = x.clone();
    x_plus[i] += h;
    x_minus[i] -= h;
    (f(&x_plus, dt) - f(&x_minus, dt)) / (2.0 * h)
}

/// Numerically differentiate a discretized state-transition map
/// `f(x, dt) -> x'` at `x`, returning the Jacobian `∂f/∂x` (column `i` is
/// the sensitivity to state component `i`).
///
/// Central differences with per-column step `h_i = ε·max(|x_i|, s_i)`,
/// `ε = ∛ε_mach`, where `scales[i] = s_i` is the per-component floor
/// (design Decision 3 of the `orbital-ballistic-filter-models` change).
/// Cost: `2·x.len()` evaluations of `f`.
///
/// # Panics
///
/// Panics if `scales.len() != x.len()`.
pub fn numeric_jacobian<F>(f: F, x: &DVector<f64>, dt: f64, scales: &[f64]) -> DMatrix<f64>
where
    F: Fn(&DVector<f64>, f64) -> DVector<f64>,
{
    assert_eq!(
        scales.len(),
        x.len(),
        "numeric_jacobian: scales length ({}) must match state length ({})",
        scales.len(),
        x.len()
    );
    let columns: Vec<DVector<f64>> = (0..x.len())
        .map(|i| central_difference_column(&f, x, dt, i, column_step(x[i], scales[i])))
        .collect();
    DMatrix::from_columns(&columns)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::cv::ConstantVelocity;
    use crate::traits::{LinearModel, MotionModel};

    // ── column_step ──────────────────────────────────────────────────────

    #[test]
    fn column_step_uses_component_magnitude_above_floor() {
        let h = column_step(-2_000_000.0, 1.0);
        assert!((h - step_epsilon() * 2_000_000.0).abs() < 1e-12);
    }

    #[test]
    fn column_step_falls_back_to_floor_scale_near_zero() {
        let h = column_step(0.0, 1e-3);
        assert!((h - step_epsilon() * 1e-3).abs() < 1e-25);
        assert!(h > 0.0, "step must never degenerate to zero");
    }

    // ── central_difference_column ────────────────────────────────────────

    #[test]
    fn central_difference_is_exact_for_quadratics() {
        // f(x) = [x0², x0·x1] — central differences are exact for degree ≤ 2.
        let f =
            |x: &DVector<f64>, _dt: f64| DVector::from_column_slice(&[x[0] * x[0], x[0] * x[1]]);
        let x = DVector::from_column_slice(&[3.0, -2.0]);
        let col = central_difference_column(&f, &x, 0.0, 0, 1e-4);
        // ∂/∂x0 = [2·x0, x1] = [6, -2]
        assert!((col[0] - 6.0).abs() < 1e-9);
        assert!((col[1] - (-2.0)).abs() < 1e-9);
    }

    // ── numeric_jacobian vs ConstantVelocity's exact F (task 3.1) ────────

    #[test]
    fn numeric_jacobian_matches_cv_exact_transition_matrix() {
        let cv = ConstantVelocity::new(1.0);
        let dt = 0.5;
        // Representative state with every component nonzero, so each column
        // step scales with its component (the floor path is covered by the
        // `column_step` tests above; a floor-sized step against large
        // function outputs is dominated by rounding by construction).
        let x = DVector::from_column_slice(&[100.0, 10.0, -50.0, 4.0, 2000.0, -7.5]);
        let scales = [1.0, 1e-3, 1.0, 1e-3, 1.0, 1e-3];

        let numeric = numeric_jacobian(|s, step| cv.predict(s, step), &x, dt, &scales);
        let exact = cv.transition_matrix(dt);

        // Linear dynamics: central differences are exact up to rounding.
        let max_err = (&numeric - &exact).amax();
        assert!(max_err < 1e-7, "max |F_num - F_exact| = {max_err:e}");
    }

    #[test]
    #[should_panic(expected = "scales length")]
    fn numeric_jacobian_rejects_mismatched_scales() {
        let f = |x: &DVector<f64>, _dt: f64| x.clone();
        let x = DVector::from_column_slice(&[1.0, 2.0]);
        let _ = numeric_jacobian(f, &x, 1.0, &[1.0]);
    }
}
