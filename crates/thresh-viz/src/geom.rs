//! Pure geometry helpers used by the plot renderer.
//!
//! Kept in its own module so the math is testable without standing up
//! an egui context.

/// Semi-axes (semi-major, semi-minor) and rotation angle of the
/// 2σ confidence ellipse for a 2D symmetric covariance:
///
/// ```text
/// [ cov_xx  cov_xy ]
/// [ cov_xy  cov_yy ]
/// ```
///
/// Returns `(semi_major, semi_minor, angle_rad)` in plot coordinates,
/// where `angle_rad` is the rotation of the major axis from +x.
///
/// Returns `None` if the covariance is degenerate (any eigenvalue ≤ 0)
/// — caller should skip drawing rather than emit a singularity.
pub fn ellipse_axes(cov_xx: f64, cov_xy: f64, cov_yy: f64) -> Option<(f64, f64, f64)> {
    if !cov_xx.is_finite() || !cov_xy.is_finite() || !cov_yy.is_finite() {
        return None;
    }
    // Eigenvalues of the 2x2 symmetric matrix.
    let trace = cov_xx + cov_yy;
    let det = cov_xx * cov_yy - cov_xy * cov_xy;
    let disc = (trace * trace * 0.25 - det).max(0.0);
    let root = disc.sqrt();
    let lambda_max = trace * 0.5 + root;
    let lambda_min = trace * 0.5 - root;

    if lambda_min <= 0.0 || lambda_max <= 0.0 {
        return None;
    }

    // Eigenvector of lambda_max gives the major axis angle.
    // For symmetric 2x2: angle = 0.5 * atan2(2*cov_xy, cov_xx - cov_yy)
    let angle = 0.5 * (2.0 * cov_xy).atan2(cov_xx - cov_yy);

    // 2σ semi-axes.
    let semi_major = 2.0 * lambda_max.sqrt();
    let semi_minor = 2.0 * lambda_min.sqrt();
    Some((semi_major, semi_minor, angle))
}

/// Polyline samples along an ellipse rim, suitable for `egui_plot::Line`.
///
/// `center` is the ellipse center in plot coordinates. `samples` is the
/// number of vertices in the polyline (closed; first vertex is
/// duplicated at the end so the line draws as a closed shape).
pub fn ellipse_polyline(
    center: [f64; 2],
    semi_major: f64,
    semi_minor: f64,
    angle_rad: f64,
    samples: usize,
) -> Vec<[f64; 2]> {
    let n = samples.max(8);
    let (sin_a, cos_a) = angle_rad.sin_cos();
    let mut pts = Vec::with_capacity(n + 1);
    for i in 0..=n {
        let theta = (i as f64) * std::f64::consts::TAU / n as f64;
        let x = semi_major * theta.cos();
        let y = semi_minor * theta.sin();
        let x_rot = cos_a * x - sin_a * y;
        let y_rot = sin_a * x + cos_a * y;
        pts.push([center[0] + x_rot, center[1] + y_rot]);
    }
    pts
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    fn close_to(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    #[test]
    fn identity_covariance_yields_unit_circle_at_2sigma() {
        // cov = I → both eigenvalues 1, so semi-axes are 2*sqrt(1) = 2.
        let (maj, min, _angle) = ellipse_axes(1.0, 0.0, 1.0).unwrap();
        assert!(close_to(maj, 2.0, 1e-9));
        assert!(close_to(min, 2.0, 1e-9));
    }

    #[test]
    fn anisotropic_diagonal_yields_axis_aligned_ellipse() {
        // cov = diag(4, 1) → eigenvalues 4 and 1, semi-axes 4 and 2.
        let (maj, min, angle) = ellipse_axes(4.0, 0.0, 1.0).unwrap();
        assert!(close_to(maj, 4.0, 1e-9));
        assert!(close_to(min, 2.0, 1e-9));
        // Major axis along x → angle 0.
        assert!(close_to(angle, 0.0, 1e-9));
    }

    #[test]
    fn rotated_45_degrees_when_off_diagonal_dominates() {
        // cov = [[2, 1], [1, 2]] → eigenvalues 3, 1; angle = 45°.
        let (maj, min, angle) = ellipse_axes(2.0, 1.0, 2.0).unwrap();
        assert!(close_to(maj, 2.0 * (3.0_f64).sqrt(), 1e-9));
        assert!(close_to(min, 2.0, 1e-9));
        assert!(close_to(angle, PI / 4.0, 1e-9));
    }

    #[test]
    fn degenerate_covariance_returns_none() {
        // Zero variance.
        assert!(ellipse_axes(0.0, 0.0, 0.0).is_none());
        // Negative eigenvalue (cov_xy too large for given diagonals).
        assert!(ellipse_axes(1.0, 5.0, 1.0).is_none());
        // NaN.
        assert!(ellipse_axes(f64::NAN, 0.0, 1.0).is_none());
    }

    #[test]
    fn polyline_is_closed_with_n_plus_one_points() {
        let pts = ellipse_polyline([10.0, 20.0], 5.0, 3.0, 0.0, 32);
        assert_eq!(pts.len(), 33);
        // First and last vertex coincide.
        assert!(close_to(pts[0][0], pts[32][0], 1e-9));
        assert!(close_to(pts[0][1], pts[32][1], 1e-9));
    }

    #[test]
    fn polyline_centered_at_requested_point() {
        let pts = ellipse_polyline([100.0, 200.0], 5.0, 3.0, 0.0, 64);
        // Mean of polyline (excluding the duplicated last vertex) should
        // be very close to the center.
        let n = pts.len() - 1;
        let mean_x: f64 = pts[..n].iter().map(|p| p[0]).sum::<f64>() / n as f64;
        let mean_y: f64 = pts[..n].iter().map(|p| p[1]).sum::<f64>() / n as f64;
        assert!(close_to(mean_x, 100.0, 1e-6));
        assert!(close_to(mean_y, 200.0, 1e-6));
    }
}
