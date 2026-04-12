//! Coordinate transformations (polar/Cartesian).

use nalgebra::Vector3;

/// Convert spherical (range, azimuth, elevation) to Cartesian (x, y, z).
///
/// Convention: azimuth is measured from +x axis toward +y, elevation from the x-y plane.
pub fn spherical_to_cartesian(range: f64, azimuth: f64, elevation: f64) -> Vector3<f64> {
    let cos_el = elevation.cos();
    Vector3::new(
        range * cos_el * azimuth.cos(),
        range * cos_el * azimuth.sin(),
        range * elevation.sin(),
    )
}

/// Convert Cartesian (x, y, z) to spherical (range, azimuth, elevation).
///
/// Returns (range, azimuth, elevation).
pub fn cartesian_to_spherical(pos: &Vector3<f64>) -> (f64, f64, f64) {
    let range = pos.norm();
    if range < 1e-15 {
        return (0.0, 0.0, 0.0);
    }
    let azimuth = pos.y.atan2(pos.x);
    let elevation = (pos.z / range).asin();
    (range, azimuth, elevation)
}

/// Convert polar 2D (range, bearing) to Cartesian (x, y).
pub fn polar_to_cartesian_2d(range: f64, bearing: f64) -> (f64, f64) {
    (range * bearing.cos(), range * bearing.sin())
}

/// Convert Cartesian 2D (x, y) to polar (range, bearing).
pub fn cartesian_to_polar_2d(x: f64, y: f64) -> (f64, f64) {
    let range = (x * x + y * y).sqrt();
    let bearing = y.atan2(x);
    (range, bearing)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::{FRAC_PI_2, FRAC_PI_4, PI};

    const TOL: f64 = 1e-10;

    #[test]
    fn spherical_cartesian_roundtrip() {
        let range = 10000.0;
        let az = 0.3;
        let el = 0.1;
        let cart = spherical_to_cartesian(range, az, el);
        let (r2, a2, e2) = cartesian_to_spherical(&cart);
        assert!((r2 - range).abs() < TOL);
        assert!((a2 - az).abs() < TOL);
        assert!((e2 - el).abs() < TOL);
    }

    #[test]
    fn along_x_axis() {
        let cart = spherical_to_cartesian(100.0, 0.0, 0.0);
        assert!((cart.x - 100.0).abs() < TOL);
        assert!(cart.y.abs() < TOL);
        assert!(cart.z.abs() < TOL);
    }

    #[test]
    fn along_z_axis() {
        let cart = spherical_to_cartesian(100.0, 0.0, FRAC_PI_2);
        assert!(cart.x.abs() < TOL);
        assert!(cart.y.abs() < TOL);
        assert!((cart.z - 100.0).abs() < TOL);
    }

    #[test]
    fn polar_2d_roundtrip() {
        let (x, y) = polar_to_cartesian_2d(50.0, FRAC_PI_4);
        let (r, theta) = cartesian_to_polar_2d(x, y);
        assert!((r - 50.0).abs() < TOL);
        assert!((theta - FRAC_PI_4).abs() < TOL);
    }

    #[test]
    fn zero_range() {
        let (r, az, el) = cartesian_to_spherical(&Vector3::zeros());
        assert_eq!(r, 0.0);
        assert_eq!(az, 0.0);
        assert_eq!(el, 0.0);
    }

    #[test]
    fn negative_y_azimuth() {
        let cart = spherical_to_cartesian(100.0, -PI / 4.0, 0.0);
        let (_, az, _) = cartesian_to_spherical(&cart);
        assert!((az - (-PI / 4.0)).abs() < TOL);
    }
}
