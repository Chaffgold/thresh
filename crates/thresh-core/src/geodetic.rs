//! Geodetic coordinate transformations (WGS84, ECEF, ENU).

use nalgebra::{Matrix3, Vector3};

// ── WGS84 ellipsoid constants ───────────────────────────────────────────────

/// Semi-major axis (equatorial radius) in meters.
pub const WGS84_A: f64 = 6_378_137.0;

/// Flattening.
pub const WGS84_F: f64 = 1.0 / 298.257_223_563;

/// Semi-minor axis (polar radius) in meters.
pub const WGS84_B: f64 = WGS84_A * (1.0 - WGS84_F);

/// First eccentricity squared.
pub const WGS84_E2: f64 = 2.0 * WGS84_F - WGS84_F * WGS84_F;

// ── Helpers ─────────────────────────────────────────────────────────────────

/// Radius of curvature in the prime vertical.
fn prime_vertical_radius(sin_lat: f64) -> f64 {
    WGS84_A / (1.0 - WGS84_E2 * sin_lat * sin_lat).sqrt()
}

/// Build the ECEF-to-ENU rotation matrix for a given reference latitude and longitude.
fn enu_rotation(sin_lat: f64, cos_lat: f64, sin_lon: f64, cos_lon: f64) -> Matrix3<f64> {
    // Rows: E, N, U
    Matrix3::new(
        -sin_lon,
        cos_lon,
        0.0,
        -sin_lat * cos_lon,
        -sin_lat * sin_lon,
        cos_lat,
        cos_lat * cos_lon,
        cos_lat * sin_lon,
        sin_lat,
    )
}

// ── Public API ──────────────────────────────────────────────────────────────

/// Convert geodetic (WGS84) coordinates to ECEF.
///
/// * `lat_rad` — geodetic latitude in radians
/// * `lon_rad` — geodetic longitude in radians
/// * `alt_m`   — altitude above the ellipsoid in meters
pub fn wgs84_to_ecef(lat_rad: f64, lon_rad: f64, alt_m: f64) -> Vector3<f64> {
    let sin_lat = lat_rad.sin();
    let cos_lat = lat_rad.cos();
    let sin_lon = lon_rad.sin();
    let cos_lon = lon_rad.cos();
    let n = prime_vertical_radius(sin_lat);

    Vector3::new(
        (n + alt_m) * cos_lat * cos_lon,
        (n + alt_m) * cos_lat * sin_lon,
        (n * (1.0 - WGS84_E2) + alt_m) * sin_lat,
    )
}

/// Convert ECEF coordinates back to geodetic (WGS84) using iterative method.
///
/// Returns `(lat_rad, lon_rad, alt_m)`.
pub fn ecef_to_wgs84(pos: &Vector3<f64>) -> (f64, f64, f64) {
    let x = pos.x;
    let y = pos.y;
    let z = pos.z;

    let lon = y.atan2(x);
    let p = (x * x + y * y).sqrt();

    // Initial latitude estimate (Bowring's starting approximation)
    let theta = z.atan2(p * (1.0 - WGS84_E2));
    let ep2 = (WGS84_A * WGS84_A - WGS84_B * WGS84_B) / (WGS84_B * WGS84_B);

    let mut lat = (z + ep2 * WGS84_B * theta.sin().powi(3))
        .atan2(p - WGS84_E2 * WGS84_A * theta.cos().powi(3));

    // Iterate to convergence
    for _ in 0..20 {
        let sin_lat = lat.sin();
        let n = prime_vertical_radius(sin_lat);
        let new_lat = (z + WGS84_E2 * n * sin_lat).atan2(p);
        if (new_lat - lat).abs() < 1e-14 {
            lat = new_lat;
            break;
        }
        lat = new_lat;
    }

    let sin_lat = lat.sin();
    let cos_lat = lat.cos();
    let n = prime_vertical_radius(sin_lat);
    let alt = if cos_lat.abs() > 1e-10 {
        p / cos_lat - n
    } else {
        z / sin_lat - n * (1.0 - WGS84_E2)
    };

    (lat, lon, alt)
}

/// Convert an ECEF position to East-North-Up (ENU) relative to a reference point.
pub fn ecef_to_enu(
    pos_ecef: &Vector3<f64>,
    ref_lat_rad: f64,
    ref_lon_rad: f64,
    ref_alt_m: f64,
) -> Vector3<f64> {
    let ref_ecef = wgs84_to_ecef(ref_lat_rad, ref_lon_rad, ref_alt_m);
    let delta = pos_ecef - ref_ecef;

    let sin_lat = ref_lat_rad.sin();
    let cos_lat = ref_lat_rad.cos();
    let sin_lon = ref_lon_rad.sin();
    let cos_lon = ref_lon_rad.cos();

    let rot = enu_rotation(sin_lat, cos_lat, sin_lon, cos_lon);
    rot * delta
}

/// Convert East-North-Up (ENU) coordinates back to ECEF given a reference point.
pub fn enu_to_ecef(
    enu: &Vector3<f64>,
    ref_lat_rad: f64,
    ref_lon_rad: f64,
    ref_alt_m: f64,
) -> Vector3<f64> {
    let ref_ecef = wgs84_to_ecef(ref_lat_rad, ref_lon_rad, ref_alt_m);

    let sin_lat = ref_lat_rad.sin();
    let cos_lat = ref_lat_rad.cos();
    let sin_lon = ref_lon_rad.sin();
    let cos_lon = ref_lon_rad.cos();

    let rot = enu_rotation(sin_lat, cos_lat, sin_lon, cos_lon);
    // Inverse of orthogonal matrix is its transpose
    ref_ecef + rot.transpose() * enu
}

/// Convenience: convert geodetic (WGS84) directly to ENU relative to a reference point.
pub fn wgs84_to_enu(
    lat_rad: f64,
    lon_rad: f64,
    alt_m: f64,
    ref_lat_rad: f64,
    ref_lon_rad: f64,
    ref_alt_m: f64,
) -> Vector3<f64> {
    let ecef = wgs84_to_ecef(lat_rad, lon_rad, alt_m);
    ecef_to_enu(&ecef, ref_lat_rad, ref_lon_rad, ref_alt_m)
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::FRAC_PI_2;

    const TOL_M: f64 = 1e-3; // 1 mm

    // ── WGS84 constants sanity ──────────────────────────────────────────

    #[test]
    fn wgs84_semi_minor_axis() {
        let expected_b = 6_356_752.314_245_179;
        assert!((WGS84_B - expected_b).abs() < 0.001);
    }

    #[test]
    fn wgs84_eccentricity_squared() {
        // e² ≈ 0.00669437999014
        assert!((WGS84_E2 - 0.006_694_379_990_14).abs() < 1e-12);
    }

    // ── Task 2.9 — JFK airport ──────────────────────────────────────────

    #[test]
    fn jfk_wgs84_to_ecef() {
        let lat = 40.6413_f64.to_radians();
        let lon = (-73.7781_f64).to_radians();
        let alt = 13.0;

        let ecef = wgs84_to_ecef(lat, lon, alt);

        // Verify ECEF is on the WGS84 ellipsoid at correct location.
        // Sanity checks: x positive (western hemisphere, lon > -90), y negative, z positive.
        assert!(ecef.x > 0.0, "x should be positive");
        assert!(ecef.y < 0.0, "y should be negative (western hemisphere)");
        assert!(ecef.z > 0.0, "z should be positive (northern hemisphere)");

        // Magnitude should be approximately Earth's radius (~6.37e6 m)
        let r = ecef.norm();
        assert!((r - 6_371_000.0).abs() < 50_000.0, "radius: {r}");

        // Verify against independently computed ECEF (WGS84 formula):
        //   x ≈ 1,353,948 m, y ≈ -4,653,681 m, z ≈ 4,132,287 m
        assert!((ecef.x - 1_353_948.0).abs() < 10.0, "x: {}", ecef.x);
        assert!((ecef.y - (-4_653_681.0)).abs() < 10.0, "y: {}", ecef.y);
        assert!((ecef.z - 4_132_287.0).abs() < 10.0, "z: {}", ecef.z);

        // Roundtrip check: ECEF → WGS84 should recover the original coordinates
        let (lat2, lon2, alt2) = ecef_to_wgs84(&ecef);
        assert!((lat2 - lat).abs() < 1e-10, "lat roundtrip");
        assert!((lon2 - lon).abs() < 1e-10, "lon roundtrip");
        assert!((alt2 - alt).abs() < 0.01, "alt roundtrip");
    }

    // ── Task 2.10 — ENU roundtrip ───────────────────────────────────────

    #[test]
    fn enu_roundtrip() {
        // Point near JFK
        let lat = 40.6413_f64.to_radians();
        let lon = (-73.7781_f64).to_radians();
        let alt = 13.0;

        // Reference: nearby point
        let ref_lat = 40.65_f64.to_radians();
        let ref_lon = (-73.78_f64).to_radians();
        let ref_alt = 10.0;

        let ecef = wgs84_to_ecef(lat, lon, alt);
        let enu = ecef_to_enu(&ecef, ref_lat, ref_lon, ref_alt);
        let ecef_back = enu_to_ecef(&enu, ref_lat, ref_lon, ref_alt);

        assert!((ecef.x - ecef_back.x).abs() < TOL_M, "x roundtrip");
        assert!((ecef.y - ecef_back.y).abs() < TOL_M, "y roundtrip");
        assert!((ecef.z - ecef_back.z).abs() < TOL_M, "z roundtrip");
    }

    #[test]
    fn wgs84_ecef_roundtrip() {
        let lat = 40.6413_f64.to_radians();
        let lon = (-73.7781_f64).to_radians();
        let alt = 13.0;

        let ecef = wgs84_to_ecef(lat, lon, alt);
        let (lat2, lon2, alt2) = ecef_to_wgs84(&ecef);

        assert!((lat2 - lat).abs() < 1e-10, "lat roundtrip: {lat2} vs {lat}");
        assert!((lon2 - lon).abs() < 1e-10, "lon roundtrip: {lon2} vs {lon}");
        assert!((alt2 - alt).abs() < 0.01, "alt roundtrip: {alt2} vs {alt}");
    }

    #[test]
    fn full_wgs84_enu_roundtrip() {
        let lat = 40.6413_f64.to_radians();
        let lon = (-73.7781_f64).to_radians();
        let alt = 13.0;

        let ref_lat = 40.65_f64.to_radians();
        let ref_lon = (-73.78_f64).to_radians();
        let ref_alt = 10.0;

        let enu = wgs84_to_enu(lat, lon, alt, ref_lat, ref_lon, ref_alt);
        let ecef_back = enu_to_ecef(&enu, ref_lat, ref_lon, ref_alt);
        let (lat2, lon2, alt2) = ecef_to_wgs84(&ecef_back);

        // Roundtrip error < 1 cm
        let ecef_orig = wgs84_to_ecef(lat, lon, alt);
        let ecef_rt = wgs84_to_ecef(lat2, lon2, alt2);
        let err = (ecef_orig - ecef_rt).norm();
        assert!(err < 0.01, "roundtrip error: {err} m");
    }

    // ── ENU at reference point should be zero ───────────────────────────

    #[test]
    fn enu_at_reference_is_zero() {
        let lat = 51.5_f64.to_radians();
        let lon = (-0.1_f64).to_radians();
        let alt = 50.0;

        let enu = wgs84_to_enu(lat, lon, alt, lat, lon, alt);
        assert!(enu.norm() < 1e-9, "ENU at ref: {enu}");
    }

    // ── Edge cases ──────────────────────────────────────────────────────

    #[test]
    fn north_pole() {
        let ecef = wgs84_to_ecef(FRAC_PI_2, 0.0, 0.0);
        assert!(ecef.x.abs() < TOL_M);
        assert!(ecef.y.abs() < TOL_M);
        assert!((ecef.z - WGS84_B).abs() < TOL_M, "z at pole: {}", ecef.z);
    }

    #[test]
    fn equator_prime_meridian() {
        let ecef = wgs84_to_ecef(0.0, 0.0, 0.0);
        assert!((ecef.x - WGS84_A).abs() < TOL_M, "x at (0,0): {}", ecef.x);
        assert!(ecef.y.abs() < TOL_M);
        assert!(ecef.z.abs() < TOL_M);
    }

    #[test]
    fn equator_90_east() {
        let ecef = wgs84_to_ecef(0.0, FRAC_PI_2, 0.0);
        assert!(ecef.x.abs() < TOL_M);
        assert!(
            (ecef.y - WGS84_A).abs() < TOL_M,
            "y at equator 90E: {}",
            ecef.y
        );
        assert!(ecef.z.abs() < TOL_M);
    }

    #[test]
    fn south_pole_roundtrip() {
        let lat = -FRAC_PI_2;
        let lon = 0.0;
        let alt = 100.0;

        let ecef = wgs84_to_ecef(lat, lon, alt);
        let (lat2, lon2, alt2) = ecef_to_wgs84(&ecef);

        assert!((lat2 - lat).abs() < 1e-10, "south pole lat");
        // Longitude is undefined at pole, skip check
        let _ = lon2;
        assert!((alt2 - alt).abs() < 0.01, "south pole alt: {alt2}");
    }

    #[test]
    fn various_latitudes_roundtrip() {
        for &lat_deg in &[-80.0_f64, -45.0, -10.0, 0.0, 10.0, 45.0, 80.0] {
            for &lon_deg in &[-180.0_f64, -90.0, 0.0, 90.0, 179.0] {
                let lat = lat_deg.to_radians();
                let lon = lon_deg.to_radians();
                let alt = 500.0;

                let ecef = wgs84_to_ecef(lat, lon, alt);
                let (lat2, lon2, alt2) = ecef_to_wgs84(&ecef);

                assert!(
                    (lat2 - lat).abs() < 1e-10,
                    "lat roundtrip at ({lat_deg}, {lon_deg}): {lat2} vs {lat}"
                );
                assert!(
                    (lon2 - lon).abs() < 1e-10,
                    "lon roundtrip at ({lat_deg}, {lon_deg}): {lon2} vs {lon}"
                );
                assert!(
                    (alt2 - alt).abs() < 0.01,
                    "alt roundtrip at ({lat_deg}, {lon_deg}): {alt2} vs {alt}"
                );
            }
        }
    }
}
