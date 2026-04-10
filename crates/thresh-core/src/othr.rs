//! OTHR (Over-The-Horizon Radar) sensor registration and coordinate transforms.

use crate::geodetic::{WGS84_A, WGS84_F, ecef_to_enu, wgs84_to_ecef};
use nalgebra::Vector3;
use serde::{Deserialize, Serialize};

/// Registration data for an OTHR sensor (transmitter location and operating parameters).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OthrSensorRegistration {
    /// Transmitter geodetic latitude (radians).
    pub transmitter_lat_rad: f64,
    /// Transmitter geodetic longitude (radians).
    pub transmitter_lon_rad: f64,
    /// Transmitter altitude above WGS84 ellipsoid (meters).
    pub transmitter_alt_m: f64,
    /// Operating frequency (MHz).
    pub operating_freq_mhz: f64,
}

// ── WGS84 derived constants for Vincenty ───────────────────────────────────

/// Semi-minor axis.
const B: f64 = WGS84_A * (1.0 - WGS84_F);

/// Convergence tolerance for Vincenty iteration.
const VINCENTY_TOL: f64 = 1e-12;

/// Maximum iterations for Vincenty.
const VINCENTY_MAX_ITER: usize = 1000;

// ── Vincenty's direct formula ──────────────────────────────────────────────

/// Given a start point (lat, lon in radians), forward azimuth (radians, clockwise from north),
/// and geodesic distance (meters), compute the endpoint on the WGS84 ellipsoid.
///
/// Returns `(lat2_rad, lon2_rad)`.
pub fn vincenty_direct(
    lat1_rad: f64,
    lon1_rad: f64,
    azimuth_rad: f64,
    distance_m: f64,
) -> (f64, f64) {
    let a = WGS84_A;
    let f = WGS84_F;
    let b = B;

    let tan_u1 = (1.0 - f) * lat1_rad.tan();
    let cos_u1 = 1.0 / (1.0 + tan_u1 * tan_u1).sqrt();
    let sin_u1 = tan_u1 * cos_u1;

    let sin_alpha1 = azimuth_rad.sin();
    let cos_alpha1 = azimuth_rad.cos();

    let sigma1 = tan_u1.atan2(cos_alpha1);
    let sin_alpha = cos_u1 * sin_alpha1;
    let cos_sq_alpha = 1.0 - sin_alpha * sin_alpha;
    let u_sq = cos_sq_alpha * (a * a - b * b) / (b * b);

    let cap_a = 1.0 + u_sq / 16384.0 * (4096.0 + u_sq * (-768.0 + u_sq * (320.0 - 175.0 * u_sq)));
    let cap_b = u_sq / 1024.0 * (256.0 + u_sq * (-128.0 + u_sq * (74.0 - 47.0 * u_sq)));

    let mut sigma = distance_m / (b * cap_a);

    for _ in 0..VINCENTY_MAX_ITER {
        let cos_2sigma_m = (2.0 * sigma1 + sigma).cos();
        let sin_sigma = sigma.sin();
        let cos_sigma = sigma.cos();

        let delta_sigma = cap_b
            * sin_sigma
            * (cos_2sigma_m
                + cap_b / 4.0
                    * (cos_sigma * (-1.0 + 2.0 * cos_2sigma_m * cos_2sigma_m)
                        - cap_b / 6.0
                            * cos_2sigma_m
                            * (-3.0 + 4.0 * sin_sigma * sin_sigma)
                            * (-3.0 + 4.0 * cos_2sigma_m * cos_2sigma_m)));

        let sigma_new = distance_m / (b * cap_a) + delta_sigma;
        if (sigma_new - sigma).abs() < VINCENTY_TOL {
            sigma = sigma_new;
            break;
        }
        sigma = sigma_new;
    }

    let sin_sigma = sigma.sin();
    let cos_sigma = sigma.cos();
    let cos_2sigma_m = (2.0 * sigma1 + sigma).cos();

    let lat2 = (sin_u1 * cos_sigma + cos_u1 * sin_sigma * cos_alpha1).atan2(
        (1.0 - f)
            * (sin_alpha * sin_alpha
                + (sin_u1 * sin_sigma - cos_u1 * cos_sigma * cos_alpha1).powi(2))
            .sqrt(),
    );

    let lambda =
        (sin_sigma * sin_alpha1).atan2(cos_u1 * cos_sigma - sin_u1 * sin_sigma * cos_alpha1);

    let cap_c = f / 16.0 * cos_sq_alpha * (4.0 + f * (4.0 - 3.0 * cos_sq_alpha));
    let cap_l = lambda
        - (1.0 - cap_c)
            * f
            * sin_alpha
            * (sigma
                + cap_c
                    * sin_sigma
                    * (cos_2sigma_m
                        + cap_c * cos_sigma * (-1.0 + 2.0 * cos_2sigma_m * cos_2sigma_m)));

    let lon2 = lon1_rad + cap_l;

    (lat2, lon2)
}

// ── Vincenty's inverse formula ─────────────────────────────────────────────

/// Given two points on the WGS84 ellipsoid, compute the geodesic distance (meters)
/// and forward azimuth (radians, clockwise from north) from point 1 to point 2.
///
/// Returns `(distance_m, azimuth_rad)`.
///
/// For nearly antipodal points, convergence may be slow; the function returns
/// an approximate great-circle result if iteration does not converge.
pub fn vincenty_inverse(lat1_rad: f64, lon1_rad: f64, lat2_rad: f64, lon2_rad: f64) -> (f64, f64) {
    let a = WGS84_A;
    let f = WGS84_F;
    let b = B;

    let u1 = ((1.0 - f) * lat1_rad.tan()).atan();
    let u2 = ((1.0 - f) * lat2_rad.tan()).atan();

    let sin_u1 = u1.sin();
    let cos_u1 = u1.cos();
    let sin_u2 = u2.sin();
    let cos_u2 = u2.cos();

    let cap_l = lon2_rad - lon1_rad;
    let mut lambda = cap_l;

    let mut sin_sigma;
    let mut cos_sigma;
    let mut sigma;
    let mut sin_alpha;
    let mut cos_sq_alpha;
    let mut cos_2sigma_m;

    let mut converged = false;

    for _ in 0..VINCENTY_MAX_ITER {
        let sin_lambda = lambda.sin();
        let cos_lambda = lambda.cos();

        sin_sigma = ((cos_u2 * sin_lambda).powi(2)
            + (cos_u1 * sin_u2 - sin_u1 * cos_u2 * cos_lambda).powi(2))
        .sqrt();

        if sin_sigma.abs() < 1e-15 {
            // Co-incident points
            return (0.0, 0.0);
        }

        cos_sigma = sin_u1 * sin_u2 + cos_u1 * cos_u2 * cos_lambda;
        sigma = sin_sigma.atan2(cos_sigma);

        sin_alpha = cos_u1 * cos_u2 * sin_lambda / sin_sigma;
        cos_sq_alpha = 1.0 - sin_alpha * sin_alpha;

        cos_2sigma_m = if cos_sq_alpha.abs() < 1e-15 {
            0.0 // Equatorial line
        } else {
            cos_sigma - 2.0 * sin_u1 * sin_u2 / cos_sq_alpha
        };

        let cap_c = f / 16.0 * cos_sq_alpha * (4.0 + f * (4.0 - 3.0 * cos_sq_alpha));

        let lambda_new = cap_l
            + (1.0 - cap_c)
                * f
                * sin_alpha
                * (sigma
                    + cap_c
                        * sin_sigma
                        * (cos_2sigma_m
                            + cap_c * cos_sigma * (-1.0 + 2.0 * cos_2sigma_m * cos_2sigma_m)));

        if (lambda_new - lambda).abs() < VINCENTY_TOL {
            lambda = lambda_new;
            converged = true;
            break;
        }
        lambda = lambda_new;
    }

    if !converged {
        // Fallback for near-antipodal: use spherical approximation
        let dlat = lat2_rad - lat1_rad;
        let dlon = lon2_rad - lon1_rad;
        let aa = (dlat / 2.0).sin().powi(2)
            + lat1_rad.cos() * lat2_rad.cos() * (dlon / 2.0).sin().powi(2);
        let c = 2.0 * aa.sqrt().atan2((1.0 - aa).sqrt());
        let dist = a * c;
        let y = dlon.sin() * lat2_rad.cos();
        let x = lat1_rad.cos() * lat2_rad.sin() - lat1_rad.sin() * lat2_rad.cos() * dlon.cos();
        let az = y.atan2(x);
        let az = (az + std::f64::consts::TAU) % std::f64::consts::TAU;
        return (dist, az);
    }

    // Recompute final values with converged lambda
    let sin_lambda = lambda.sin();
    let cos_lambda = lambda.cos();

    sin_sigma = ((cos_u2 * sin_lambda).powi(2)
        + (cos_u1 * sin_u2 - sin_u1 * cos_u2 * cos_lambda).powi(2))
    .sqrt();
    cos_sigma = sin_u1 * sin_u2 + cos_u1 * cos_u2 * cos_lambda;
    sigma = sin_sigma.atan2(cos_sigma);

    sin_alpha = cos_u1 * cos_u2 * sin_lambda / sin_sigma;
    cos_sq_alpha = 1.0 - sin_alpha * sin_alpha;

    cos_2sigma_m = if cos_sq_alpha.abs() < 1e-15 {
        0.0
    } else {
        cos_sigma - 2.0 * sin_u1 * sin_u2 / cos_sq_alpha
    };

    let u_sq = cos_sq_alpha * (a * a - b * b) / (b * b);
    let cap_a = 1.0 + u_sq / 16384.0 * (4096.0 + u_sq * (-768.0 + u_sq * (320.0 - 175.0 * u_sq)));
    let cap_b_val = u_sq / 1024.0 * (256.0 + u_sq * (-128.0 + u_sq * (74.0 - 47.0 * u_sq)));

    let delta_sigma = cap_b_val
        * sin_sigma
        * (cos_2sigma_m
            + cap_b_val / 4.0
                * (cos_sigma * (-1.0 + 2.0 * cos_2sigma_m * cos_2sigma_m)
                    - cap_b_val / 6.0
                        * cos_2sigma_m
                        * (-3.0 + 4.0 * sin_sigma * sin_sigma)
                        * (-3.0 + 4.0 * cos_2sigma_m * cos_2sigma_m)));

    let distance = b * cap_a * (sigma - delta_sigma);

    let azimuth = (cos_u2 * sin_lambda).atan2(cos_u1 * sin_u2 - sin_u1 * cos_u2 * cos_lambda);
    let azimuth = (azimuth + std::f64::consts::TAU) % std::f64::consts::TAU;

    (distance, azimuth)
}

// ── OTHR coordinate conversions ────────────────────────────────────────────

/// Convert an OTHR measurement (ground range and azimuth) to geodetic coordinates.
///
/// Uses Vincenty's direct formula from the transmitter position.
/// Returns `(lat_rad, lon_rad)` of the target on the WGS84 ellipsoid surface.
pub fn othr_to_geodetic(
    registration: &OthrSensorRegistration,
    ground_range_m: f64,
    azimuth_rad: f64,
) -> (f64, f64) {
    vincenty_direct(
        registration.transmitter_lat_rad,
        registration.transmitter_lon_rad,
        azimuth_rad,
        ground_range_m,
    )
}

/// Convert an OTHR measurement to East-North-Up (ENU) coordinates relative to a reference point.
///
/// The target's geodetic position is computed via `othr_to_geodetic`, then converted
/// through ECEF to ENU. An estimated altitude must be provided since OTHR does not
/// directly measure altitude.
pub fn othr_to_enu(
    registration: &OthrSensorRegistration,
    ground_range_m: f64,
    azimuth_rad: f64,
    estimated_alt_m: f64,
    ref_lat_rad: f64,
    ref_lon_rad: f64,
    ref_alt_m: f64,
) -> Vector3<f64> {
    let (lat, lon) = othr_to_geodetic(registration, ground_range_m, azimuth_rad);
    let ecef = wgs84_to_ecef(lat, lon, estimated_alt_m);
    ecef_to_enu(&ecef, ref_lat_rad, ref_lon_rad, ref_alt_m)
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::{FRAC_PI_2, PI};

    const CM: f64 = 0.01; // 1 cm tolerance

    // ── Task 3.6 — Vincenty direct/inverse roundtrip ───────────────────

    #[test]
    fn vincenty_roundtrip_100km() {
        vincenty_roundtrip_at_distance(100_000.0);
    }

    #[test]
    fn vincenty_roundtrip_1000km() {
        vincenty_roundtrip_at_distance(1_000_000.0);
    }

    #[test]
    fn vincenty_roundtrip_3000km() {
        vincenty_roundtrip_at_distance(3_000_000.0);
    }

    fn vincenty_roundtrip_at_distance(dist_m: f64) {
        let lat1 = 40.0_f64.to_radians();
        let lon1 = (-74.0_f64).to_radians();
        let az_fwd = 45.0_f64.to_radians();

        let (lat2, lon2) = vincenty_direct(lat1, lon1, az_fwd, dist_m);
        let (inv_dist, _inv_az) = vincenty_inverse(lat1, lon1, lat2, lon2);

        assert!(
            (inv_dist - dist_m).abs() < CM,
            "roundtrip distance error at {dist_m}m: {} m",
            (inv_dist - dist_m).abs()
        );
    }

    // ── Task 3.7 — OTHR registration at 2000 km ───────────────────────

    #[test]
    fn othr_registration_2000km_due_north() {
        // Transmitter at equator, prime meridian
        let reg = OthrSensorRegistration {
            transmitter_lat_rad: 0.0,
            transmitter_lon_rad: 0.0,
            transmitter_alt_m: 0.0,
            operating_freq_mhz: 15.0,
        };

        let (lat, lon) = othr_to_geodetic(&reg, 2_000_000.0, 0.0);

        // Due north from equator: longitude should stay ~0, latitude should be ~18°
        assert!(
            lon.abs() < 1e-6,
            "longitude should be near 0 for due north: {lon}"
        );
        assert!(lat > 0.0, "latitude should be positive for due north");

        // Verify by inverse
        let (inv_dist, _) = vincenty_inverse(0.0, 0.0, lat, lon);
        assert!(
            (inv_dist - 2_000_000.0).abs() < CM,
            "distance roundtrip: {inv_dist}"
        );
    }

    // ── Various azimuths ───────────────────────────────────────────────

    #[test]
    fn othr_various_azimuths() {
        let reg = OthrSensorRegistration {
            transmitter_lat_rad: 35.0_f64.to_radians(),
            transmitter_lon_rad: (-100.0_f64).to_radians(),
            transmitter_alt_m: 0.0,
            operating_freq_mhz: 15.0,
        };

        let dist = 1_500_000.0;

        // Test 0° (north), 90° (east), 180° (south), 270° (west)
        for &az_deg in &[0.0_f64, 90.0, 180.0, 270.0] {
            let az_rad = az_deg.to_radians();
            let (lat, lon) = othr_to_geodetic(&reg, dist, az_rad);

            // Roundtrip via inverse
            let (inv_dist, inv_az) =
                vincenty_inverse(reg.transmitter_lat_rad, reg.transmitter_lon_rad, lat, lon);
            assert!(
                (inv_dist - dist).abs() < CM,
                "distance roundtrip at az={az_deg}°: error={} m",
                (inv_dist - dist).abs()
            );

            // Azimuth check (modulo 2pi)
            let az_err = ((inv_az - az_rad + PI) % (2.0 * PI) - PI).abs();
            assert!(
                az_err < 1e-8,
                "azimuth roundtrip at az={az_deg}°: error={az_err} rad"
            );
        }
    }

    // ── Antipodal edge case ────────────────────────────────────────────

    #[test]
    fn vincenty_inverse_near_antipodal() {
        // Nearly antipodal points — should not panic and should return reasonable result
        let lat1 = 0.0_f64;
        let lon1 = 0.0_f64;
        let lat2 = 0.0_f64;
        let lon2 = PI - 0.001; // nearly antipodal

        let (dist, _az) = vincenty_inverse(lat1, lon1, lat2, lon2);

        // Distance should be close to half circumference
        let half_circ = PI * WGS84_A;
        assert!(
            (dist - half_circ).abs() / half_circ < 0.01,
            "near-antipodal distance: {dist}, expected ~{half_circ}"
        );
    }

    #[test]
    fn vincenty_inverse_coincident() {
        let (dist, _az) = vincenty_inverse(0.5, 1.0, 0.5, 1.0);
        assert!(dist.abs() < 1e-10, "coincident distance: {dist}");
    }

    // ── OTHR to ENU ────────────────────────────────────────────────────

    #[test]
    fn othr_to_enu_basic() {
        let reg = OthrSensorRegistration {
            transmitter_lat_rad: 40.0_f64.to_radians(),
            transmitter_lon_rad: (-74.0_f64).to_radians(),
            transmitter_alt_m: 0.0,
            operating_freq_mhz: 15.0,
        };

        // Use transmitter position as reference
        let enu = othr_to_enu(
            &reg,
            100_000.0, // 100 km north
            0.0,       // due north
            0.0,       // sea level
            reg.transmitter_lat_rad,
            reg.transmitter_lon_rad,
            reg.transmitter_alt_m,
        );

        // Due north => E ~0, N ~100 km, U ~0 (approximately, curved earth)
        assert!(
            enu.x.abs() < 100.0,
            "East should be near 0 for due north: {}",
            enu.x
        );
        assert!(
            (enu.y - 100_000.0).abs() < 1000.0,
            "North should be ~100 km: {}",
            enu.y
        );
    }

    // ── Direct formula specific azimuths ───────────────────────────────

    #[test]
    fn vincenty_direct_due_east_on_equator() {
        let (lat2, lon2) = vincenty_direct(0.0, 0.0, FRAC_PI_2, 1_000_000.0);

        // Should stay on equator
        assert!(
            lat2.abs() < 1e-6,
            "latitude should stay near 0 for due east on equator: {lat2}"
        );
        // Longitude should increase
        assert!(lon2 > 0.0, "longitude should increase for due east");

        // Roundtrip
        let (inv_dist, _) = vincenty_inverse(0.0, 0.0, lat2, lon2);
        assert!(
            (inv_dist - 1_000_000.0).abs() < CM,
            "roundtrip error: {}",
            (inv_dist - 1_000_000.0).abs()
        );
    }

    // ── Serialization ──────────────────────────────────────────────────

    #[test]
    fn registration_serialization_roundtrip() {
        let reg = OthrSensorRegistration {
            transmitter_lat_rad: 0.7,
            transmitter_lon_rad: -1.2,
            transmitter_alt_m: 50.0,
            operating_freq_mhz: 12.5,
        };
        let json = serde_json::to_string(&reg).expect("serialize");
        let reg2: OthrSensorRegistration = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(reg.transmitter_lat_rad, reg2.transmitter_lat_rad);
        assert_eq!(reg.transmitter_lon_rad, reg2.transmitter_lon_rad);
        assert_eq!(reg.transmitter_alt_m, reg2.transmitter_alt_m);
        assert_eq!(reg.operating_freq_mhz, reg2.operating_freq_mhz);
    }
}
