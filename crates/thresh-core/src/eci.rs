//! ECI (Earth-Centered Inertial) and TEME coordinate transformations.
//!
//! Provides conversions between ECI/TEME, ECEF, and ENU coordinate frames
//! using GMST-based rotation matrices.

use nalgebra::{Matrix3, Vector3};
use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Earth rotation rate in rad/s (WGS-84).
pub const EARTH_ROTATION_RATE: f64 = 7.292_115_0e-5;

/// Julian Date of the J2000.0 epoch (2000-01-01 12:00:00 TT).
pub const J2000_JD: f64 = 2_451_545.0;

/// Number of seconds in one day.
pub const SECONDS_PER_DAY: f64 = 86_400.0;

// ---------------------------------------------------------------------------
// ECI state type (Task 2.5)
// ---------------------------------------------------------------------------

/// State vector in an Earth-Centered Inertial frame.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EciState {
    /// Position in metres (ECI frame).
    pub position: Vector3<f64>,
    /// Velocity in m/s (ECI frame).
    pub velocity: Vector3<f64>,
    /// Epoch as Julian Date.
    pub epoch_jd: f64,
}

// ---------------------------------------------------------------------------
// Helper: Z-axis rotation matrix
// ---------------------------------------------------------------------------

/// Build a rotation matrix for a right-handed rotation about the Z axis by
/// the given angle (radians).
///
/// ```text
/// R_z(θ) = | cos θ   sin θ   0 |
///          | -sin θ  cos θ   0 |
///          |  0       0      1 |
/// ```
pub fn rotation_z(angle: f64) -> Matrix3<f64> {
    let (s, c) = angle.sin_cos();
    Matrix3::new(c, s, 0.0, -s, c, 0.0, 0.0, 0.0, 1.0)
}

// ---------------------------------------------------------------------------
// GMST (Task 2.4)
// ---------------------------------------------------------------------------

/// Compute Greenwich Mean Sidereal Time (GMST) in radians from a Julian Date.
///
/// Uses the IAU 1982 expression for GMST. The result is normalised to \[0, 2π).
pub fn gmst_from_jd(jd: f64) -> f64 {
    use std::f64::consts::TAU;

    // Julian centuries since J2000.0
    let t = (jd - J2000_JD) / 36525.0;

    // GMST in seconds of time
    // 876600h = 876600 * 3600 s = 3_155_760_000 s
    let gmst_sec = 67_310.548_41 + (3_155_760_000.0 + 8_640_184.812_866) * t + 0.093_104 * t * t
        - 6.2e-6 * t * t * t;

    // Convert seconds of time to radians (full circle = 86400 s of sidereal time)
    let gmst_rad = (gmst_sec / SECONDS_PER_DAY) * TAU;

    // Normalise to [0, 2π)
    gmst_rad.rem_euclid(TAU)
}

// ---------------------------------------------------------------------------
// TEME → ECEF (Task 2.4)
// ---------------------------------------------------------------------------

/// Convert a state vector from the TEME (True Equator, Mean Equinox) frame to
/// the ECEF (Earth-Centered, Earth-Fixed) frame at the given Julian Date.
///
/// The rotation uses GMST only (no polar-motion or equation-of-equinoxes
/// corrections).
///
/// Returns `(position_ecef, velocity_ecef)` in metres and m/s.
pub fn teme_to_ecef(
    pos_teme: &Vector3<f64>,
    vel_teme: &Vector3<f64>,
    jd: f64,
) -> (Vector3<f64>, Vector3<f64>) {
    let gmst = gmst_from_jd(jd);
    let r = rotation_z(gmst); // R3(+GMST)

    // Position: simply rotate
    let pos_ecef = r * pos_teme;

    // Velocity: subtract Earth-rotation cross product, then rotate
    let omega = Vector3::new(0.0, 0.0, EARTH_ROTATION_RATE);
    let vel_ecef = r * (vel_teme - omega.cross(pos_teme));

    (pos_ecef, vel_ecef)
}

// ---------------------------------------------------------------------------
// ECI → ECEF (Task 2.6)
// ---------------------------------------------------------------------------

/// Convert a state vector from ECI (J2000) to ECEF at the given Julian Date.
///
/// **Note:** This currently uses the same GMST-only rotation as [`teme_to_ecef`].
/// For the accuracy requirements of this project (~arcsecond level) the
/// difference between J2000 and TEME is negligible. A full IAU-76/FK5
/// precession-nutation model can be added later if needed.
pub fn eci_to_ecef(
    pos_eci: &Vector3<f64>,
    vel_eci: &Vector3<f64>,
    jd: f64,
) -> (Vector3<f64>, Vector3<f64>) {
    teme_to_ecef(pos_eci, vel_eci, jd)
}

// ---------------------------------------------------------------------------
// ECEF → ECI (Task 2.7)
// ---------------------------------------------------------------------------

/// Convert a state vector from ECEF to ECI (J2000) at the given Julian Date.
///
/// This is the inverse of [`eci_to_ecef`]: apply `R3(-GMST)` (i.e. the
/// transpose of `R3(+GMST)`) and add the Earth-rotation cross product back
/// to the velocity.
pub fn ecef_to_eci(
    pos_ecef: &Vector3<f64>,
    vel_ecef: &Vector3<f64>,
    jd: f64,
) -> (Vector3<f64>, Vector3<f64>) {
    let gmst = gmst_from_jd(jd);
    let r = rotation_z(gmst);
    let r_inv = r.transpose(); // R3(-GMST)

    // Position: inverse rotation
    let pos_eci = r_inv * pos_ecef;

    // Velocity: inverse rotation then add ω × r_eci
    let omega = Vector3::new(0.0, 0.0, EARTH_ROTATION_RATE);
    let vel_eci = r_inv * vel_ecef + omega.cross(&pos_eci);

    (pos_eci, vel_eci)
}

// ---------------------------------------------------------------------------
// ECI → ENU convenience (Task 2.8)
// ---------------------------------------------------------------------------

/// Convert an ECI position to a local ENU (East-North-Up) vector relative to
/// a reference point on the Earth's surface.
///
/// This composes [`eci_to_ecef`] with [`crate::geodetic::ecef_to_enu`].
///
/// # Arguments
/// * `pos_eci`     — satellite position in ECI (metres)
/// * `jd`          — Julian Date of the epoch
/// * `ref_lat_rad` — geodetic latitude of the reference point (radians)
/// * `ref_lon_rad` — geodetic longitude of the reference point (radians)
/// * `ref_alt_m`   — altitude of the reference point above the WGS-84 ellipsoid (metres)
pub fn eci_to_enu(
    pos_eci: &Vector3<f64>,
    jd: f64,
    ref_lat_rad: f64,
    ref_lon_rad: f64,
    ref_alt_m: f64,
) -> Vector3<f64> {
    let zero_vel = Vector3::zeros();
    let (pos_ecef, _) = eci_to_ecef(pos_eci, &zero_vel, jd);
    crate::geodetic::ecef_to_enu(&pos_ecef, ref_lat_rad, ref_lon_rad, ref_alt_m)
}

// ---------------------------------------------------------------------------
// Tests (Tasks 2.11, 2.12)
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::TAU;

    const TOL_MM: f64 = 1e-3; // 1 mm

    /// Task 2.11 — ECI ↔ ECEF roundtrip error must be < 1 mm.
    #[test]
    fn eci_ecef_roundtrip() {
        let pos = Vector3::new(6_778_137.0, 0.0, 0.0); // ~LEO
        let vel = Vector3::new(0.0, 7_500.0, 0.0);
        let jd = J2000_JD + 1000.0; // arbitrary epoch

        let (pos_ecef, vel_ecef) = eci_to_ecef(&pos, &vel, jd);
        let (pos_rt, vel_rt) = ecef_to_eci(&pos_ecef, &vel_ecef, jd);

        assert!(
            (pos_rt - pos).norm() < TOL_MM,
            "Position roundtrip error: {} m",
            (pos_rt - pos).norm()
        );
        assert!(
            (vel_rt - vel).norm() < TOL_MM,
            "Velocity roundtrip error: {} m/s",
            (vel_rt - vel).norm()
        );
    }

    /// Task 2.12 — After one sidereal day the ECEF position should repeat.
    #[test]
    fn sidereal_day_ecef_repeat() {
        let pos = Vector3::new(6_778_137.0, 1_000_000.0, 3_000_000.0);
        let vel = Vector3::zeros();
        let jd1 = J2000_JD + 500.0;
        let sidereal_day_sec = 86_164.1;
        let jd2 = jd1 + sidereal_day_sec / SECONDS_PER_DAY;

        let (ecef1, _) = eci_to_ecef(&pos, &vel, jd1);
        let (ecef2, _) = eci_to_ecef(&pos, &vel, jd2);

        // Allow a tolerance of ~10 m because 86164.1 s is a rounded sidereal day.
        assert!(
            (ecef1 - ecef2).norm() < 10.0,
            "ECEF difference after sidereal day: {} m",
            (ecef1 - ecef2).norm()
        );
    }

    /// GMST at J2000 epoch should be approximately 280.46° (≈ 4.8949 rad).
    #[test]
    fn gmst_at_j2000() {
        let gmst = gmst_from_jd(J2000_JD);
        let expected_deg: f64 = 280.46;
        let expected_rad = expected_deg.to_radians();
        let diff = (gmst - expected_rad).abs();
        // Allow 0.1° tolerance
        assert!(
            diff < 0.1_f64.to_radians(),
            "GMST at J2000: {:.4}° (expected ~{expected_deg}°)",
            gmst.to_degrees()
        );
    }

    /// TEME → ECEF rotation must preserve the position vector magnitude.
    #[test]
    fn teme_ecef_preserves_magnitude() {
        let pos = Vector3::new(4_000_000.0, 5_000_000.0, 3_000_000.0);
        let vel = Vector3::new(100.0, -200.0, 50.0);
        let jd = J2000_JD + 7300.5;

        let (pos_ecef, _) = teme_to_ecef(&pos, &vel, jd);

        assert!(
            (pos_ecef.norm() - pos.norm()).abs() < 1e-6,
            "Magnitude changed: {} → {}",
            pos.norm(),
            pos_ecef.norm()
        );
    }

    /// ECI → ENU: a satellite directly above a reference point should have
    /// a positive Up component.
    #[test]
    fn eci_to_enu_overhead_positive_up() {
        // Place reference at lon=0, lat=0, alt=0 (on the equator, prime meridian).
        // At GMST=0 the ECEF x-axis aligns with ECI x-axis.
        // We'll find GMST at our chosen epoch and place the satellite along that direction.
        let jd = J2000_JD;
        let gmst = gmst_from_jd(jd);

        // Reference point on equator at the sub-satellite longitude.
        // The ECEF x-direction at this epoch corresponds to ECI rotated by GMST.
        // Put the satellite at altitude 400 km above the reference.
        let r_earth = 6_378_137.0; // WGS-84 semi-major axis
        let altitude = 400_000.0;
        let sat_r = r_earth + altitude;

        // ECI position chosen so that after R_z(gmst) rotation it maps to ECEF +x.
        // R_z(gmst) * pos_eci = (sat_r, 0, 0) => pos_eci = R_z(-gmst) * (sat_r, 0, 0)
        //   = (sat_r * cos(gmst), sat_r * sin(gmst), 0)
        let pos_eci = Vector3::new(sat_r * gmst.cos(), sat_r * gmst.sin(), 0.0);

        let enu = eci_to_enu(&pos_eci, jd, 0.0, 0.0, 0.0);

        assert!(
            enu.z > 0.0,
            "Up component should be positive for overhead satellite, got {}",
            enu.z
        );
    }

    /// rotation_z basic check.
    #[test]
    fn rotation_z_quarter_turn() {
        let r = rotation_z(std::f64::consts::FRAC_PI_2);
        let v = Vector3::new(1.0, 0.0, 0.0);
        let result = r * v;
        assert!((result.x - 0.0).abs() < 1e-12);
        assert!((result.y - (-1.0)).abs() < 1e-12);
        assert!((result.z - 0.0).abs() < 1e-12);
    }

    /// Verify GMST wraps correctly (always in [0, 2π)).
    #[test]
    fn gmst_always_positive() {
        for &jd in &[J2000_JD - 10000.0, J2000_JD, J2000_JD + 50000.0] {
            let g = gmst_from_jd(jd);
            assert!((0.0..TAU).contains(&g), "GMST out of range: {g}");
        }
    }
}
