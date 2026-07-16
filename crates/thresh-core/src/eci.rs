//! ECI (Earth-Centered Inertial) and TEME coordinate transformations.
//!
//! Provides conversions between ECI/TEME, ECEF, and ENU coordinate frames
//! using GMST-based rotation matrices. In the tagged-frame vocabulary of
//! [`crate::frames`], the GMST-only spin used throughout this module is the
//! **TEME → PEF** rotation of the IAU-76/FK5 chain (design Decision 6 of
//! `astro-time-and-frames`): the "ECEF" it produces is PEF, identical to
//! ITRF while polar motion is neglected, so the chain is correct for TEME
//! input — the frame SGP4 emits. States expressed in **GCRF** must instead
//! route through the full reduction: see [`gcrf_to_enu`] for the
//! ground-station projection and [`crate::frames`] for general transforms.

use nalgebra::{Matrix3, Vector3};
use serde::{Deserialize, Serialize};

use crate::frames::{Frame, FrameError, FrameProvider, Iau76Fk5Provider};
use crate::time::Epoch;

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
    /// Time-scale-aware absolute epoch (was `epoch_jd: f64`; construct
    /// from a UTC Julian date via [`Epoch::from_jde_utc`]).
    pub epoch: Epoch,
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

/// Epoch-taking form of [`gmst_from_jd`] (design Decision 5 of
/// `astro-time-and-frames`): Greenwich Mean Sidereal Time (IAU 1982) in
/// radians, `[0, 2π)`.
///
/// GMST is a function of **UT1**; this form approximates UT1 by the epoch's
/// **UTC** reading (UT1 ≈ UTC: the |ΔUT1| ≤ 0.9 s bound kept by IERS
/// leap-second scheduling corresponds to ≤ ~430 m of ECEF longitude at the
/// equator). For an explicit ΔUT1, use
/// [`crate::frames::legs::gmst_iau1982`].
///
/// Delegates to [`gmst_from_jd`] at `epoch.to_jde_utc_days()`, so for the
/// same instant expressed as the same `f64` Julian date the two forms
/// produce **identical** numbers — the epoch type changed in
/// `astro-time-and-frames`, the rotation math did not (design Decision 6).
pub fn gmst(epoch: &Epoch) -> f64 {
    gmst_from_jd(epoch.to_jde_utc_days())
}

// ---------------------------------------------------------------------------
// TEME → ECEF (Task 2.4)
// ---------------------------------------------------------------------------

/// Convert a state vector from the TEME (True Equator, Mean Equinox) frame to
/// the ECEF (Earth-Centered, Earth-Fixed) frame at the given Julian Date.
///
/// The rotation uses GMST only (no polar-motion or equation-of-equinoxes
/// corrections) — in the [`crate::frames`] vocabulary this is exactly the
/// **TEME → PEF** leg of the IAU-76/FK5 chain, so the returned "ECEF" is
/// PEF (identical to ITRF while polar motion is neglected) and the rotation
/// is the correct Earth-fixing spin for TEME input such as SGP4 output.
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
/// **Note:** This applies the same GMST-only rotation as [`teme_to_ecef`],
/// i.e. the **TEME → PEF** leg of the IAU-76/FK5 chain — it is only correct
/// when the input is TEME-consistent (the frame SGP4 emits, and the frame
/// the benchmark's GMST-only chain is internally consistent in — design
/// Decision 6 of `astro-time-and-frames`). Feeding a true GCRF/J2000 state
/// through it mis-rotates by the accumulated precession/nutation:
/// kilometre-scale at LEO radius for modern epochs. GCRF states should use
/// the full reduction instead — [`gcrf_to_enu`] for the ground-station
/// projection, or [`crate::frames`] transforms in general.
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
// ENU → ECI inverse (orbital-ballistic-filter-models, task 6.1)
// ---------------------------------------------------------------------------

/// Convert a local ENU (East-North-Up) position back to ECI, inverting
/// [`eci_to_enu`] with the same station geodetics and GMST epoch.
///
/// This composes [`crate::geodetic::enu_to_ecef`] with [`ecef_to_eci`]
/// (position only, like `eci_to_enu`), so `eci_to_enu ∘ enu_to_eci = id`
/// exactly — both directions use the identical GMST rotation and ENU
/// rotation matrices. The benchmark runners use this to lift station-frame
/// radar detections into the ECI frame the orbital / ballistic motion
/// models integrate in (design Decision 7 of `orbital-ballistic-filter-models`).
///
/// # Arguments
/// * `enu`         — position in the station's ENU frame (metres)
/// * `jd`          — Julian Date of the measurement epoch
/// * `ref_lat_rad` — geodetic latitude of the station (radians)
/// * `ref_lon_rad` — geodetic longitude of the station (radians)
/// * `ref_alt_m`   — station altitude above the WGS-84 ellipsoid (metres)
pub fn enu_to_eci(
    enu: &Vector3<f64>,
    jd: f64,
    ref_lat_rad: f64,
    ref_lon_rad: f64,
    ref_alt_m: f64,
) -> Vector3<f64> {
    let pos_ecef = crate::geodetic::enu_to_ecef(enu, ref_lat_rad, ref_lon_rad, ref_alt_m);
    let zero_vel = Vector3::zeros();
    let (pos_eci, _) = ecef_to_eci(&pos_ecef, &zero_vel, jd);
    pos_eci
}

// ---------------------------------------------------------------------------
// GCRF → ENU via the full reduction (astro-time-and-frames, task 5.1)
// ---------------------------------------------------------------------------

/// Convert a GCRF position to a local ENU (East-North-Up) vector relative to
/// a reference point on the Earth's surface, routing through the **full
/// IAU-76/FK5 reduction** (GCRF → ITRF via [`crate::frames`], default
/// zero-EOP [`Iau76Fk5Provider`]) followed by
/// [`crate::geodetic::ecef_to_enu`].
///
/// This is the GCRF-input counterpart of [`eci_to_enu`], whose GMST-only
/// rotation is the TEME → PEF leg and therefore expects TEME-consistent
/// input: the same physical state expressed in TEME and projected with
/// [`eci_to_enu`], or expressed in GCRF and projected with this function,
/// lands on the same ENU vector to within the transform-chain tolerance
/// (spec scenario "TEME and GCRF states project consistently").
///
/// # Arguments
/// * `pos_gcrf`    — position in GCRF (metres)
/// * `epoch`       — epoch of the state (its TT drives precession/nutation;
///   its UTC ≈ UT1 drives the sidereal rotation under the zero-EOP default)
/// * `ref_lat_rad` — geodetic latitude of the reference point (radians)
/// * `ref_lon_rad` — geodetic longitude of the reference point (radians)
/// * `ref_alt_m`   — altitude of the reference point above the WGS-84 ellipsoid (metres)
pub fn gcrf_to_enu(
    pos_gcrf: &Vector3<f64>,
    epoch: &Epoch,
    ref_lat_rad: f64,
    ref_lon_rad: f64,
    ref_alt_m: f64,
) -> Vector3<f64> {
    gcrf_to_enu_with(
        &Iau76Fk5Provider::default(),
        pos_gcrf,
        epoch,
        ref_lat_rad,
        ref_lon_rad,
        ref_alt_m,
    )
    .expect("IAU-76/FK5 provider supports every Frame pair")
}

/// Provider-seam form of [`gcrf_to_enu`]: the GCRF → ITRF rotation comes
/// from the supplied [`FrameProvider`], so an EOP-carrying
/// [`Iau76Fk5Provider`] or a custom provider substitutes without call-site
/// changes (design Decision 4 of `astro-time-and-frames`).
///
/// # Errors
///
/// Propagates [`FrameError`] from the provider (the default
/// [`Iau76Fk5Provider`] never errors).
pub fn gcrf_to_enu_with<P: FrameProvider + ?Sized>(
    provider: &P,
    pos_gcrf: &Vector3<f64>,
    epoch: &Epoch,
    ref_lat_rad: f64,
    ref_lon_rad: f64,
    ref_alt_m: f64,
) -> Result<Vector3<f64>, FrameError> {
    let rot = provider.rotation(Frame::Gcrf, Frame::Itrf, epoch)?;
    let pos_itrf = rot.rotate_position(pos_gcrf);
    Ok(crate::geodetic::ecef_to_enu(
        &pos_itrf,
        ref_lat_rad,
        ref_lon_rad,
        ref_alt_m,
    ))
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

    /// The epoch-taking `gmst` is the identical IAU-1982 polynomial: it
    /// matches `gmst_from_jd` **bitwise** at the epoch's own UTC JD
    /// (design Decision 6 of `astro-time-and-frames`: the epoch type
    /// changed, the rotation math did not), and agrees with the frames
    /// module's ΔUT1-taking form at ΔUT1 = 0.
    #[test]
    fn epoch_taking_gmst_is_bitwise_identical_to_jd_form() {
        for &jd in &[2_460_310.5, 2_451_545.0, 2_453_101.5] {
            let epoch = Epoch::from_jde_utc(jd);
            let via_epoch = gmst(&epoch);
            assert_eq!(
                via_epoch.to_bits(),
                gmst_from_jd(epoch.to_jde_utc_days()).to_bits()
            );
            assert_eq!(
                via_epoch.to_bits(),
                crate::frames::legs::gmst_iau1982(&epoch, 0.0).to_bits()
            );
        }
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

    /// Task 6.1 of `orbital-ballistic-filter-models` — round trip
    /// `eci_to_enu ∘ enu_to_eci = id` at the benchmark epochs: the cached
    /// ISS / Starlink-train TLE epoch (2024-01-01 00:00 UTC, JD 2460310.5,
    /// `orbital-iss.toml` / `orbital-starlink-train.toml`) and the
    /// ballistic-mrbm launch epoch (J2000, JD 2451545.0), each sampled
    /// across a 3 h scenario window. Both directions share the same GMST
    /// and ENU rotations, so the round trip is exact to float noise.
    #[test]
    fn enu_eci_roundtrip_at_benchmark_epochs() {
        let station_lat = 38.8339_f64.to_radians(); // benchmark ground station
        let station_lon = (-104.8214_f64).to_radians();
        let station_alt = 1885.0;

        let benchmark_epochs = [2_460_310.5, 2_451_545.0];
        // LEO-pass-scale ENU offsets (hundreds of km, above and below horizon).
        let enu_samples = [
            Vector3::new(100_000.0, 200_000.0, 400_000.0),
            Vector3::new(-800_000.0, 350_000.0, 90_000.0),
            Vector3::new(50_000.0, -1_200_000.0, -30_000.0),
        ];

        for &epoch_jd in &benchmark_epochs {
            // Offsets spanning the 3 h scenario window at the 5 s cadence scale.
            for &offset_s in &[0.0, 5.0, 3_600.0, 10_800.0] {
                let jd = epoch_jd + offset_s / SECONDS_PER_DAY;
                for enu in &enu_samples {
                    let eci = enu_to_eci(enu, jd, station_lat, station_lon, station_alt);
                    let back = eci_to_enu(&eci, jd, station_lat, station_lon, station_alt);
                    assert!(
                        (back - enu).norm() < TOL_MM,
                        "ENU round-trip error {} m at JD {jd}",
                        (back - enu).norm()
                    );
                }
            }
        }
    }

    /// The inverse must agree with the forward transform on a physically
    /// meaningful satellite state: converting a LEO ECI position to ENU and
    /// back recovers the original ECI vector.
    #[test]
    fn eci_enu_eci_roundtrip_leo_position() {
        let station_lat = 38.8339_f64.to_radians();
        let station_lon = (-104.8214_f64).to_radians();
        let pos_eci = Vector3::new(4_000_000.0, -3_500_000.0, 4_200_000.0);
        let jd = 2_460_310.5 + 1234.0 / SECONDS_PER_DAY;

        let enu = eci_to_enu(&pos_eci, jd, station_lat, station_lon, 1885.0);
        let back = enu_to_eci(&enu, jd, station_lat, station_lon, 1885.0);
        assert!(
            (back - pos_eci).norm() < TOL_MM,
            "ECI round-trip error {} m",
            (back - pos_eci).norm()
        );
    }

    /// Spec "TEME and GCRF states project consistently" (task 5.1 of
    /// `astro-time-and-frames`): the same physical LEO state expressed once
    /// in TEME (projected via the GMST-based path — the TEME → PEF spin)
    /// and once in GCRF (projected via the full reduction with
    /// [`gcrf_to_enu`]) lands on the same ENU vector.
    ///
    /// Error budget for the 0.5 m tolerance: both routes evaluate the
    /// identical IAU-1982 GMST polynomial (asserted bitwise in
    /// `epoch_taking_gmst_is_bitwise_identical_to_jd_form`), so the residual
    /// is only the f64-JD quantization of the `Epoch` accessors (≤ 1 JD ulp
    /// ≈ 40 µs ≈ 2 cm of Earth rotation at LEO radius) plus
    /// rotation-composition round-off.
    #[test]
    fn teme_and_gcrf_states_project_to_agreeing_enu() {
        let station_lat = 38.8339_f64.to_radians(); // benchmark ground station
        let station_lon = (-104.8214_f64).to_radians();
        let station_alt = 1885.0;

        // Benchmark TLE epoch (2024-01-01 00:00 UTC) + an odd offset.
        let jd = 2_460_310.5 + 1234.0 / SECONDS_PER_DAY;
        let epoch = Epoch::from_jde_utc(jd);

        let pos_teme = Vector3::new(4_000_000.0, -3_500_000.0, 4_200_000.0);
        let vel_teme = Vector3::new(1_500.0, 7_100.0, -2_300.0);

        // Route 1: TEME through the GMST-based path (TEME → PEF → ENU).
        let enu_from_teme = eci_to_enu(&pos_teme, jd, station_lat, station_lon, station_alt);

        // Route 2: the same physical state expressed in GCRF, through the
        // full reduction (GCRF → ITRF → ENU).
        let (pos_gcrf, _) = crate::frames::teme_to_gcrf(&pos_teme, &vel_teme, &epoch);
        let enu_from_gcrf = gcrf_to_enu(&pos_gcrf, &epoch, station_lat, station_lon, station_alt);

        assert!(
            (enu_from_gcrf - enu_from_teme).norm() < 0.5,
            "TEME- and GCRF-expressed projections disagree by {} m",
            (enu_from_gcrf - enu_from_teme).norm()
        );

        // Feeding the GCRF state through the GMST-only path instead would
        // mis-rotate by the accumulated precession/nutation — the km-scale
        // error the GCRF route exists to avoid.
        let enu_mislabeled = eci_to_enu(&pos_gcrf, jd, station_lat, station_lon, station_alt);
        assert!(
            (enu_mislabeled - enu_from_teme).norm() > 1_000.0,
            "expected km-scale error from the mislabeled route, got {} m",
            (enu_mislabeled - enu_from_teme).norm()
        );
    }

    /// A provider whose rotation is always the identity — a stand-in for
    /// any externally sourced reduction.
    struct IdentityProvider;

    impl FrameProvider for IdentityProvider {
        fn rotation(
            &self,
            _from: Frame,
            _to: Frame,
            _epoch: &Epoch,
        ) -> Result<crate::frames::FrameRotation, FrameError> {
            Ok(crate::frames::FrameRotation {
                r: Matrix3::identity(),
                r_dot: Matrix3::zeros(),
            })
        }
    }

    /// `gcrf_to_enu` is exactly the default-provider case of
    /// [`gcrf_to_enu_with`], and a custom provider substitutes at the same
    /// call site (design Decision 4 of `astro-time-and-frames`).
    #[test]
    fn gcrf_to_enu_with_substitutes_provider() {
        let station_lat = 38.8339_f64.to_radians();
        let station_lon = (-104.8214_f64).to_radians();
        let epoch = Epoch::from_jde_utc(2_460_310.5);
        let pos_gcrf = Vector3::new(4_000_000.0, -3_500_000.0, 4_200_000.0);

        let via_default = gcrf_to_enu(&pos_gcrf, &epoch, station_lat, station_lon, 1885.0);
        let via_explicit = gcrf_to_enu_with(
            &Iau76Fk5Provider::default(),
            &pos_gcrf,
            &epoch,
            station_lat,
            station_lon,
            1885.0,
        )
        .unwrap();
        assert_eq!(via_default, via_explicit);

        // The identity provider skips the reduction entirely: the result is
        // the plain geodetic ENU projection of the unrotated vector, and it
        // must differ from the real reduction.
        let via_identity = gcrf_to_enu_with(
            &IdentityProvider,
            &pos_gcrf,
            &epoch,
            station_lat,
            station_lon,
            1885.0,
        )
        .unwrap();
        let expected = crate::geodetic::ecef_to_enu(&pos_gcrf, station_lat, station_lon, 1885.0);
        assert_eq!(via_identity, expected);
        assert!(
            (via_identity - via_default).norm() > 1_000.0,
            "substituted provider must be used"
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
