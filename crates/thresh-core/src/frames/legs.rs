//! Rotation legs of the IAU-76/FK5 reduction, one phase helper per leg.
//!
//! Each leg is a small, individually tested function; the provider in
//! [`super`] composes them into full frame-to-frame rotations. Algorithms
//! follow the ERFA reference implementations (liberfa/erfa `master`, fetched
//! 2026-07-15; ERFA is derived, with permission, from IAU SOFA):
//! `prec76.c` (Lieske 1979), `nut80.c` (IAU 1980 nutation), `obl80.c`
//! (IAU 1980 mean obliquity), `gmst82.c` (IAU 1982 GMST), `eqeq94.c`
//! (IAU 1994 equation of the equinoxes), and `pom00.c` (polar motion,
//! with the FK5 convention s′ = 0).
//!
//! # Conventions
//!
//! * All rotation matrices are **frame rotations** acting on column vectors:
//!   `v_to = R * v_from`. [`rotation_x`]/[`rotation_y`] here and
//!   [`crate::eci::rotation_z`] follow the same convention (matching ERFA's
//!   `rx`/`ry`/`rz`).
//! * All angles are radians; `t_tt` arguments are Julian centuries of **TT**
//!   since J2000.0 (see [`julian_centuries_tt`]).
//! * Sidereal legs take UT1 via UTC + ΔUT1; with the zero default,
//!   UT1 ≈ UTC to within the |ΔUT1| ≤ 0.9 s that IERS bulletins guarantee.

use nalgebra::Matrix3;
use std::f64::consts::TAU;

use crate::eci::{J2000_JD, SECONDS_PER_DAY};
use crate::time::Epoch;

use super::nutation1980::IAU1980_NUTATION_TERMS;

/// Arcseconds to radians.
const AS2R: f64 = std::f64::consts::PI / (180.0 * 3600.0);

/// Days per Julian century.
const DAYS_PER_CENTURY: f64 = 36_525.0;

/// Julian centuries of **TT** elapsed since J2000.0 (JD 2451545.0 TT).
///
/// This is the time argument (`t_tt`) of every precession/nutation leg in
/// this module.
pub fn julian_centuries_tt(epoch: &Epoch) -> f64 {
    (epoch.to_jde_tt_days() - J2000_JD) / DAYS_PER_CENTURY
}

/// Frame rotation about the X axis by `angle` radians (ERFA `rx`
/// convention): `v_rotated_frame = rotation_x(angle) * v`.
pub fn rotation_x(angle: f64) -> Matrix3<f64> {
    let (s, c) = angle.sin_cos();
    Matrix3::new(1.0, 0.0, 0.0, 0.0, c, s, 0.0, -s, c)
}

/// Frame rotation about the Y axis by `angle` radians (ERFA `ry`
/// convention): `v_rotated_frame = rotation_y(angle) * v`.
pub fn rotation_y(angle: f64) -> Matrix3<f64> {
    let (s, c) = angle.sin_cos();
    Matrix3::new(c, 0.0, -s, 0.0, 1.0, 0.0, s, 0.0, c)
}

/// IAU-76 precession angles ζ, θ, z (radians) from J2000.0 to the epoch
/// `t_tt` Julian centuries (TT) later.
///
/// Polynomials from Lieske (1979) as implemented in ERFA `prec76.c` with the
/// start epoch fixed at J2000.0 (t₀ = 0).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PrecessionAngles {
    /// ζ (zeta), radians.
    pub zeta: f64,
    /// θ (theta), radians.
    pub theta: f64,
    /// z, radians.
    pub z: f64,
}

/// Evaluates the IAU-76 precession angles at `t_tt` Julian centuries of TT
/// since J2000.0. See [`PrecessionAngles`] for the source.
pub fn precession_angles_iau76(t_tt: f64) -> PrecessionAngles {
    let t = t_tt;
    let tas2r = t * AS2R;
    let w = 2306.2181;
    PrecessionAngles {
        zeta: (w + (0.30188 + 0.017998 * t) * t) * tas2r,
        theta: (2004.3109 + (-0.42665 - 0.041833 * t) * t) * tas2r,
        z: (w + (1.09468 + 0.018203 * t) * t) * tas2r,
    }
}

/// IAU-76 precession matrix mapping **GCRF/J2000 mean-equator coordinates to
/// mean-of-date (MOD)**: `v_mod = P * v_gcrf`.
///
/// `P = R3(−z) · R2(θ) · R3(−ζ)` (ERFA `pmat76.c` composition) with the
/// angles from [`precession_angles_iau76`].
pub fn precession_matrix_iau76(t_tt: f64) -> Matrix3<f64> {
    let a = precession_angles_iau76(t_tt);
    crate::eci::rotation_z(-a.z) * rotation_y(a.theta) * crate::eci::rotation_z(-a.zeta)
}

/// IAU-1980 nutation components Δψ (longitude) and Δε (obliquity), radians.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NutationAngles {
    /// Nutation in longitude Δψ, radians.
    pub dpsi: f64,
    /// Nutation in obliquity Δε, radians.
    pub deps: f64,
}

/// Delaunay fundamental arguments (l, l′, F, D, Ω) of the IAU-1980 nutation
/// theory, radians, at `t_tt` Julian centuries of TT since J2000.0.
///
/// Polynomials from ERFA `nut80.c`. The arguments are left unnormalized
/// (they feed periodic `sin`/`cos` only).
fn delaunay_arguments_iau1980(t: f64) -> (f64, f64, f64, f64, f64) {
    // Mean longitude of Moon minus mean longitude of Moon's perigee.
    let el = (485866.733 + (715922.633 + (31.310 + 0.064 * t) * t) * t) * AS2R
        + ((1325.0 * t) % 1.0) * TAU;
    // Mean longitude of Sun minus mean longitude of Sun's perigee.
    let elp = (1287099.804 + (1292581.224 + (-0.577 - 0.012 * t) * t) * t) * AS2R
        + ((99.0 * t) % 1.0) * TAU;
    // Mean longitude of Moon minus mean longitude of Moon's node.
    let f = (335778.877 + (295263.137 + (-13.257 + 0.011 * t) * t) * t) * AS2R
        + ((1342.0 * t) % 1.0) * TAU;
    // Mean elongation of Moon from Sun.
    let d = (1072261.307 + (1105601.328 + (-6.891 + 0.019 * t) * t) * t) * AS2R
        + ((1236.0 * t) % 1.0) * TAU;
    let om = moon_ascending_node_iau1980(t);
    (el, elp, f, d, om)
}

/// Longitude of the mean ascending node of the lunar orbit (Ω), radians,
/// measured from the mean equinox of date (ERFA `nut80.c` / `eqeq94.c`
/// polynomial). Shared by the nutation series and the equation of the
/// equinoxes.
fn moon_ascending_node_iau1980(t: f64) -> f64 {
    (450160.280 + (-482890.539 + (7.455 + 0.008 * t) * t) * t) * AS2R + ((-5.0 * t) % 1.0) * TAU
}

/// Evaluates the full 106-term IAU-1980 nutation series (the const
/// `IAU1980_NUTATION_TERMS` table, provenance in `frames/nutation1980.rs`)
/// at `t_tt` Julian centuries of TT since J2000.0, returning Δψ and Δε in
/// radians.
///
/// Table-driven: forms each term's argument from the Delaunay arguments and
/// accumulates the sine/cosine contributions, smallest terms first (the ERFA
/// `nut80.c` summation order).
pub fn nutation_iau1980(t_tt: f64) -> NutationAngles {
    let (el, elp, f, d, om) = delaunay_arguments_iau1980(t_tt);
    // Units of 0.1 mas to radians.
    const U2R: f64 = AS2R / 1.0e4;

    let mut dp = 0.0;
    let mut de = 0.0;
    for (n, c) in IAU1980_NUTATION_TERMS.iter().rev() {
        let arg = f64::from(n[0]) * el
            + f64::from(n[1]) * elp
            + f64::from(n[2]) * f
            + f64::from(n[3]) * d
            + f64::from(n[4]) * om;
        let s = c[0] + c[1] * t_tt;
        let e = c[2] + c[3] * t_tt;
        if s != 0.0 {
            dp += s * arg.sin();
        }
        if e != 0.0 {
            de += e * arg.cos();
        }
    }
    NutationAngles {
        dpsi: dp * U2R,
        deps: de * U2R,
    }
}

/// IAU-1980 mean obliquity of the ecliptic ε̄ (radians) at `t_tt` Julian
/// centuries of TT since J2000.0 (ERFA `obl80.c` polynomial).
pub fn mean_obliquity_iau1980(t_tt: f64) -> f64 {
    let t = t_tt;
    AS2R * (84381.448 + (-46.8150 + (-0.00059 + 0.001813 * t) * t) * t)
}

/// Nutation matrix mapping **mean-of-date (MOD) to true-of-date (TOD)**:
/// `v_tod = N * v_mod`, assembled from a mean obliquity ε̄ and nutation
/// components Δψ, Δε (all radians).
///
/// `N = R1(−(ε̄ + Δε)) · R3(−Δψ) · R1(ε̄)` (ERFA `numat.c` composition).
/// Use [`nutation_matrix_iau1980`] when starting from an epoch argument.
pub fn nutation_matrix(mean_obliquity: f64, dpsi: f64, deps: f64) -> Matrix3<f64> {
    rotation_x(-(mean_obliquity + deps))
        * crate::eci::rotation_z(-dpsi)
        * rotation_x(mean_obliquity)
}

/// Nutation matrix (MOD → TOD, see [`nutation_matrix`]) evaluated from
/// `t_tt` Julian centuries of TT since J2000.0 using the IAU-1980 series and
/// mean obliquity.
pub fn nutation_matrix_iau1980(t_tt: f64) -> Matrix3<f64> {
    let angles = nutation_iau1980(t_tt);
    nutation_matrix(mean_obliquity_iau1980(t_tt), angles.dpsi, angles.deps)
}

/// Greenwich Mean Sidereal Time (IAU 1982 model), radians in `[0, 2π)`.
///
/// GMST is a function of **UT1**; this form takes it as
/// UT1 = UTC + `delta_ut1_s` seconds. With the zero-ΔUT1 default of
/// [`super::Iau76Fk5Provider`], **UT1 ≈ UTC**: the |ΔUT1| ≤ 0.9 s bound
/// kept by IERS leap-second scheduling corresponds to ≤ ~430 m of ECEF
/// longitude at the equator.
///
/// Delegates to [`crate::eci::gmst_from_jd`], which evaluates the same
/// IAU 1982 GMST polynomial (the two agree at any JD by construction —
/// tested below). `f64` JD quantization bounds the result at ~3 nrad
/// (≈ 2 cm at LEO radius); see the precision note on [`Epoch`].
pub fn gmst_iau1982(epoch: &Epoch, delta_ut1_s: f64) -> f64 {
    let jd_ut1 = epoch.to_jde_utc_days() + delta_ut1_s / SECONDS_PER_DAY;
    crate::eci::gmst_from_jd(jd_ut1)
}

/// Equation of the equinoxes (IAU 1994 model), radians:
/// `Δψ·cos ε̄ + 0.00264″·sin Ω + 0.000063″·sin 2Ω` (ERFA `eqeq94.c`).
///
/// Takes the already-evaluated IAU-1980 nutation-in-longitude `dpsi` and
/// mean obliquity ε̄ (radians) so callers evaluating several legs at one
/// epoch do not recompute the 106-term series; `t_tt` (Julian centuries of
/// TT since J2000.0) drives the Ω complementary terms.
///
/// This is exactly GAST − GMST in this module, so the TEME↔TOD and
/// TEME→PEF junctions stay mutually consistent.
pub fn equation_of_equinoxes_iau1994(t_tt: f64, dpsi: f64, mean_obliquity: f64) -> f64 {
    let om = moon_ascending_node_iau1980(t_tt);
    dpsi * mean_obliquity.cos() + AS2R * (0.00264 * om.sin() + 0.000063 * (om + om).sin())
}

/// Greenwich Apparent Sidereal Time, radians in `[0, 2π)`:
/// GAST = GMST(IAU 1982) + equation of the equinoxes (IAU 1994).
///
/// Sidereal time is a function of UT1 = UTC + `delta_ut1_s` (see
/// [`gmst_iau1982`] for the UT1 ≈ UTC floor); the equation of the equinoxes
/// is evaluated at the epoch's TT.
pub fn gast_iau1994(epoch: &Epoch, delta_ut1_s: f64) -> f64 {
    let t_tt = julian_centuries_tt(epoch);
    let nut = nutation_iau1980(t_tt);
    let eqe = equation_of_equinoxes_iau1994(t_tt, nut.dpsi, mean_obliquity_iau1980(t_tt));
    (gmst_iau1982(epoch, delta_ut1_s) + eqe).rem_euclid(TAU)
}

/// Polar-motion matrix mapping **PEF to ITRF**: `v_itrf = W * v_pef`, with
/// pole coordinates `xp`, `yp` in **radians** (EOP bulletins publish them in
/// arcseconds; multiply by π/(180·3600)).
///
/// `W = R1(−yp) · R2(−xp)` — ERFA `pom00.c` with the TIO locator s′ fixed at
/// 0, the FK5 convention (s′ is < 0.1 mas for decades around J2000).
/// Vallado's `ROT1(yp)·ROT2(xp)` ITRF→PEF form is this matrix's transpose
/// to O(xp·yp) ≈ 10⁻¹² rad.
pub fn polar_motion_matrix(xp: f64, yp: f64) -> Matrix3<f64> {
    rotation_x(-yp) * rotation_y(-xp)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::eci::gmst_from_jd;

    // -----------------------------------------------------------------
    // Reference values: generated this session with pyerfa 2.0.1.5
    // (the ERFA library, derived from IAU SOFA) via the scratchpad script
    // `gen_refs.py` — erfa.prec76 / obl80 / nut80 / gmst82 / eqeq94 /
    // pmat76 / nutm80 at 2020-01-01T00:00:00 UTC (JD 2458849.5) and
    // 2024-01-01T00:00:00 UTC (JD 2460310.5), TT = UTC + 69.184 s on both
    // dates (TAI−UTC = 37 s per IERS Bulletin C + TT = TAI + 32.184 s).
    // -----------------------------------------------------------------

    /// TT Julian centuries with the identical two-part arithmetic the
    /// pyerfa run used, so leg inputs are bit-identical to the reference.
    fn t_tt_of(jd_utc: f64) -> f64 {
        ((jd_utc - J2000_JD) + 69.184 / 86400.0) / DAYS_PER_CENTURY
    }

    const JD2020: f64 = 2_458_849.5;
    const JD2024: f64 = 2_460_310.5;

    /// pyerfa `erfa.prec76(2451545.0, 0.0, jd_utc, 69.184/86400)` (session
    /// run, both epochs). Note ERFA returns (zeta, z, theta) in that order.
    #[test]
    fn precession_angles_match_erfa_prec76() {
        let a = precession_angles_iau76(t_tt_of(JD2020));
        assert!((a.zeta - 0.002_236_078_592_612_029).abs() < 1e-12);
        assert!((a.theta - 0.001_943_217_534_050_066).abs() < 1e-12);
        assert!((a.z - 0.002_236_232_323_663_898).abs() < 1e-12);

        let b = precession_angles_iau76(t_tt_of(JD2024));
        assert!((b.zeta - 0.002_683_339_292_310_728_4).abs() < 1e-12);
        assert!((b.theta - 0.002_331_866_888_633_876).abs() < 1e-12);
        assert!((b.z - 0.002_683_560_672_358_028).abs() < 1e-12);
    }

    /// pyerfa `erfa.pmat76` at 2020-01-01 (session run), elementwise.
    #[test]
    fn precession_matrix_matches_erfa_pmat76() {
        let expected = Matrix3::new(
            9.999_881_111_970_46e-1,
            -4.472_291_785_614_8e-3,
            -1.943_211_452_335_270_2e-3,
            4.472_291_785_324_548e-3,
            9.999_899_992_436_443e-1,
            -4.345_479_504_958_906e-6,
            1.943_211_453_003_282_3e-3,
            -4.345_180_773_018_283e-6,
            9.999_981_119_534_018e-1,
        );
        let p = precession_matrix_iau76(t_tt_of(JD2020));
        assert!(
            (p - expected).norm() < 1e-12,
            "pmat76 mismatch: {:e}",
            (p - expected).norm()
        );
    }

    /// pyerfa `erfa.obl80` (session run, both epochs).
    #[test]
    fn mean_obliquity_matches_erfa_obl80() {
        assert!((mean_obliquity_iau1980(t_tt_of(JD2020)) - 0.409_047_414_175_282_1).abs() < 1e-12);
        assert!((mean_obliquity_iau1980(t_tt_of(JD2024)) - 0.409_038_335_555_134_47).abs() < 1e-12);
    }

    /// pyerfa `erfa.nut80` (session run, both epochs) — exercises the whole
    /// 106-term table and the Delaunay-argument polynomials.
    #[test]
    fn nutation_matches_erfa_nut80() {
        let a = nutation_iau1980(t_tt_of(JD2020));
        assert!((a.dpsi - -7.992_803_019_910_079e-5).abs() < 1e-12);
        assert!((a.deps - -8.277_038_551_942_59e-6).abs() < 1e-12);

        let b = nutation_iau1980(t_tt_of(JD2024));
        assert!((b.dpsi - -2.599_383_092_728_149_6e-5).abs() < 1e-12);
        assert!((b.deps - 3.907_668_991_417_104e-5).abs() < 1e-12);
    }

    /// pyerfa `erfa.nutm80` at 2020-01-01 (session run), elementwise.
    #[test]
    fn nutation_matrix_matches_erfa_nutm80() {
        let expected = Matrix3::new(
            9.999_999_968_057_55e-1,
            7.333_397_692_105_627e-5,
            3.179_021_589_807_285_6e-5,
            -7.333_424_004_738_68e-5,
            9.999_999_972_767_997e-1,
            8.275_872_896_117_277e-6,
            -3.178_960_890_882_976e-5,
            -8.278_204_180_988_673e-6,
            9.999_999_994_604_46e-1,
        );
        let n = nutation_matrix_iau1980(t_tt_of(JD2020));
        assert!(
            (n - expected).norm() < 1e-12,
            "nutm80 mismatch: {:e}",
            (n - expected).norm()
        );
    }

    /// pyerfa `erfa.gmst82(jd_utc, 0.0)` (session run, both epochs; ΔUT1 = 0
    /// so UT1 = UTC). Tolerance covers the ~3 nrad f64-JD quantization of
    /// the `Epoch` accessor.
    #[test]
    fn gmst_matches_erfa_gmst82() {
        let e2020 = Epoch::from_jde_utc(JD2020);
        assert!((gmst_iau1982(&e2020, 0.0) - 1.747_455_428_294_941_3).abs() < 5e-8);
        let e2024 = Epoch::from_jde_utc(JD2024);
        assert!((gmst_iau1982(&e2024, 0.0) - 1.747_993_146_284_756).abs() < 5e-8);
    }

    /// The epoch-taking GMST agrees with the existing `eci` GMST at the same
    /// JD by construction (it delegates); a non-zero ΔUT1 shifts the JD it
    /// evaluates at.
    #[test]
    fn gmst_agrees_with_eci_gmst_from_jd() {
        let epoch = Epoch::from_gregorian_utc(2024, 1, 1, 0, 0, 0, 0);
        let jd_utc = epoch.to_jde_utc_days();
        assert_eq!(gmst_iau1982(&epoch, 0.0), gmst_from_jd(jd_utc));
        let dut1 = -0.2;
        assert_eq!(
            gmst_iau1982(&epoch, dut1),
            gmst_from_jd(jd_utc + dut1 / SECONDS_PER_DAY)
        );
    }

    /// pyerfa `erfa.eqeq94` (session run, both epochs), fed with this
    /// module's own Δψ and ε̄ (already verified against ERFA above).
    #[test]
    fn equation_of_equinoxes_matches_erfa_eqeq94() {
        let t = t_tt_of(JD2020);
        let nut = nutation_iau1980(t);
        let eqe = equation_of_equinoxes_iau1994(t, nut.dpsi, mean_obliquity_iau1980(t));
        assert!((eqe - -7.332_139_685_567_044e-5).abs() < 1e-12);

        let t = t_tt_of(JD2024);
        let nut = nutation_iau1980(t);
        let eqe = equation_of_equinoxes_iau1994(t, nut.dpsi, mean_obliquity_iau1980(t));
        assert!((eqe - -2.384_467_194_349_733_5e-5).abs() < 1e-12);
    }

    /// GAST = GMST + Eq_equinox exactly (mod 2π) — the identity the TEME
    /// junction relies on.
    #[test]
    fn gast_is_gmst_plus_equation_of_equinoxes() {
        let epoch = Epoch::from_gregorian_utc(2024, 1, 1, 0, 0, 0, 0);
        let t = julian_centuries_tt(&epoch);
        let nut = nutation_iau1980(t);
        let eqe = equation_of_equinoxes_iau1994(t, nut.dpsi, mean_obliquity_iau1980(t));
        let gast = gast_iau1994(&epoch, 0.0);
        let gmst = gmst_iau1982(&epoch, 0.0);
        assert!((gast - (gmst + eqe).rem_euclid(TAU)).abs() < 1e-15);
    }

    /// Polar-motion matrix structure: orthonormal, det = 1, and the
    /// small-angle off-diagonals carry xp/yp where expected
    /// (W ≈ I + [[0,0,xp],[0,0,−yp],[−xp,yp,0]] to first order).
    #[test]
    fn polar_motion_matrix_structure() {
        let xp = 0.15 * AS2R;
        let yp = 0.30 * AS2R;
        let w = polar_motion_matrix(xp, yp);
        assert!((w * w.transpose() - Matrix3::identity()).norm() < 1e-15);
        assert!((w.determinant() - 1.0).abs() < 1e-15);
        assert!((w[(0, 2)] - xp).abs() < 1e-12);
        assert!((w[(1, 2)] - -yp).abs() < 1e-12);
        assert!((w[(2, 0)] - -xp).abs() < 1e-12);
        assert!((w[(2, 1)] - yp).abs() < 1e-12);
        assert_eq!(polar_motion_matrix(0.0, 0.0), Matrix3::identity());
    }

    /// Frame-rotation convention checks for the axis helpers: rotating the
    /// frame by +90° about X maps +Y onto the new +Z... (ERFA rx/ry/rz
    /// convention, same as `eci::rotation_z`).
    #[test]
    fn rotation_axis_conventions() {
        let quarter = std::f64::consts::FRAC_PI_2;
        let rx = rotation_x(quarter);
        let v = nalgebra::Vector3::new(0.0, 1.0, 0.0);
        let r = rx * v;
        assert!((r.x).abs() < 1e-15 && (r.y).abs() < 1e-15 && (r.z - -1.0).abs() < 1e-15);

        let ry = rotation_y(quarter);
        let v = nalgebra::Vector3::new(1.0, 0.0, 0.0);
        let r = ry * v;
        assert!((r.x).abs() < 1e-15 && (r.y).abs() < 1e-15 && (r.z - 1.0).abs() < 1e-15);
    }

    /// Precession and nutation matrices are proper rotations at a spread of
    /// epochs (structural self-consistency, no external truth needed).
    #[test]
    fn leg_matrices_are_proper_rotations() {
        for &t in &[-0.3, 0.0, 0.2, 0.24, 0.5] {
            for m in [precession_matrix_iau76(t), nutation_matrix_iau1980(t)] {
                assert!((m * m.transpose() - Matrix3::identity()).norm() < 1e-14);
                assert!((m.determinant() - 1.0).abs() < 1e-14);
            }
        }
    }

    /// `julian_centuries_tt` spot: 2020-01-01T00:00:00 UTC is
    /// JD 2458849.5 UTC (Fliegel & Van Flandern day count, verified in
    /// `time.rs` tests) = 7304.5 days + 69.184 s after J2000.0 in TT.
    #[test]
    fn julian_centuries_tt_spot() {
        let epoch = Epoch::from_gregorian_utc(2020, 1, 1, 0, 0, 0, 0);
        let expected = (7304.5 + 69.184 / 86400.0) / DAYS_PER_CENTURY;
        // f64 JD quantization of the accessor is ~40 µs ≈ 1.3e-11 centuries.
        assert!((julian_centuries_tt(&epoch) - expected).abs() < 1e-10);
    }
}
