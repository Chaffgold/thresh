//! Point-mass (two-body) and J2 zonal-harmonic gravitational accelerations.
//!
//! All functions are parameterized on a [`GravityModel`] — no central-body
//! constant is hardcoded inside the math, so the same code serves Earth,
//! lunar, or any other central body.

use nalgebra::Vector3;
use serde::{Deserialize, Serialize};

/// Gravitational parameters of a central body.
///
/// Passed by reference into the acceleration functions of this module so the
/// force math never hardcodes an Earth constant (design Decision 1 of the
/// `orbital-ballistic-filter-models` change).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct GravityModel {
    /// Gravitational parameter μ = GM (m³/s²).
    pub mu: f64,
    /// J2 zonal harmonic coefficient (dimensionless).
    pub j2: f64,
    /// Equatorial radius of the central body (m).
    pub equatorial_radius: f64,
}

impl GravityModel {
    /// Earth per WGS-84: μ = 3.986004418e14 m³/s², J2 = 1.08263e-3,
    /// equatorial radius 6 378 137 m.
    pub const EARTH_WGS84: Self = Self {
        mu: 3.986_004_418e14,
        j2: 1.082_63e-3,
        equatorial_radius: 6_378_137.0,
    };
}

/// Compute the point-mass (two-body) gravitational acceleration at `pos`.
///
/// `a = −μ·r / ‖r‖³`, directed from `pos` toward the central body.
pub fn two_body_acceleration(pos: &Vector3<f64>, gravity: &GravityModel) -> Vector3<f64> {
    let r2 = pos.x * pos.x + pos.y * pos.y + pos.z * pos.z;
    let r = r2.sqrt();
    let mu_r3 = gravity.mu / (r2 * r);
    Vector3::new(-mu_r3 * pos.x, -mu_r3 * pos.y, -mu_r3 * pos.z)
}

/// Compute the **total** gravitational acceleration at `pos`: two-body plus
/// the J2 zonal-harmonic perturbation.
///
/// Note this returns two-body + J2 combined, not the perturbation term alone
/// (it is the body of the former `thresh_synth::orbital::acceleration_j2`).
/// The z component carries the `(5z²/r² − 3)` factor where x and y carry
/// `(5z²/r² − 1)` — see `docs/reference/orbital-ballistic-dynamics-reference.md`
/// for the derivation of the asymmetry.
pub fn j2_acceleration(pos: &Vector3<f64>, gravity: &GravityModel) -> Vector3<f64> {
    let x = pos.x;
    let y = pos.y;
    let z = pos.z;
    let r2 = x * x + y * y + z * z;
    let r = r2.sqrt();
    let r5 = r2 * r2 * r;

    let mu_r3 = gravity.mu / (r2 * r);
    let j2_coeff =
        1.5 * gravity.j2 * gravity.mu * gravity.equatorial_radius * gravity.equatorial_radius / r5;
    let z2_r2 = 5.0 * z * z / r2;

    Vector3::new(
        -mu_r3 * x + j2_coeff * x * (z2_r2 - 1.0),
        -mu_r3 * y + j2_coeff * y * (z2_r2 - 1.0),
        -mu_r3 * z + j2_coeff * z * (z2_r2 - 3.0),
    )
}

/// Earth's J3 zonal harmonic coefficient (dimensionless, tide-free).
///
/// Derived from the fetched EGM96 distribution (see the provenance notes in
/// the `egm96` module): J₃ = −√7 · C̄₃₀ with C̄₃₀ = 9.57254173792e-7 from
/// `egm96_to360.ascii`, giving −2.5326564853322355e-6. The
/// `zonal_constants_tie_to_the_table` test in `egm96` asserts this constant
/// equals the table-derived value bit-for-bit, and
/// `zonal_constants_match_published_values` below cross-checks it against
/// the value Vallado's `PERTACCEL` initializes (−2.54e-6, `ASTPERT.FOR` in
/// <https://github.com/CelesTrak/fundamentals-of-astrodynamics>, fetched
/// 2026-07-16).
pub const EARTH_J3: f64 = -2.5326564853322355e-6;

/// Earth's J4 zonal harmonic coefficient (dimensionless, tide-free).
///
/// Derived from the fetched EGM96 distribution: J₄ = −√9 · C̄₄₀ with
/// C̄₄₀ = 5.39873863789e-7, giving −1.619621591367e-6. Cross-checked in
/// tests against Vallado's `PERTACCEL` initialization (−1.61e-6) and
/// `valladopy` `constants.py` (J4 = −1.6198976e-6, same repo, fetched
/// 2026-07-16).
pub const EARTH_J4: f64 = -1.619621591367e-6;

/// Compute the J3 zonal-harmonic **perturbation** acceleration at `pos`.
///
/// Unlike [`j2_acceleration`] (which returns two-body + J2 combined for
/// historical reasons), this returns the J3 term *alone*, so a force stack
/// composes as `j2_acceleration + j3_acceleration + j4_acceleration`.
///
/// `j3` is passed explicitly (typically [`EARTH_J3`], or the table-derived
/// `egm96::egm96_jn(3)`) rather than stored on [`GravityModel`], keeping
/// that struct's layout — and every existing constructor — untouched.
///
/// Closed form (SI): with k = −(5/2)·J₃·μ·Re³/r⁷,
///
/// ```text
/// aₓ = k · x · (3z − 7z³/r²)
/// a_y = k · y · (3z − 7z³/r²)
/// a_z = k · (6z² − 7z⁴/r² − (3/5)·r²)
/// ```
///
/// Transcribed from Vallado's `PERTACCEL` (`ASTPERT.FOR`, canonical units,
/// <https://github.com/CelesTrak/fundamentals-of-astrodynamics>, fetched
/// 2026-07-16) with the z-component in its non-singular form (the source's
/// z-branch multiplied through by z, removing the division by z and its
/// spurious `|z| → 0` special case). Verified in tests against a numeric
/// gradient of the J3 potential term and, independently, against the EGM96
/// C̄₃₀ harmonic path.
pub fn j3_acceleration(pos: &Vector3<f64>, gravity: &GravityModel, j3: f64) -> Vector3<f64> {
    let (x, y, z) = (pos.x, pos.y, pos.z);
    let r2 = x * x + y * y + z * z;
    let r = r2.sqrt();
    let r7 = r2 * r2 * r2 * r;
    let re = gravity.equatorial_radius;
    let coeff = -2.5 * j3 * gravity.mu * re * re * re / r7;
    let z2 = z * z;
    let xy_factor = 3.0 * z - 7.0 * z * z2 / r2;
    Vector3::new(
        coeff * x * xy_factor,
        coeff * y * xy_factor,
        coeff * (6.0 * z2 - 7.0 * z2 * z2 / r2 - 0.6 * r2),
    )
}

/// Compute the J4 zonal-harmonic **perturbation** acceleration at `pos`
/// (the J4 term alone; see [`j3_acceleration`] for the composition
/// convention and the explicit-`j4` parameter rationale — pass
/// [`EARTH_J4`] or `egm96::egm96_jn(4)`).
///
/// Closed form (SI): with k = +(15/8)·J₄·μ·Re⁴/r⁷,
///
/// ```text
/// aₓ = k · x · (1 − 14z²/r² + 21z⁴/r⁴)
/// a_y = k · y · (1 − 14z²/r² + 21z⁴/r⁴)
/// a_z = k · z · (5 − (70/3)·z²/r² + 21z⁴/r⁴)
/// ```
///
/// The polynomial structure is transcribed from Vallado's `PERTACCEL` /
/// `AccNonSph` (`ASTPERT.FOR` / `astPert.cpp`, citing "vallado 2013, 526",
/// <https://github.com/CelesTrak/fundamentals-of-astrodynamics>, fetched
/// 2026-07-16), but the leading sign is **+15/8, not the fetched code's
/// −15/8**: with the J₄ < 0 sign convention that code itself uses four
/// lines earlier (J4 = −0.00000161), its −15/8 factor is inconsistent with
/// its own J2/J3 blocks. Two independent executable checks pin the sign
/// used here: the numeric gradient of the J4 potential term
/// (−(μ/r)·J₄·(Re/r)⁴·P₄(z/r), test below) and the EGM96 C̄₄₀ harmonic
/// path (`zonal_only_reproduces_closed_form_j3_j4` in `egm96`) — both
/// agree with +15/8 and disagree with −15/8 by an exact factor of −1.
pub fn j4_acceleration(pos: &Vector3<f64>, gravity: &GravityModel, j4: f64) -> Vector3<f64> {
    let (x, y, z) = (pos.x, pos.y, pos.z);
    let r2 = x * x + y * y + z * z;
    let r = r2.sqrt();
    let r7 = r2 * r2 * r2 * r;
    let re2 = gravity.equatorial_radius * gravity.equatorial_radius;
    let coeff = 1.875 * j4 * gravity.mu * re2 * re2 / r7;
    let z2_r2 = z * z / r2;
    let z4_r4 = z2_r2 * z2_r2;
    let xy_factor = 1.0 - 14.0 * z2_r2 + 21.0 * z4_r4;
    Vector3::new(
        coeff * x * xy_factor,
        coeff * y * xy_factor,
        coeff * z * (5.0 - 70.0 * z2_r2 / 3.0 + 21.0 * z4_r4),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    const EARTH: GravityModel = GravityModel::EARTH_WGS84;

    // ── Two-body direction and magnitude ────────────────────────────────

    #[test]
    fn two_body_points_inward_with_mu_over_r2_magnitude() {
        let r = EARTH.equatorial_radius + 400_000.0;
        let pos = Vector3::new(r, 0.0, 0.0);
        let acc = two_body_acceleration(&pos, &EARTH);

        let expected = -EARTH.mu / (r * r);
        assert!((acc.x - expected).abs() / expected.abs() < 1e-14);
        assert_eq!(acc.y, 0.0);
        assert_eq!(acc.z, 0.0);
    }

    // ── Two-body honors the supplied mu (no Earth constant leaks in) ────

    #[test]
    fn two_body_uses_supplied_mu() {
        let moon = GravityModel {
            mu: 4.904_869_5e12, // lunar GM (m³/s²)
            j2: 2.033e-4,
            equatorial_radius: 1_738_100.0,
        };
        let pos = Vector3::new(2_000_000.0, 500_000.0, -300_000.0);

        let a_earth = two_body_acceleration(&pos, &EARTH);
        let a_moon = two_body_acceleration(&pos, &moon);

        // Same geometry, so the accelerations scale exactly with mu.
        let ratio = moon.mu / EARTH.mu;
        for (m, e) in a_moon.iter().zip(a_earth.iter()) {
            assert!((m - ratio * e).abs() <= 1e-14 * e.abs());
        }
    }

    // ── J2 at the equator: radially inward, closed-form magnitude ───────

    #[test]
    fn j2_perturbation_at_equator_matches_closed_form() {
        let r = EARTH.equatorial_radius + 400_000.0;
        let pos = Vector3::new(r, 0.0, 0.0);

        let perturbation = j2_acceleration(&pos, &EARTH) - two_body_acceleration(&pos, &EARTH);

        // At z = 0 the perturbation is −(3/2)·J2·μ·Re²/r⁴ along +x (inward).
        let expected = -1.5 * EARTH.j2 * EARTH.mu * EARTH.equatorial_radius.powi(2) / r.powi(4);
        assert!((perturbation.x - expected).abs() / expected.abs() < 1e-12);
        assert_eq!(perturbation.y, 0.0);
        assert_eq!(perturbation.z, 0.0);
    }

    // ── J2 at the pole: the (5z²/r² − 3) factor gives an outward term ───

    #[test]
    fn j2_perturbation_at_pole_matches_closed_form() {
        let r = EARTH.equatorial_radius + 400_000.0;
        let pos = Vector3::new(0.0, 0.0, r);

        let perturbation = j2_acceleration(&pos, &EARTH) - two_body_acceleration(&pos, &EARTH);

        // On the z axis: 5z²/r² = 5, so the z factor is (5 − 3) = 2 and the
        // perturbation is +3·J2·μ·Re²/r⁴ along +z (outward).
        let expected = 3.0 * EARTH.j2 * EARTH.mu * EARTH.equatorial_radius.powi(2) / r.powi(4);
        assert_eq!(perturbation.x, 0.0);
        assert_eq!(perturbation.y, 0.0);
        assert!((perturbation.z - expected).abs() / expected.abs() < 1e-12);
    }

    // ── J2 perturbation stays small relative to two-body at LEO ─────────

    #[test]
    fn j2_perturbation_is_small_at_leo() {
        let pos = Vector3::new(EARTH.equatorial_radius + 400_000.0, 0.0, 0.0);
        let a_total = j2_acceleration(&pos, &EARTH);
        let a_2body = two_body_acceleration(&pos, &EARTH);

        assert!(a_total.x < 0.0, "acceleration should point inward");
        assert!(
            (a_total - a_2body).norm() / a_2body.norm() < 0.01,
            "J2 perturbation too large at the equator"
        );
    }
}

#[cfg(test)]
mod zonal_tests {
    use super::*;

    const EARTH: GravityModel = GravityModel::EARTH_WGS84;

    /// A LEO-magnitude general position (r ≈ 6778 km, all components
    /// nonzero, well off both the equator and the pole).
    fn general_position() -> Vector3<f64> {
        Vector3::new(5_300_000.0, -3_400_000.0, 2_500_000.0)
    }

    /// Legendre polynomial Pₙ(x) by the classic three-term recurrence
    /// (n+1)·Pₙ₊₁ = (2n+1)·x·Pₙ − n·Pₙ₋₁ from P₀ = 1, P₁ = x — built in
    /// the test rather than transcribed, so the potential reference is
    /// independent of the closed forms under test.
    fn legendre_polynomial(n: usize, x: f64) -> f64 {
        if n == 0 {
            return 1.0;
        }
        let (mut prev, mut curr) = (1.0, x);
        for k in 1..n {
            let kf = k as f64;
            let next = ((2.0 * kf + 1.0) * x * curr - kf * prev) / (kf + 1.0);
            prev = curr;
            curr = next;
        }
        curr
    }

    /// The degree-n zonal perturbation potential
    /// Rₙ = −(μ/r)·Jₙ·(Re/r)ⁿ·Pₙ(z/r); the perturbation acceleration is
    /// its gradient.
    fn zonal_potential(pos: &Vector3<f64>, n: usize, jn: f64, gravity: &GravityModel) -> f64 {
        let r = pos.norm();
        -(gravity.mu / r)
            * jn
            * (gravity.equatorial_radius / r).powi(n as i32)
            * legendre_polynomial(n, pos.z / r)
    }

    fn numeric_gradient(f: impl Fn(&Vector3<f64>) -> f64, pos: &Vector3<f64>) -> Vector3<f64> {
        let h = 100.0;
        let mut gradient = Vector3::zeros();
        for i in 0..3 {
            let mut plus = *pos;
            let mut minus = *pos;
            plus[i] += h;
            minus[i] -= h;
            gradient[i] = (f(&plus) - f(&minus)) / (2.0 * h);
        }
        gradient
    }

    // ── The closed forms are gradients of the zonal potentials ──────────

    /// Calibration of the test helpers themselves: the *existing*
    /// `j2_acceleration` minus two-body must equal the numeric gradient of
    /// the degree-2 zonal potential — if the potential's sign convention
    /// or the recurrence were wrong, this fails against untouched code.
    #[test]
    fn j2_perturbation_matches_numeric_zonal_gradient() {
        let pos = general_position();
        let analytic = j2_acceleration(&pos, &EARTH) - two_body_acceleration(&pos, &EARTH);
        let numeric = numeric_gradient(|p| zonal_potential(p, 2, EARTH.j2, &EARTH), &pos);
        assert!((analytic - numeric).norm() <= 1e-6 * numeric.norm());
    }

    #[test]
    fn j3_matches_numeric_zonal_gradient() {
        let pos = general_position();
        let analytic = j3_acceleration(&pos, &EARTH, EARTH_J3);
        let numeric = numeric_gradient(|p| zonal_potential(p, 3, EARTH_J3, &EARTH), &pos);
        assert!(
            (analytic - numeric).norm() <= 1e-6 * numeric.norm(),
            "J3 closed form vs numeric gradient: {analytic:?} vs {numeric:?}"
        );
    }

    #[test]
    fn j4_matches_numeric_zonal_gradient() {
        let pos = general_position();
        let analytic = j4_acceleration(&pos, &EARTH, EARTH_J4);
        let numeric = numeric_gradient(|p| zonal_potential(p, 4, EARTH_J4, &EARTH), &pos);
        assert!(
            (analytic - numeric).norm() <= 1e-6 * numeric.norm(),
            "J4 closed form vs numeric gradient: {analytic:?} vs {numeric:?}"
        );
    }

    // ── Structural spot checks at the equator and pole ───────────────────

    /// At z = 0 the J3 term reduces to a pure-z acceleration of
    /// (3/2)·J₃·μ·Re³/r⁵ (negative for Earth's J₃ < 0).
    #[test]
    fn j3_at_equator_is_pure_z_with_closed_form_magnitude() {
        let r = EARTH.equatorial_radius + 400_000.0;
        let pos = Vector3::new(r, 0.0, 0.0);
        let acc = j3_acceleration(&pos, &EARTH, EARTH_J3);
        let expected_z = 1.5 * EARTH_J3 * EARTH.mu * EARTH.equatorial_radius.powi(3) / r.powi(5);
        assert_eq!(acc.x, 0.0);
        assert_eq!(acc.y, 0.0);
        assert!((acc.z - expected_z).abs() <= 1e-12 * expected_z.abs());
        assert!(
            acc.z < 0.0,
            "Earth's negative J3 pulls south at the equator"
        );
    }

    /// On the polar axis: a_J3 = 4·J₃·μ·Re³/r⁵ ẑ and
    /// a_J4 = 5·J₄·μ·Re⁴/r⁶ ẑ, both pure-z.
    #[test]
    fn j3_j4_at_pole_match_closed_form_magnitudes() {
        let r = EARTH.equatorial_radius + 400_000.0;
        let pos = Vector3::new(0.0, 0.0, r);
        let re = EARTH.equatorial_radius;

        let acc3 = j3_acceleration(&pos, &EARTH, EARTH_J3);
        let expected3 = 4.0 * EARTH_J3 * EARTH.mu * re.powi(3) / r.powi(5);
        assert_eq!(acc3.x, 0.0);
        assert_eq!(acc3.y, 0.0);
        assert!((acc3.z - expected3).abs() <= 1e-12 * expected3.abs());

        let acc4 = j4_acceleration(&pos, &EARTH, EARTH_J4);
        let expected4 = 5.0 * EARTH_J4 * EARTH.mu * re.powi(4) / r.powi(6);
        assert_eq!(acc4.x, 0.0);
        assert_eq!(acc4.y, 0.0);
        assert!((acc4.z - expected4).abs() <= 1e-12 * expected4.abs());
    }

    // ── Constants against independently published values ────────────────

    /// EGM96-derived constants against Vallado's `PERTACCEL` inits
    /// (J3 = −2.54e-6, J4 = −1.61e-6; rounded) and `valladopy`
    /// `constants.py` (J4 = −1.6198976e-6), both fetched 2026-07-16 from
    /// <https://github.com/CelesTrak/fundamentals-of-astrodynamics>.
    #[test]
    fn zonal_constants_match_published_values() {
        assert!((EARTH_J3 - (-2.54e-6)).abs() / 2.54e-6 < 5e-3);
        assert!((EARTH_J4 - (-1.61e-6)).abs() / 1.61e-6 < 1e-2);
        assert!((EARTH_J4 - (-1.6198976e-6)).abs() / 1.6198976e-6 < 5e-4);
    }

    // ── Magnitude ordering at LEO ────────────────────────────────────────

    #[test]
    fn j3_j4_are_small_relative_to_j2_perturbation_at_leo() {
        let pos = general_position();
        let j2_pert = (j2_acceleration(&pos, &EARTH) - two_body_acceleration(&pos, &EARTH)).norm();
        let j3_pert = j3_acceleration(&pos, &EARTH, EARTH_J3).norm();
        let j4_pert = j4_acceleration(&pos, &EARTH, EARTH_J4).norm();
        assert!(j3_pert < 0.01 * j2_pert, "J3 {j3_pert} vs J2 {j2_pert}");
        assert!(j4_pert < 0.01 * j2_pert, "J4 {j4_pert} vs J2 {j2_pert}");
        assert!(j3_pert > 0.0 && j4_pert > 0.0);
    }
}
