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
