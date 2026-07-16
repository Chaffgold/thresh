//! Cannonball solar-radiation-pressure acceleration with a cylindrical
//! Earth-shadow eclipse factor.
//!
//! Added by the `orbit-propagation-fidelity` change (design Decision 2's
//! SRP leg). The model is the classic cannonball: acceleration along the
//! Sun→spacecraft line, proportional to the reflectivity-area-to-mass
//! coefficient `Cr·A/m`, scaled by the inverse-square actual Sun→spacecraft
//! distance, and multiplied by a shadow factor that is **exactly 0 in
//! umbra and exactly 1 in sunlight**.
//!
//! # Shadow model (documented approximation)
//!
//! The Earth shadow is a **cylinder** of radius
//! [`SHADOW_CYLINDER_RADIUS_M`] extending anti-sunward: the spacecraft is
//! shadowed iff it is on the anti-Sun side of the geocentre
//! (`r · ŝ < 0`) *and* within the cylinder radius of the Earth–Sun axis.
//! There is no penumbra — the factor is a step function (a conical
//! penumbra model is the recorded upgrade path; the adaptive integrator's
//! step rejection handles the discontinuity, see
//! [`super::dormand_prince`]). The rule matches the golden-fixture
//! convention bit for bit (`test-data/golden/propagation/PROVENANCE.md`).
//!
//! # Solar-pressure convention (design.md open question, resolved)
//!
//! `P(d) = (S₀/c)·(AU/d)²` with S₀ = 1361 W/m² (IAU 2015 Resolution B3
//! nominal total solar irradiance), c = 299 792 458 m/s (exact),
//! AU = [`ASTRONOMICAL_UNIT_M`] (exact), and `d` the actual
//! Sun→spacecraft distance. [`SOLAR_PRESSURE_1AU_N_M2`] is *computed* as
//! `1361.0 / 299_792_458.0` so both the Rust stack and the fixture
//! generator produce the identical IEEE-754 value
//! (≈ 4.53980733564685e-6 N/m², asserted in the tests).

use nalgebra::Vector3;

use super::ephemeris::ASTRONOMICAL_UNIT_M;
use crate::geodetic::WGS84_A;

/// Solar radiation pressure at 1 AU (N/m²): nominal TSI 1361 W/m²
/// (IAU 2015 Resolution B3) over the exact speed of light — see the module
/// docs for the convention's provenance and the fixture record.
pub const SOLAR_PRESSURE_1AU_N_M2: f64 = 1361.0 / 299_792_458.0;

/// Radius of the cylindrical Earth shadow (m): the WGS-84 equatorial
/// radius, reused from [`crate::geodetic::WGS84_A`] — the modeling choice
/// recorded in the golden fixtures' `force_config.srp.shadow` blocks.
pub const SHADOW_CYLINDER_RADIUS_M: f64 = WGS84_A;

/// Cylindrical Earth-shadow factor: exactly `0.0` in umbra, exactly `1.0`
/// in sunlight (see the module docs for the rule and its documented
/// cylindrical approximation).
///
/// `r` is the spacecraft position and `sun` the Sun position, both
/// geocentric in the same inertial frame (metres).
pub fn cylindrical_shadow_factor(r: &Vector3<f64>, sun: &Vector3<f64>) -> f64 {
    let sun_hat = sun.normalize();
    let along = r.dot(&sun_hat);
    if along >= 0.0 {
        return 1.0;
    }
    let transverse = r - along * sun_hat;
    if transverse.norm() < SHADOW_CYLINDER_RADIUS_M {
        0.0
    } else {
        1.0
    }
}

/// Cannonball SRP acceleration (m/s²):
/// `a = ν · P₁AU · (AU/d)² · (Cr·A/m) · unit(r − r_sun)` with
/// `d = ‖r − r_sun‖` (Sun→spacecraft distance) and ν the
/// [`cylindrical_shadow_factor`] — anti-sunward when lit, **exactly zero**
/// in shadow.
///
/// * `r` — spacecraft position, geocentric inertial (m).
/// * `sun` — Sun position in the same frame (m), e.g.
///   [`super::ephemeris::sun_position_gcrf`].
/// * `cr_area_over_mass` — reflectivity coefficient times area over mass,
///   `Cr·A/m` (m²/kg).
pub fn srp_acceleration(
    r: &Vector3<f64>,
    sun: &Vector3<f64>,
    cr_area_over_mass: f64,
) -> Vector3<f64> {
    if cylindrical_shadow_factor(r, sun) == 0.0 {
        return Vector3::zeros();
    }
    let from_sun = r - sun;
    let distance = from_sun.norm();
    let ratio = ASTRONOMICAL_UNIT_M / distance;
    SOLAR_PRESSURE_1AU_N_M2 * ratio * ratio * cr_area_over_mass * (from_sun / distance)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Cr·A/m used across the tests (the LEO golden arc's value).
    const CR_A_M: f64 = 0.0065;

    /// The computed pressure constant is bit-identical to the value the
    /// fixture generator records (`pressure_at_1au_n_m2` in every
    /// SRP-bearing fixture) — the exactly-mirrorable convention resolving
    /// the design.md open question.
    #[test]
    fn pressure_constant_matches_fixture_record_bitwise() {
        assert_eq!(SOLAR_PRESSURE_1AU_N_M2, 4.53980733564685e-06);
        assert_eq!(SHADOW_CYLINDER_RADIUS_M, 6_378_137.0);
    }

    // ── Spec: "SRP vanishes in shadow" ───────────────────────────────────

    /// Inside the cylinder on the anti-Sun side the acceleration is exactly
    /// zero; outside it is nonzero and anti-sunward.
    #[test]
    fn srp_vanishes_in_shadow_and_pushes_anti_sunward_outside() {
        let sun = Vector3::new(ASTRONOMICAL_UNIT_M, 0.0, 0.0);

        // Anti-sunward on the axis at LEO radius: umbra.
        let shadowed = Vector3::new(-7.0e6, 0.0, 0.0);
        assert_eq!(cylindrical_shadow_factor(&shadowed, &sun), 0.0);
        assert_eq!(srp_acceleration(&shadowed, &sun, CR_A_M), Vector3::zeros());

        // Sun side at the same radius: lit, and the acceleration points
        // along unit(r − r_sun), i.e. away from the Sun.
        let lit = Vector3::new(7.0e6, 0.0, 0.0);
        assert_eq!(cylindrical_shadow_factor(&lit, &sun), 1.0);
        let accel = srp_acceleration(&lit, &sun, CR_A_M);
        assert!(accel.norm() > 0.0);
        let anti_sun = (lit - sun).normalize();
        assert!(accel.normalize().dot(&anti_sun) > 1.0 - 1e-12);
    }

    /// The shadow boundary geometry: anti-sunward but outside the cylinder
    /// radius is lit; the terminator plane (r ⊥ ŝ, `along = 0`) is lit by
    /// the `>= 0` rule — both straight from the fixture-recorded rule.
    #[test]
    fn shadow_boundary_follows_the_fixture_rule() {
        let sun = Vector3::new(ASTRONOMICAL_UNIT_M, 0.0, 0.0);

        // Anti-sunward, transverse offset just outside the cylinder.
        let outside = Vector3::new(-1.0e7, SHADOW_CYLINDER_RADIUS_M + 1.0, 0.0);
        assert_eq!(cylindrical_shadow_factor(&outside, &sun), 1.0);
        // Just inside.
        let inside = Vector3::new(-1.0e7, SHADOW_CYLINDER_RADIUS_M - 1.0, 0.0);
        assert_eq!(cylindrical_shadow_factor(&inside, &sun), 0.0);
        // On the terminator plane, inside the would-be cylinder: lit.
        let terminator = Vector3::new(0.0, 1.0e6, 0.0);
        assert_eq!(cylindrical_shadow_factor(&terminator, &sun), 1.0);
    }

    // ── Magnitude structure ──────────────────────────────────────────────

    /// Inverse-square distance scaling: at half the solar distance the
    /// magnitude is exactly 4 × the pressure-at-1AU baseline
    /// (`P₁AU · (AU/d)² · Cr·A/m`).
    #[test]
    fn magnitude_scales_inverse_square_with_solar_distance() {
        let sun = Vector3::new(ASTRONOMICAL_UNIT_M, 0.0, 0.0);
        // Spacecraft between Earth and Sun at half an AU: sunlit.
        let halfway = Vector3::new(ASTRONOMICAL_UNIT_M / 2.0, 0.0, 0.0);
        let accel = srp_acceleration(&halfway, &sun, CR_A_M);
        let expected = 4.0 * SOLAR_PRESSURE_1AU_N_M2 * CR_A_M;
        assert!((accel.norm() - expected).abs() <= 1e-12 * expected);
        assert!(accel.x < 0.0, "anti-sunward is −x here");
    }

    /// At geocentric LEO scale (d ≈ 1 AU) the magnitude is
    /// P₁AU · Cr·A/m to ~5 significant figures and scales linearly with
    /// Cr·A/m.
    #[test]
    fn magnitude_at_one_au_matches_pressure_times_area_over_mass() {
        let sun = Vector3::new(ASTRONOMICAL_UNIT_M, 0.0, 0.0);
        let r = Vector3::new(0.0, 7.0e6, 0.0);
        let accel = srp_acceleration(&r, &sun, CR_A_M);
        let expected = SOLAR_PRESSURE_1AU_N_M2 * CR_A_M;
        assert!((accel.norm() - expected).abs() <= 1e-4 * expected);

        let doubled = srp_acceleration(&r, &sun, 2.0 * CR_A_M);
        assert!((doubled.norm() - 2.0 * accel.norm()).abs() <= 1e-12 * accel.norm());
    }
}
