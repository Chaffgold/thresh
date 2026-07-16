//! Point-mass lunisolar third-body perturbation acceleration.
//!
//! Added by the `orbit-propagation-fidelity` change (design Decision 2's
//! third-body leg). The physics is the standard **direct minus indirect**
//! tidal formulation
//!
//! ```text
//! a = GM_b · ( (s − r)/‖s − r‖³ − s/‖s‖³ )
//! ```
//!
//! (`r` spacecraft, `s` third body, both geocentric inertial), evaluated
//! here in Battin's cancellation-free `F(q)` form — the resolution of the
//! design.md open question on numerical stabilization:
//!
//! ```text
//! a = −GM_b/‖r − s‖³ · ( r + F(q)·s ),
//! F(q) = q·(3 + 3q + q²) / (1 + (1 + q)^{3/2}),
//! q = r·(r − 2s) / (s·s)
//! ```
//!
//! The two forms are algebraically identical (`‖r − s‖² = s²(1 + q)`, so
//! `F(q) = (1 + q)^{3/2} − 1` without the subtraction), but the naive form
//! differences two nearly equal ~`GM_b/s²` vectors when `r ≪ s` — at LEO
//! radius against the Sun that cancellation costs ~4 significant digits,
//! measured (not folklore) in `stabilized_form_beats_naive_cancellation`
//! below against a colinear-geometry closed form. Not catastrophic, but
//! free to avoid.
//!
//! # Formula provenance (fetched 2026-07-16)
//!
//! Battin, *An Introduction to the Mathematics and Methods of
//! Astrodynamics*, Revised Ed., AIAA Education Series (1999),
//! pp. 448–450 (the Battin–Giorgi/Encke `F(q)`), transcribed from the
//! statement in Proietti & Pontani, *Long-Term Orbit Dynamics of
//! Decommissioned Geostationary Satellites*, Eq. (15)
//! (<https://arxiv.org/pdf/2104.01240>, which cites those Battin pages),
//! and re-derived algebraically from the direct-minus-indirect form (the
//! executable equivalence tests below pin both the transcription and the
//! derivation).
//!
//! # GM constants (fetched 2026-07-16)
//!
//! [`GM_SUN`] and [`GM_MOON`] are the JPL DE440 values from the JPL SSD
//! *Astrodynamic Parameters* page
//! (<https://ssd.jpl.nasa.gov/astro_par.html>): GM_Sun =
//! 1.32712440041279419e20 m³/s², GM_Moon = 4902.800118 km³/s² — the same
//! values recorded in every third-body-bearing golden fixture's
//! `force_config` block.

use nalgebra::Vector3;

/// Heliocentric gravitational constant GM☉ (m³/s²), JPL DE440 — see the
/// module docs for provenance. The JPL-printed value is
/// 1.32712440041279419e20; the literal below is its (identical) shortest
/// f64 round-trip form, matching the fixtures' `gm_sun_m3_s2` bit for bit.
pub const GM_SUN: f64 = 1.327_124_400_412_794_2e20;

/// Lunar gravitational constant GM☾ (m³/s²), JPL DE440 (4902.800118
/// km³/s²) — see the module docs for provenance.
pub const GM_MOON: f64 = 4.902_800_118e12;

/// Battin's `F(q) = q·(3 + 3q + q²)/(1 + (1 + q)^{3/2})` — the
/// cancellation-free evaluation of `(1 + q)^{3/2} − 1` (module docs).
fn battin_f(q: f64) -> f64 {
    q * (3.0 + q * (3.0 + q)) / (1.0 + (1.0 + q).powf(1.5))
}

/// Third-body perturbation acceleration (m/s²) on a spacecraft at
/// geocentric position `r` from a point-mass body at geocentric position
/// `body` with gravitational parameter `gm_body` (typically [`GM_SUN`] or
/// [`GM_MOON`], positions from
/// [`super::ephemeris::sun_position_gcrf`] /
/// [`super::ephemeris::moon_position_gcrf`]).
///
/// Direct minus indirect (the differential pull on the spacecraft versus
/// the Earth), evaluated in the Battin `F(q)` stabilized form — see the
/// module docs for the formulation, its provenance, and the measured
/// cancellation comparison. Both vectors must share one inertial frame.
pub fn third_body_acceleration(
    r: &Vector3<f64>,
    body: &Vector3<f64>,
    gm_body: f64,
) -> Vector3<f64> {
    let q = r.dot(&(r - 2.0 * body)) / body.norm_squared();
    let relative = r - body;
    let d = relative.norm();
    -(gm_body / (d * d * d)) * (r + battin_f(q) * body)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::frames::legs::julian_centuries_tt;
    use crate::orbital::ephemeris::{
        ASTRONOMICAL_UNIT_M, moon_position_gcrf, moon_position_mod, sun_position_mod,
    };
    use crate::time::Epoch;

    /// The naive direct-minus-indirect form — the reference implementation
    /// for the equivalence and cancellation measurements (kept test-only;
    /// production uses the stabilized form).
    fn third_body_naive(r: &Vector3<f64>, body: &Vector3<f64>, gm: f64) -> Vector3<f64> {
        let direct = body - r;
        let dn = direct.norm();
        let bn = body.norm();
        gm * (direct / (dn * dn * dn) - body / (bn * bn * bn))
    }

    // ── F(q) structure ───────────────────────────────────────────────────

    /// F(q) ≡ (1+q)^{3/2} − 1 where the subtraction is benign, and the
    /// small-q leading term is 3q/2.
    #[test]
    fn battin_f_matches_its_closed_form() {
        for q in [-0.5_f64, -0.1, 0.0, 0.3, 1.0, 4.0] {
            let direct = (1.0 + q).powf(1.5) - 1.0;
            assert!(
                (battin_f(q) - direct).abs() <= 1e-12 * direct.abs().max(1.0),
                "F({q})"
            );
        }
        let tiny = 1e-9;
        assert!((battin_f(tiny) - 1.5 * tiny).abs() <= 1e-17);
    }

    // ── Formulation equivalence (no cancellation regime) ────────────────

    /// At GEO radius against the real fixture-epoch Moon and Sun the
    /// stabilized and naive forms agree to ≤ 1e-12 relative — the two
    /// code paths implement the same math (and the naive form is the
    /// generator's convention, so this also pins the fixture formulation).
    #[test]
    fn stabilized_equals_naive_where_both_are_accurate() {
        let epoch = Epoch::from_iso8601("2026-05-10T00:00:00.000Z").unwrap();
        let t_tt = julian_centuries_tt(&epoch);
        let r = Vector3::new(30_000_000.0, -25_000_000.0, 8_000_000.0);
        for (body, gm) in [
            (moon_position_mod(t_tt), GM_MOON),
            (sun_position_mod(t_tt), GM_SUN),
        ] {
            let stabilized = third_body_acceleration(&r, &body, gm);
            let naive = third_body_naive(&r, &body, gm);
            assert!(
                (stabilized - naive).norm() <= 1e-12 * naive.norm(),
                "forms disagree: {stabilized:?} vs {naive:?}"
            );
        }
    }

    // ── The design.md open question, resolved by measurement ────────────

    /// Cancellation test at LEO radius (design open question / spec "SHALL
    /// NOT suffer catastrophic cancellation"): in colinear geometry the
    /// exact answer has the cancellation-free closed form
    /// `a = GM·k·(2s − k)/((s − k)²·s²)` along the line (from
    /// 1/(s−k)² − 1/s² over a common denominator). Against it, at solar
    /// distance, the stabilized form is exact to round-off (measured worst
    /// 4.0e-16 relative over the sampled LEO radii) while the naive
    /// difference loses ~4 digits (measured worst 1.6e-12) — the basis for
    /// adopting F(q), recorded in design.md.
    #[test]
    fn stabilized_form_beats_naive_cancellation_at_leo() {
        let s = ASTRONOMICAL_UNIT_M;
        let body = Vector3::new(s, 0.0, 0.0);
        let mut worst_stabilized = 0.0_f64;
        let mut worst_naive = 0.0_f64;
        for k in [6.6e6, 6.778e6, 6.9e6, 7.1e6] {
            let r = Vector3::new(k, 0.0, 0.0);
            let exact = GM_SUN * k * (2.0 * s - k) / ((s - k) * (s - k) * s * s);
            let stabilized = third_body_acceleration(&r, &body, GM_SUN).x;
            let naive = third_body_naive(&r, &body, GM_SUN).x;
            worst_stabilized = worst_stabilized.max((stabilized - exact).abs() / exact);
            worst_naive = worst_naive.max((naive - exact).abs() / exact);
        }
        assert!(
            worst_stabilized <= 1e-14,
            "stabilized error {worst_stabilized}"
        );
        assert!(
            worst_naive > worst_stabilized,
            "naive {worst_naive} should show the cancellation the stabilized \
             form avoids ({worst_stabilized})"
        );
        // The naive loss is real but bounded (~4 digits, not catastrophic).
        assert!(worst_naive < 1e-10, "naive error {worst_naive}");
    }

    // ── Spec: "Third-body magnitude at GEO-class radius" ─────────────────

    /// Lunar third-body acceleration at GEO radius, satellite placed along
    /// the fixture-epoch Moon direction: magnitude in the documented
    /// ~1e-6 m/s² range (colinear tide ≈ 2·GM☾·r/d³ ≈ 6–9e-6 over the
    /// lunar distance band) and pointing at the Moon.
    #[test]
    fn lunar_acceleration_at_geo_is_micro_scale_and_points_at_the_moon() {
        let epoch = Epoch::from_iso8601("2026-05-10T00:00:00.000Z").unwrap();
        let moon = moon_position_gcrf(&epoch);
        let r = 42_164_000.0 * moon.normalize();
        let accel = third_body_acceleration(&r, &moon, GM_MOON);
        assert!(
            (1e-6..1e-5).contains(&accel.norm()),
            "magnitude {} outside the documented ~1e-6 m/s² range",
            accel.norm()
        );
        assert!(
            accel.normalize().dot(&moon.normalize()) > 1.0 - 1e-9,
            "colinear geometry must pull straight toward the Moon"
        );
    }

    /// Structural: on the far side (anti-Moon) the tide pulls *away* from
    /// the Moon, and the near-side pull exceeds the far-side pull.
    #[test]
    fn tide_reverses_sign_across_the_earth() {
        let epoch = Epoch::from_iso8601("2026-05-10T00:00:00.000Z").unwrap();
        let moon = moon_position_gcrf(&epoch);
        let near = 42_164_000.0 * moon.normalize();
        let far = -near;
        let a_near = third_body_acceleration(&near, &moon, GM_MOON);
        let a_far = third_body_acceleration(&far, &moon, GM_MOON);
        assert!(a_far.normalize().dot(&moon.normalize()) < -(1.0 - 1e-9));
        assert!(a_near.norm() > a_far.norm());
    }
}
