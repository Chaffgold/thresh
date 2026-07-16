//! Atmospheric measurement propagation: deterministic biases microwave
//! radar measurements actually carry, and the corrections that remove
//! them (`atmospheric-measurement-propagation` change).
//!
//! # Conventions (stated once, here)
//!
//! - A **measured** (apparent) quantity equals the **true** geometric
//!   quantity plus the propagation **bias**: refraction lifts apparent
//!   elevation and lengthens range; ionospheric group delay lengthens
//!   range. Corrections **subtract** modeled biases from measured values.
//! - Atmospheric **attenuation** (SNR/P_d, ITU-R P.676) is explicitly
//!   not this module's concern — see the radar-equation capability.
//! - The OTHR/HF sky-wave machinery (`thresh-synth`'s Chapman/MUF/skip
//!   zones) is a different propagation regime and is untouched; the
//!   ionospheric side here is microwave group delay only.

pub mod iono_delay;
pub mod refraction;

use iono_delay::{IonoDelayConfig, correct_iono_delay};
use refraction::{RefractionConfig, correct_refraction};

/// Remove both modeled atmospheric biases from an apparent (measured) radar
/// `(elevation, range)`, composing the ionospheric and tropospheric inverses
/// (task 1.5 of the `atmospheric-measurement-propagation` change).
///
/// The inverse is applied in the reverse order of the forward biasing: the
/// forward path (synthetic generation, `thresh-synth`) first bends the
/// elevation and lengthens the range by refraction, then adds the ionospheric
/// group delay evaluated at the *apparent* (bent) elevation. So the correction
///
/// 1. subtracts the ionospheric delay at the apparent elevation
///    ([`correct_iono_delay`]) — range-only, exact and range-independent, so it
///    cancels the forward delay to floating-point precision; then
/// 2. removes the refraction bias ([`correct_refraction`], one documented
///    fixed-point step) from the de-ionised range and the apparent elevation.
///
/// Either model is optional (`None` skips that term). Returns
/// `(corrected_elevation_rad, corrected_range_m)`. With the same parameters the
/// generation used, a bias-then-correct round trip recovers the geometry to the
/// refraction quadrature / one-step fixed-point tolerance (the ionospheric leg
/// is exact); see the `round_trip_recovers_truth` test.
///
/// Like the forward primitives, the ionospheric leg here is the ungated
/// full-traversal delay. A caller whose forward path gated that leg on target
/// altitude (the step-function simplification documented at
/// [`IonoDelayConfig`] — e.g. `thresh-synth`'s `apply_to_rae`) must gate the
/// inverse identically, passing `iono = None` for sub-ionospheric targets so
/// no delay that was never applied gets subtracted.
///
/// `station_alt_m` / `target_alt_m` are the refraction ray endpoints (m above
/// the surface); they are ignored when `refraction` is `None`.
pub fn correct_atmosphere(
    refraction: Option<&RefractionConfig>,
    iono: Option<&IonoDelayConfig>,
    apparent_elevation_rad: f64,
    apparent_range_m: f64,
    station_alt_m: f64,
    target_alt_m: f64,
) -> (f64, f64) {
    let de_ionised_range_m = match iono {
        Some(cfg) => correct_iono_delay(cfg, apparent_range_m, apparent_elevation_rad),
        None => apparent_range_m,
    };
    match refraction {
        Some(cfg) => correct_refraction(
            cfg,
            apparent_elevation_rad,
            de_ionised_range_m,
            station_alt_m,
            target_alt_m,
        ),
        None => (apparent_elevation_rad, de_ionised_range_m),
    }
}

#[cfg(test)]
mod tests {
    use super::iono_delay::{IonoDelayConfig, iono_range_delay_m};
    use super::refraction::{RefractionConfig, apply_refraction};
    use super::*;
    use std::f64::consts::PI;

    fn rad(deg: f64) -> f64 {
        deg * PI / 180.0
    }

    /// Reproduce the forward biasing `thresh-synth`'s generation applies:
    /// refraction first (apparent elevation + excess range), then the
    /// ionospheric group delay evaluated at the apparent elevation. Returns
    /// the apparent `(elevation, range)` a biased measurement would carry
    /// (before noise).
    fn apply_forward(
        refr: &RefractionConfig,
        iono: &IonoDelayConfig,
        true_el: f64,
        true_range: f64,
        station_alt_m: f64,
        target_alt_m: f64,
    ) -> (f64, f64) {
        let (app_el, refr_range) =
            apply_refraction(refr, true_el, true_range, station_alt_m, target_alt_m);
        (app_el, refr_range + iono_range_delay_m(iono, app_el))
    }

    // Task 1.5 (composed): bias with both models, then `correct_atmosphere`
    // with the same parameters recovers the geometry. The ionospheric leg
    // cancels exactly (range-independent, evaluated at the identical apparent
    // elevation both ways); the refraction leg leaves only its one-step
    // fixed-point residual, so the whole round trip sits at the refraction
    // tolerance. Recorded residuals (CRPL + 15 TECU L-band, space target):
    // 20° → 0.46″ elevation / 0.015 m range; 45° → 0.04″ / 0.001 m.
    #[test]
    fn round_trip_recovers_truth() {
        let refr = RefractionConfig::crpl();
        let iono = IonoDelayConfig::new(15.0, 1.3e9);
        let true_range = 150_000.0;
        for &theta_deg in &[20.0, 30.0, 45.0] {
            let true_el = rad(theta_deg);
            let (app_el, app_range) = apply_forward(&refr, &iono, true_el, true_range, 0.0, 1.0e9);
            // The composed forward bias is strictly positive in both channels.
            assert!(app_el > true_el && app_range > true_range);

            let (corr_el, corr_range) =
                correct_atmosphere(Some(&refr), Some(&iono), app_el, app_range, 0.0, 1.0e9);
            let el_resid_arcsec = (corr_el - true_el).to_degrees().abs() * 3600.0;
            let range_resid = (corr_range - true_range).abs();
            assert!(
                el_resid_arcsec < 2.0,
                "θ={theta_deg}°: elevation round-trip residual {el_resid_arcsec:.4}″"
            );
            assert!(
                range_resid < 0.05,
                "θ={theta_deg}°: range round-trip residual {range_resid:.5} m"
            );
        }
    }

    // The ionospheric leg of the composition is exact: with refraction absent,
    // correcting the ionosphere-only bias returns the range to floating point.
    #[test]
    fn iono_only_leg_is_exact() {
        let iono = IonoDelayConfig::new(20.0, 1.3e9);
        let true_el = rad(10.0);
        let true_range = 200_000.0;
        let biased_range = true_range + iono_range_delay_m(&iono, true_el);
        let (el, range) = correct_atmosphere(None, Some(&iono), true_el, biased_range, 0.0, 1.0e9);
        assert_eq!(el.to_bits(), true_el.to_bits(), "elevation untouched");
        assert!(
            (range - true_range).abs() < 1e-8,
            "range residual {}",
            range - true_range
        );
    }

    // Both models absent: `correct_atmosphere` is the identity (no seam cost
    // for the default, correction-off benchmark path).
    #[test]
    fn no_models_is_identity() {
        let (el, range) = correct_atmosphere(None, None, rad(7.5), 123_456.0, 12.0, 40_000.0);
        assert_eq!(el.to_bits(), rad(7.5).to_bits());
        assert_eq!(range.to_bits(), 123_456.0_f64.to_bits());
    }
}
