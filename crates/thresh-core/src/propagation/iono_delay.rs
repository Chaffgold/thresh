//! Microwave ionospheric group delay: `Δr = K·STEC/f²` with thin-shell
//! obliquity mapping from a configured vertical TEC (design Decision 3 of
//! the `atmospheric-measurement-propagation` change).
//!
//! # Model
//!
//! The first-order ionospheric group delay lengthens a measured radar range
//! by
//!
//! ```text
//! Δr = K · STEC / f²        (metres, with STEC in el/m² and f in Hz)
//! ```
//!
//! where [`IONO_DELAY_K_M3_PER_S2`] is the dispersion constant and the
//! slant TEC comes from the configured **vertical** TEC through the
//! standard single-layer ("thin-shell") obliquity mapping
//!
//! ```text
//! STEC = VTEC · M(E),   M(E) = 1 / √(1 − (Rₑ·cos E / (Rₑ + h_s))²)
//! ```
//!
//! with `E` the elevation angle at the station and `h_s` the shell height
//! ([`DEFAULT_SHELL_HEIGHT_M`]). `M` is 1 at zenith and grows monotonically
//! toward the horizon, where it stays finite (≈ 2.8 for the default shell).
//!
//! # Sign and scope
//!
//! - Group (code/ranging) delay is **positive** — the measured range is
//!   longer than the geometric range; the corresponding carrier-**phase**
//!   advance has the opposite sign and is irrelevant to radar ranging
//!   (ESA Navipedia, *Ionospheric Delay*,
//!   <https://gssc.esa.int/navipedia/index.php/Ionospheric_Delay>, fetched
//!   2026-07-16: code measurements suffer a positive delay, phase
//!   measurements an advance). Per the module conventions
//!   (`crate::propagation`), [`correct_iono_delay`] therefore *subtracts*
//!   the modeled delay from a measured range.
//! - Ionospheric **elevation bending** at microwave frequencies is
//!   negligible (it shares the 1/f² dispersion scaling that makes the
//!   metre-scale HF effect collapse by orders of magnitude at GHz, far
//!   below tropospheric bending) and is **not modeled** here; elevation
//!   bending is the troposphere module's concern.
//! - Higher-order ionospheric terms (f⁻³, f⁻⁴) are below the modeled
//!   first-order term by orders of magnitude at radar frequencies and are
//!   out of scope.
//! - The OTHR/HF sky-wave machinery in `thresh-synth` is a different
//!   propagation regime and is untouched; [`chapman_vtec_tecu`] only
//!   mirrors its Chapman *profile shape* so the two regimes can share one
//!   consistent ionosphere in scenarios that use both.
//!
//! # Magnitude regimes
//!
//! At a representative quiet mid-latitude vertical TEC of 10 TECU and
//! moderate (45°) elevation, the delay is metre-scale at L-band
//! (~3.2 m at 1.3 GHz), decimetre-scale at S-band (~0.6 m at 3 GHz), and
//! centimetre-scale at X-band (~5 cm at 10 GHz) — pinned by the
//! `band_magnitudes_are_physical` test.

use serde::{Deserialize, Serialize};
use std::f64::consts::FRAC_PI_2;

/// First-order ionospheric dispersion constant `K` (m³·s⁻²).
///
/// `Δr = K·STEC/f²` with `STEC` in electrons/m² and `f` in Hz yields metres
/// (units equivalently m·m²·Hz²). Value κ ≈ 40.308193 m³·s⁻², derived from
/// physical constants as `κ = q²/(8π²·ε₀·mₑ) = c²·rₑ/(2π)` (electron charge
/// `q`, electron mass `mₑ`, vacuum permittivity `ε₀`, classical electron
/// radius `rₑ`). Source: Wikipedia, *Total electron content*,
/// <https://en.wikipedia.org/wiki/Total_electron_content>, fetched
/// 2026-07-16. The familiar "40.3" (e.g. ESA Navipedia's
/// `α_f = 40.3·10¹⁶/f²` m/TECU) is a rounding of this value.
pub const IONO_DELAY_K_M3_PER_S2: f64 = 40.308_193;

/// Electrons per m² in one TEC unit: `1 TECU = 10¹⁶ el/m²`.
///
/// Source: ESA Navipedia, *Ionospheric Delay*,
/// <https://gssc.esa.int/navipedia/index.php/Ionospheric_Delay>, fetched
/// 2026-07-16 (also Wikipedia, *Total electron content*).
pub const TECU_ELECTRONS_PER_M2: f64 = 1.0e16;

/// Default thin-shell (single-layer) ionosphere height (m).
///
/// 450 km is the published single-layer model height used by the IGS
/// convention and the GLONASS IAC ionosphere methods page
/// (<https://glonass-iac.ru/en/iono/methods/>, fetched 2026-07-16:
/// `h = 450 km` with `cos E′ = (R⊕/(R⊕+h))·cos E`). Published practice
/// spans 350–450 km (Ren et al., Ann. Geophys. 37:263, 2019,
/// <https://angeo.copernicus.org/articles/37/263/2019/>, fetched
/// 2026-07-16: "the shell height is typically set to a fixed value between
/// 350 and 450 km"); the height is configurable via
/// [`IonoDelayConfig::shell_height_m`].
pub const DEFAULT_SHELL_HEIGHT_M: f64 = 450_000.0;

/// Conventional mean Earth radius (m) used by the thin-shell geometry.
///
/// The single-layer mapping literature uses the spherical mean radius, not
/// the WGS-84 equatorial radius. 6371 km is the conventional globally
/// averaged value (IUGG arithmetic mean radius R₁ = 6371.0087714 km;
/// Wikipedia, *Earth radius*, <https://en.wikipedia.org/wiki/Earth_radius>,
/// fetched 2026-07-16), and matches `thresh-synth`'s OTHR
/// `EARTH_RADIUS_KM = 6371.0` so both ionospheric regimes share one Earth.
pub const MEAN_EARTH_RADIUS_M: f64 = 6_371_000.0;

/// Configuration for the microwave ionospheric group-delay model.
///
/// Pure data: vertical TEC in TEC units, thin-shell height in metres
/// (serde-defaulted to [`DEFAULT_SHELL_HEIGHT_M`]), and the radar carrier
/// frequency in Hz.
///
/// # Target-altitude step-function simplification
///
/// The thin-shell model computes the delay for a ray traversing the **whole**
/// ionosphere (a target at or above the shell). Consumers that know the
/// target altitude (e.g. `AtmosphereBiasConfig::apply_to_rae` in
/// `thresh-synth`) apply the delay as a step function of that altitude:
/// targets at or above [`Self::shell_height_m`] traverse effectively all of
/// the TEC and get the full slant delay; sub-ionospheric targets (aircraft,
/// low ballistic segments) accumulate essentially none of it and get zero.
/// Fractional traversal for targets *inside* the ionosphere is out of scope —
/// the electron content is concentrated around the shell height, so the step
/// is the documented approximation for intermediate altitudes. The pure
/// functions in this module ([`iono_range_delay_m`], [`correct_iono_delay`])
/// are deliberately ungated: they answer "the full-traversal delay at this
/// elevation", and the altitude gate belongs to the caller that knows the
/// geometry.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct IonoDelayConfig {
    /// Vertical total electron content (TECU; 1 TECU = 10¹⁶ el/m²).
    pub vtec_tecu: f64,
    /// Thin-shell ionosphere height (m). Defaults to
    /// [`DEFAULT_SHELL_HEIGHT_M`] when omitted from serialized configs.
    #[serde(default = "default_shell_height_m")]
    pub shell_height_m: f64,
    /// Radar carrier frequency (Hz).
    pub frequency_hz: f64,
}

impl IonoDelayConfig {
    /// Config with the published default shell height
    /// ([`DEFAULT_SHELL_HEIGHT_M`]).
    pub fn new(vtec_tecu: f64, frequency_hz: f64) -> Self {
        Self {
            vtec_tecu,
            shell_height_m: DEFAULT_SHELL_HEIGHT_M,
            frequency_hz,
        }
    }
}

/// Serde default for [`IonoDelayConfig::shell_height_m`].
fn default_shell_height_m() -> f64 {
    DEFAULT_SHELL_HEIGHT_M
}

/// Thin-shell obliquity (mapping) factor `M(E)` converting vertical TEC to
/// slant TEC: `STEC = VTEC · M(E)`.
///
/// `M(E) = 1/√(1 − (Rₑ·cos E/(Rₑ+h_s))²)` — algebraically `1/sin E′` with
/// `cos E′ = (Rₑ/(Rₑ+h_s))·cos E`, the standard single-layer mapping
/// (GLONASS IAC, <https://glonass-iac.ru/en/iono/methods/>; equivalently
/// `f(z) = 1/cos z`, `z = arcsin(Rₑ·cos E/(Rₑ+h_s))` in Ren et al., Ann.
/// Geophys. 37:263, 2019 — both fetched 2026-07-16). `Rₑ` is
/// [`MEAN_EARTH_RADIUS_M`].
///
/// `M` is 1 at zenith, grows monotonically toward the horizon, and remains
/// finite at `E = 0`. The elevation is clamped to `[0, π/2]`: slightly
/// negative apparent elevations (radar horizon dips) evaluate at the
/// horizon obliquity rather than extrapolating the shell geometry.
pub fn thin_shell_obliquity(elevation_rad: f64, shell_height_m: f64) -> f64 {
    let el = elevation_rad.clamp(0.0, FRAC_PI_2);
    let ratio = MEAN_EARTH_RADIUS_M / (MEAN_EARTH_RADIUS_M + shell_height_m);
    let cos_zp = ratio * el.cos(); // cos E′ at the ionospheric pierce point
    1.0 / (1.0 - cos_zp * cos_zp).sqrt()
}

/// Ionospheric group delay `Δr = K·STEC/f²` (m, positive = range
/// lengthening) at the given apparent elevation.
///
/// `STEC = vtec_tecu · TECU_ELECTRONS_PER_M2 · M(E)` with the thin-shell
/// mapping [`thin_shell_obliquity`]. Deterministic pure function; see the
/// module docs for constants, sign convention, and magnitude regimes.
pub fn iono_range_delay_m(config: &IonoDelayConfig, elevation_rad: f64) -> f64 {
    let stec_el_per_m2 = config.vtec_tecu
        * TECU_ELECTRONS_PER_M2
        * thin_shell_obliquity(elevation_rad, config.shell_height_m);
    IONO_DELAY_K_M3_PER_S2 * stec_el_per_m2 / (config.frequency_hz * config.frequency_hz)
}

/// Remove the modeled ionospheric delay from a measured range (m).
///
/// Closed-form inverse of [`iono_range_delay_m`]: the delay does not depend
/// on range, so the correction is an exact subtraction —
/// `r_corrected = r_measured − Δr(config, E)`. With the true parameters the
/// bias-then-correct round trip recovers the geometric range to
/// floating-point precision (two roundings); with a mis-set vertical TEC
/// the residual equals exactly the mis-set fraction of the true delay
/// (the delay is linear in VTEC) — see the
/// `mismatched_tec_correction_still_helps` test for recorded values.
///
/// The elevation passed here is the *measured* (apparent) elevation;
/// ionospheric bending is unmodeled (module docs), so no elevation
/// iteration is needed.
pub fn correct_iono_delay(
    config: &IonoDelayConfig,
    measured_range_m: f64,
    elevation_rad: f64,
) -> f64 {
    measured_range_m - iono_range_delay_m(config, elevation_rad)
}

/// Chapman-layer electron density (el/m³) at height `height_m`, mirroring
/// `thresh-synth`'s OTHR profile exactly.
///
/// `N(h) = N_max · exp(0.5·(1 − z − e^(−z)))`, `z = (h − hm)/H` — the same
/// functional form as `chapman_density` in
/// `crates/thresh-synth/src/ionosphere.rs` (which takes km; `z` is
/// dimensionless, so metres here produce the identical shape). Kept private:
/// the public bridge is [`chapman_vtec_tecu`]; the OTHR code itself is the
/// authority on HF propagation and is untouched.
fn chapman_electron_density_el_per_m3(
    height_m: f64,
    n_max_el_per_m3: f64,
    hm_m: f64,
    scale_height_m: f64,
) -> f64 {
    let z = (height_m - hm_m) / scale_height_m;
    n_max_el_per_m3 * (0.5 * (1.0 - z - (-z).exp())).exp()
}

/// Fixed Simpson interval count for [`chapman_vtec_tecu`] (even, fixed for
/// determinism; quadrature error is negligible against the documented
/// 1e-7 relative bridge tolerance — see `hand_integrated_profile_matches`).
const CHAPMAN_SIMPSON_INTERVALS: usize = 4096;

/// Chapman-profile → vertical TEC bridge (TECU).
///
/// Integrates the Chapman electron-density profile (the exact functional
/// form of `thresh-synth`'s OTHR `chapman_density`, see
/// `chapman_electron_density_el_per_m3`) over height to a vertical TEC:
///
/// ```text
/// VTEC [TECU] = ∫₀^{hm+40H} N(h) dh  [el/m²]  /  10¹⁶ [el/m² per TECU]
/// ```
///
/// Parameters are SI: `n_max_el_per_m3` is the peak density (el/m³), `hm_m`
/// the peak height (m), `scale_height_m` the scale height (m) — callers
/// holding `thresh-synth`'s km-denominated `IonosphereParams` convert km→m
/// and their `n_max` is already el/m³. This is a convenience constructor
/// for [`IonoDelayConfig::vtec_tecu`], not a coupling of the OTHR and
/// microwave capabilities.
///
/// Quadrature: composite Simpson with `CHAPMAN_SIMPSON_INTERVALS` (4096)
/// fixed intervals over `[0, hm + 40·H]` (deterministic). The topside tail
/// above
/// `hm + 40·H` decays as `e^(−z/2)` and contributes < 1e-8 relative for any
/// parameters; the analytic full-line integral is `N_max·H·√(2πe)`
/// (documented tolerance 1e-7 relative in `hand_integrated_profile_matches`
/// for F-layer-like `hm/H`, where the below-ground truncation is also
/// negligible).
pub fn chapman_vtec_tecu(n_max_el_per_m3: f64, hm_m: f64, scale_height_m: f64) -> f64 {
    let upper_m = hm_m + 40.0 * scale_height_m;
    let step_m = upper_m / CHAPMAN_SIMPSON_INTERVALS as f64;
    let density =
        |h: f64| chapman_electron_density_el_per_m3(h, n_max_el_per_m3, hm_m, scale_height_m);
    let mut sum = density(0.0) + density(upper_m);
    for i in 1..CHAPMAN_SIMPSON_INTERVALS {
        let weight = if i % 2 == 1 { 4.0 } else { 2.0 };
        sum += weight * density(step_m * i as f64);
    }
    let vtec_el_per_m2 = sum * step_m / 3.0;
    vtec_el_per_m2 / TECU_ELECTRONS_PER_M2
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::{E, PI};

    /// Representative test geometry: quiet mid-latitude vertical TEC,
    /// moderate elevation, default (cited) shell height.
    const TEST_VTEC_TECU: f64 = 10.0;
    const TEST_ELEVATION_RAD: f64 = 45.0 * PI / 180.0;
    const L_BAND_HZ: f64 = 1.3e9;

    /// Tripwire: the cited constants must not drift from their fetched
    /// sources (module-doc citations, all fetched 2026-07-16). Any edit to
    /// these literals must re-fetch and re-cite.
    #[test]
    fn constant_provenance_tripwire() {
        assert_eq!(IONO_DELAY_K_M3_PER_S2.to_bits(), 40.308_193_f64.to_bits());
        assert_eq!(TECU_ELECTRONS_PER_M2.to_bits(), 1.0e16_f64.to_bits());
        assert_eq!(DEFAULT_SHELL_HEIGHT_M.to_bits(), 450_000.0_f64.to_bits());
        assert_eq!(MEAN_EARTH_RADIUS_M.to_bits(), 6_371_000.0_f64.to_bits());
    }

    // Spec scenario: "Inverse-square frequency scaling" — same geometry and
    // TEC, two frequencies an octave apart, delays scale as 1/f² exactly.
    // Exact in IEEE-754: doubling f is exact, squaring then dividing by the
    // power-of-two-scaled square commutes with rounding, so
    // delay(f) == 4·delay(2f) bit-for-bit.
    #[test]
    fn inverse_square_frequency_scaling() {
        let base = IonoDelayConfig::new(TEST_VTEC_TECU, L_BAND_HZ);
        let octave = IonoDelayConfig::new(TEST_VTEC_TECU, 2.0 * L_BAND_HZ);
        let delay_f = iono_range_delay_m(&base, TEST_ELEVATION_RAD);
        let delay_2f = iono_range_delay_m(&octave, TEST_ELEVATION_RAD);
        assert_eq!(delay_f.to_bits(), (4.0 * delay_2f).to_bits());
    }

    // Spec scenario: "Band magnitudes are physical" — at 10 TECU vertical
    // and 45° elevation the delay regimes are metres (L), decimetres (S),
    // centimetres (X). Recorded values: ≈3.18 m at 1.3 GHz, ≈0.60 m at
    // 3.0 GHz, ≈0.054 m at 10 GHz.
    #[test]
    fn band_magnitudes_are_physical() {
        let bands = [
            ("L-band 1.3 GHz", 1.3e9, 1.0, 10.0),
            ("S-band 3.0 GHz", 3.0e9, 0.1, 1.0),
            ("X-band 10 GHz", 10.0e9, 0.01, 0.1),
        ];
        for (name, freq_hz, lo_m, hi_m) in bands {
            let config = IonoDelayConfig::new(TEST_VTEC_TECU, freq_hz);
            let delay = iono_range_delay_m(&config, TEST_ELEVATION_RAD);
            assert!(
                delay >= lo_m && delay < hi_m,
                "{name}: delay {delay} m outside [{lo_m}, {hi_m}) m"
            );
        }
    }

    // Spec scenario: "Obliquity grows toward the horizon" — slant delay at
    // fixed VTEC increases strictly monotonically from zenith down to the
    // horizon, tracking the thin-shell mapping factor.
    #[test]
    fn obliquity_grows_toward_the_horizon() {
        let config = IonoDelayConfig::new(TEST_VTEC_TECU, L_BAND_HZ);
        let zenith_m = thin_shell_obliquity(FRAC_PI_2, DEFAULT_SHELL_HEIGHT_M);
        assert!(
            (zenith_m - 1.0).abs() < 1e-15,
            "mapping at zenith must be 1, got {zenith_m}"
        );
        let mut prev_delay = iono_range_delay_m(&config, FRAC_PI_2);
        let mut prev_obliquity = zenith_m;
        for deg in (0..90).rev() {
            let el = f64::from(deg) * PI / 180.0;
            let obliquity = thin_shell_obliquity(el, DEFAULT_SHELL_HEIGHT_M);
            let delay = iono_range_delay_m(&config, el);
            assert!(
                obliquity > prev_obliquity,
                "obliquity not monotonic at {deg}°: {obliquity} <= {prev_obliquity}"
            );
            assert!(
                delay > prev_delay,
                "delay not monotonic at {deg}°: {delay} <= {prev_delay}"
            );
            prev_obliquity = obliquity;
            prev_delay = delay;
        }
        // Horizon mapping stays finite; below-horizon clamps to it.
        assert!(prev_obliquity.is_finite());
        let below = thin_shell_obliquity(-0.02, DEFAULT_SHELL_HEIGHT_M);
        assert_eq!(below.to_bits(), prev_obliquity.to_bits());
    }

    // Spec scenario: "Closed-form round trip" — bias then correct with
    // identical parameters recovers the range to floating-point precision
    // (two roundings: one add, one subtract of the identical delay value).
    #[test]
    fn closed_form_round_trip() {
        let config = IonoDelayConfig::new(TEST_VTEC_TECU, L_BAND_HZ);
        let true_range_m = 150_000.0;
        let measured_m = true_range_m + iono_range_delay_m(&config, TEST_ELEVATION_RAD);
        let corrected_m = correct_iono_delay(&config, measured_m, TEST_ELEVATION_RAD);
        assert!(
            (corrected_m - true_range_m).abs() < 1e-8,
            "round trip residual {} m exceeds floating-point tolerance",
            corrected_m - true_range_m
        );
    }

    // Spec scenario: "Mismatched TEC correction still helps" — correcting
    // with VTEC mis-set by ±25% leaves exactly the mis-set fraction of the
    // true delay (the model is linear in VTEC). Recorded at this geometry
    // (10 TECU, 45°, L-band 1.3 GHz): true delay ≈ 3.176 m; +25% VTEC
    // over-corrects leaving ≈ −0.794 m; −25% under-corrects leaving
    // ≈ +0.794 m — both 4× smaller than the uncorrected 3.176 m bias.
    #[test]
    fn mismatched_tec_correction_still_helps() {
        let true_config = IonoDelayConfig::new(TEST_VTEC_TECU, L_BAND_HZ);
        let true_range_m = 150_000.0;
        let true_delay_m = iono_range_delay_m(&true_config, TEST_ELEVATION_RAD);
        let measured_m = true_range_m + true_delay_m;
        // Record the magnitude the fractions below are relative to.
        assert!((true_delay_m - 3.176).abs() < 5e-3, "geometry drifted");

        for misset_fraction in [0.25, -0.25] {
            let misset_config =
                IonoDelayConfig::new(TEST_VTEC_TECU * (1.0 + misset_fraction), L_BAND_HZ);
            let corrected_m = correct_iono_delay(&misset_config, measured_m, TEST_ELEVATION_RAD);
            let residual_m = corrected_m - true_range_m;
            let expected_residual_m = -misset_fraction * true_delay_m;
            assert!(
                (residual_m - expected_residual_m).abs() < 1e-9,
                "VTEC mis-set {misset_fraction:+}: residual {residual_m} m, \
                 expected exactly the mis-set fraction {expected_residual_m} m"
            );
            assert!(
                residual_m.abs() < true_delay_m,
                "mismatched correction must still beat no correction"
            );
        }
    }

    // Spec scenario: "Hand-integrated profile matches" — the bridge matches
    // an independently hand-computed integral in TECU.
    //
    // Hand computation: ∫₋∞^∞ N_max·exp(0.5(1 − z − e^(−z)))·H dz with the
    // substitution u = e^(−z) (dz = −du/u) becomes
    //   N_max·H·e^(1/2)·∫₀^∞ u^(−1/2)·e^(−u/2) du
    //     = N_max·H·e^(1/2)·√2·Γ(1/2) = N_max·H·√(2πe).
    // For N_max = 1e12 el/m³, hm = 300 km, H = 80 km (thresh-synth's OTHR
    // test F-layer parameters): VTEC = 1e12·8e4·√(2πe) ≈ 3.3062e17 el/m²
    // ≈ 33.06 TECU. Truncation of the below-ground tail (z < −3.75, double-
    // exponential decay) and the topside above hm+40H are each < 1e-8
    // relative; Simpson error at 4096 fixed intervals is smaller still.
    // Documented quadrature tolerance: 1e-7 relative.
    #[test]
    fn hand_integrated_profile_matches() {
        let n_max = 1.0e12; // el/m³
        let hm_m = 300_000.0;
        let scale_height_m = 80_000.0;
        let hand_vtec_tecu = n_max * scale_height_m * (2.0 * PI * E).sqrt() / TECU_ELECTRONS_PER_M2;
        let bridge_vtec_tecu = chapman_vtec_tecu(n_max, hm_m, scale_height_m);
        let relative_err = (bridge_vtec_tecu - hand_vtec_tecu).abs() / hand_vtec_tecu;
        assert!(
            relative_err < 1e-7,
            "bridge {bridge_vtec_tecu} TECU vs hand {hand_vtec_tecu} TECU \
             (relative error {relative_err})"
        );
        // Unit-slip guard: a km/m slip would shift the result by 1e3; the
        // correct answer is tens of TECU for these F-layer parameters.
        assert!(
            (10.0..100.0).contains(&bridge_vtec_tecu),
            "VTEC {bridge_vtec_tecu} TECU outside the physical tens-of-TECU \
             regime — check the el/m³ → TECU unit chain"
        );
    }

    /// The mirrored Chapman form keeps `thresh-synth`'s peak property:
    /// density at `hm` equals `N_max` (same check as the OTHR module's
    /// `chapman_peak_at_hm` test, in SI units).
    #[test]
    fn chapman_mirror_peaks_at_hm() {
        let n_max = 1.0e12;
        let hm_m = 300_000.0;
        let scale_height_m = 80_000.0;
        let at_peak = chapman_electron_density_el_per_m3(hm_m, n_max, hm_m, scale_height_m);
        assert!((at_peak - n_max).abs() / n_max < 1e-12);
        let above = chapman_electron_density_el_per_m3(hm_m + 1.0e5, n_max, hm_m, scale_height_m);
        let below = chapman_electron_density_el_per_m3(hm_m - 1.0e5, n_max, hm_m, scale_height_m);
        assert!(at_peak > above && at_peak > below);
    }

    /// Serde round trip; omitting `shell_height_m` takes the cited default.
    #[test]
    fn serde_default_shell_height() {
        let parsed: IonoDelayConfig =
            serde_json::from_str(r#"{"vtec_tecu": 10.0, "frequency_hz": 1.3e9}"#).unwrap();
        assert_eq!(
            parsed.shell_height_m.to_bits(),
            DEFAULT_SHELL_HEIGHT_M.to_bits()
        );
        let full = IonoDelayConfig::new(10.0, 1.3e9);
        let json = serde_json::to_string(&full).unwrap();
        let back: IonoDelayConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(back, full);
    }
}
