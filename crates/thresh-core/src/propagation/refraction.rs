//! Tropospheric microwave refraction: apparent-elevation bending and excess
//! range from a Bean–Dutton / CRPL exponential refractivity profile,
//! integrated along the spherically-stratified ray path, plus the classic
//! 4/3-Earth cheap tier (design Decision 2 of the
//! `atmospheric-measurement-propagation` change).
//!
//! # Model
//!
//! Refractivity decays exponentially with height above the surface,
//!
//! ```text
//! N(h) = N_s · e^(−h/H),      n(h) = 1 + N(h)·10⁻⁶
//! ```
//!
//! the CRPL exponential reference atmosphere ([`DEFAULT_SURFACE_REFRACTIVITY_N`]
//! `N_s = 313`, scale height `H = 1/`[`CRPL_DECAY_CONSTANT_PER_KM`] ≈ 6.95 km).
//! A radio ray from a ground station to an elevated target bends toward the
//! Earth and travels a longer, slower optical path. Both effects are found by
//! a deterministic fixed-count Simpson quadrature of the standard
//! spherically-stratified ray equations (Bouguer's rule / Snell's law in polar
//! coordinates, `n·r·cos E = const`; ITU-R P.834-8 §1, §4, Eq. 5–8, fetched
//! 2026-07-16, <https://www.itu.int/rec/R-REC-P.834/en>):
//!
//! ```text
//! bending  τ = −∫ (n′/n)·cot φ dh        (ray-turning; ITU-R P.834-8 Eq. 5)
//! excess   Δr = ∫ (n−1)/sin φ dh + (∫ ds − chord)   (delay + bent-path;
//!                                          ITU-R P.834-8 Eq. 15)
//! cos φ = c / (n·(Rₑ+h)),   c = n(h_s)·(Rₑ+h_s)·cos E    (Eq. 6–7)
//! ```
//!
//! # Conventions and scope
//!
//! - Per the module conventions ([`crate::propagation`]) the **apparent**
//!   (measured) elevation and range equal the **true** geometric values plus
//!   the modeled bias — refraction raises apparent elevation (the ray arrives
//!   from a steeper angle than the straight line) and lengthens range;
//!   [`correct_refraction`] subtracts the modeled bias.
//! - The reported **bending** is the total ray-turning τ. For a target at or
//!   above the top of the modeled atmosphere ([`ATMOSPHERE_TOP_M`]) this equals
//!   the apparent-minus-true elevation and reproduces the tabulated total
//!   refraction (ITU-R P.834-8 Eq. 9 / Table 1). For a target *embedded* in the
//!   troposphere the true apparent-minus-true elevation is smaller than τ (the
//!   tangent-vs-chord factor, up to ≈2× at high elevation); modeling that
//!   exactly is a two-point boundary-value ray solve, deferred — the forward
//!   model applies τ as the (conservative, upper-bound) bias, and the inverse
//!   [`correct_refraction`] inverts whatever forward bias was applied, so
//!   round-trip fidelity is unaffected.
//! - **Validity floor**: below [`RefractionConfig::min_elevation_rad`]
//!   (default 1°) the elevation is clamped and the result equals the floor
//!   evaluation. Low-elevation ducting / super-refraction is out of scope;
//!   the CRPL model itself is documented for elevations above ≈10 mrad
//!   (0.573°) (MathWorks `refractionexp`, fetched 2026-07-16), so the 1° floor
//!   is conservative.
//! - Atmospheric **attenuation** (SNR / P_d, ITU-R P.676) is a different,
//!   amplitude-domain effect and is not this module's concern.

use serde::{Deserialize, Serialize};

/// CRPL exponential reference atmosphere surface refractivity `N_s` (N-units).
///
/// Default surface value of the Central Radio Propagation Laboratory
/// exponential reference atmosphere. Source: Bean & Thayer, "CRPL Exponential
/// Reference Atmosphere," *J. Res. NBS* 63D(3), 1959; surfaced as the
/// `refractionexp` default `Ns = 313` (MathWorks,
/// <https://www.mathworks.com/help/radar/ref/refractionexp.html>, fetched
/// 2026-07-16).
pub const DEFAULT_SURFACE_REFRACTIVITY_N: f64 = 313.0;

/// CRPL exponential decay constant `R_exp` (km⁻¹) for `N(h) = N_s·e^(−R_exp·h)`.
///
/// Default decay constant of the CRPL exponential reference atmosphere for
/// `N_s = 313`; the scale height used here is its reciprocal
/// (`H = 1/R_exp ≈ 6.951 km`). Source: MathWorks `refractionexp` default
/// `Rexp = 0.143859` km⁻¹
/// (<https://www.mathworks.com/help/radar/ref/refractionexp.html>, fetched
/// 2026-07-16; Bean & Thayer 1959, Dutton & Thayer NBS TN 97).
pub const CRPL_DECAY_CONSTANT_PER_KM: f64 = 0.143859;

/// Default exponential scale height `H` (m) — reciprocal of the CRPL decay
/// constant, `H = 1000/`[`CRPL_DECAY_CONSTANT_PER_KM`] ≈ 6951.25 m.
pub const DEFAULT_SCALE_HEIGHT_M: f64 = 1000.0 / CRPL_DECAY_CONSTANT_PER_KM;

/// ITU-R P.834 reference-atmosphere surface refractivity `N_s` (N-units).
///
/// The exponential terrestrial atmosphere of ITU-R P.834-8 Eq. (8),
/// `n(x) = 1 + a·e^(−b·x)` with `a = 0.000315` (i.e. `N_s = 315`). Used by
/// [`RefractionConfig::itu_p834_reference`] to reproduce the Recommendation's
/// worked bending values (Eq. 9). Source: ITU-R P.834-8 §4.2 (fetched
/// 2026-07-16).
pub const ITU_SURFACE_REFRACTIVITY_N: f64 = 315.0;

/// ITU-R P.834 reference-atmosphere decay constant `b` (km⁻¹).
///
/// From ITU-R P.834-8 Eq. (8) (`b = 0.1361`; scale height `1/b ≈ 7.348 km`).
/// Source: ITU-R P.834-8 §4.2 (fetched 2026-07-16).
pub const ITU_DECAY_CONSTANT_PER_KM: f64 = 0.1361;

/// Spherical mean Earth radius (m) for the stratified-atmosphere geometry.
///
/// The stratified ray equations use the spherical mean radius, not the WGS-84
/// ellipsoid. 6371 km is the conventional value (matches the OTHR/ionospheric
/// sibling [`crate::propagation::iono_delay::MEAN_EARTH_RADIUS_M`]); ITU-R
/// P.834-8 Eq. (7) uses 6370 km, a <0.02% difference that is negligible
/// against the documented quadrature tolerance.
pub const MEAN_EARTH_RADIUS_M: f64 = 6_371_000.0;

/// Top of the modeled atmosphere (m): refractivity is negligible above this,
/// so the ray integral is capped here regardless of target altitude.
///
/// `e^(−80/6.95) ≈ 1e-5`, so truncating at 80 km alters the zenith excess by
/// `< 3e-5` m; capping also keeps the fixed-count Simpson step small
/// (≤ 80 km / [`RAY_QUADRATURE_INTERVALS`] ≈ 78 m) and target-height
/// independent, resolving the exponentially-concentrated integrand.
pub const ATMOSPHERE_TOP_M: f64 = 80_000.0;

/// Fixed composite-Simpson interval count for the ray integral (even).
///
/// Resolution of the design open question (fixed count vs fixed step): a
/// **fixed count** over the capped `[h_s, `[`ATMOSPHERE_TOP_M`]`]` domain
/// reproduces ITU-R P.834-8 Eq. (9) within 6% (3°–10°) and the zenith excess
/// to `< 0.1` mm, converged to `~1e-6` relative by 512 intervals; a fixed
/// step-length would scale the count and cost with the integration span
/// without accuracy benefit on this bounded, exponentially-concentrated
/// integrand. Fixed count gives deterministic `O(1)` cost and a
/// bounded-complexity loop (mirrors the ionospheric sibling's
/// `CHAPMAN_SIMPSON_INTERVALS`).
pub const RAY_QUADRATURE_INTERVALS: usize = 1024;

/// Effective-Earth k-factor for the 4/3 cheap tier.
///
/// ITU-R P.834-8 §2: "for heights below 1000 m the exponential model … can be
/// approximated by a linear one. The corresponding k-factor is k = 4/3"
/// (fetched 2026-07-16).
pub const EFFECTIVE_EARTH_K: f64 = 4.0 / 3.0;

/// Default validity-floor elevation (degrees); see
/// [`RefractionConfig::min_elevation_rad`].
pub const DEFAULT_MIN_ELEVATION_DEG: f64 = 1.0;

/// Configuration for the exponential-profile refraction model.
///
/// Pure data: surface refractivity `N_s` (N-units), exponential scale height
/// `H` (m), and the validity-floor elevation (rad). Serde-defaulted to the
/// CRPL exponential reference atmosphere ([`RefractionConfig::crpl`]).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct RefractionConfig {
    /// Surface refractivity `N_s` (N-units; `n(0) = 1 + N_s·10⁻⁶`).
    #[serde(default = "default_surface_refractivity_n")]
    pub n_s: f64,
    /// Exponential scale height `H` (m) in `N(h) = N_s·e^(−h/H)`.
    #[serde(default = "default_scale_height_m")]
    pub scale_height_m: f64,
    /// Minimum valid elevation (rad); below it evaluation clamps to this floor.
    #[serde(default = "default_min_elevation_rad")]
    pub min_elevation_rad: f64,
}

impl RefractionConfig {
    /// CRPL exponential reference atmosphere: `N_s = 313`, `H ≈ 6.951 km`,
    /// 1° floor (the default).
    pub fn crpl() -> Self {
        Self {
            n_s: DEFAULT_SURFACE_REFRACTIVITY_N,
            scale_height_m: DEFAULT_SCALE_HEIGHT_M,
            min_elevation_rad: default_min_elevation_rad(),
        }
    }

    /// Config with explicit `N_s` and scale height, default 1° floor.
    pub fn new(n_s: f64, scale_height_m: f64) -> Self {
        Self {
            n_s,
            scale_height_m,
            min_elevation_rad: default_min_elevation_rad(),
        }
    }

    /// ITU-R P.834-8 reference atmosphere (`N_s = 315`, `H = 1/b ≈ 7.348 km`,
    /// Eq. 8), for reproducing the Recommendation's worked bending values
    /// (Eq. 9).
    pub fn itu_p834_reference() -> Self {
        Self {
            n_s: ITU_SURFACE_REFRACTIVITY_N,
            scale_height_m: 1000.0 / ITU_DECAY_CONSTANT_PER_KM,
            min_elevation_rad: default_min_elevation_rad(),
        }
    }

    /// Panic with a clear message on physically invalid parameters —
    /// serde-deserialized configs are not otherwise validated, and a
    /// negative or non-finite refractivity/scale height silently produces
    /// NaN or wrong-signed biases. Checked once at every public
    /// evaluation entry point (the `DpConfig`/`ForceModelConfig` pattern).
    pub(crate) fn validate(&self) {
        assert!(
            self.n_s.is_finite() && self.n_s >= 0.0,
            "surface refractivity N_s must be finite and >= 0, got {}",
            self.n_s
        );
        assert!(
            self.scale_height_m.is_finite() && self.scale_height_m > 0.0,
            "scale height must be finite and > 0, got {}",
            self.scale_height_m
        );
        assert!(
            self.min_elevation_rad > 0.0 && self.min_elevation_rad <= std::f64::consts::FRAC_PI_2,
            "minimum elevation must be in (0, pi/2], got {}",
            self.min_elevation_rad
        );
    }
}

impl Default for RefractionConfig {
    fn default() -> Self {
        Self::crpl()
    }
}

fn default_surface_refractivity_n() -> f64 {
    DEFAULT_SURFACE_REFRACTIVITY_N
}
fn default_scale_height_m() -> f64 {
    DEFAULT_SCALE_HEIGHT_M
}
fn default_min_elevation_rad() -> f64 {
    DEFAULT_MIN_ELEVATION_DEG.to_radians()
}

/// Accumulated ray-path integrals over `[h_s, min(h_t, ATMOSPHERE_TOP_M)]`.
#[derive(Debug, Clone, Copy, Default)]
struct RayIntegrals {
    /// Total ray-turning τ (rad; ITU-R P.834-8 Eq. 5).
    bending_rad: f64,
    /// Refractive delay `∫(n−1)/sin φ dh` (m; ITU-R P.834-8 Eq. 15).
    excess_delay_m: f64,
    /// Geocentric angle swept by the ray (rad).
    geocentric_angle_rad: f64,
    /// Geometric arc length `∫ds` (m).
    arc_length_m: f64,
}

/// Refractive index `n(h) = 1 + N_s·e^(−h/H)·10⁻⁶` at height `h` (m).
fn refractive_index(config: &RefractionConfig, height_m: f64) -> f64 {
    1.0 + config.n_s * (-height_m / config.scale_height_m).exp() * 1e-6
}

/// The four ray integrands at height `h` for Bouguer invariant `c`.
///
/// Returns `(bending, delay, dθ/dh, ds/dh)`. `cos φ = c/(n·r)` (ITU-R P.834-8
/// Eq. 6), clamped to `≤ 1` against rounding; `sin φ ≥ sin(floor) > 0` for the
/// clamped elevations this module evaluates, so no integrand is singular.
fn ray_sample(config: &RefractionConfig, c: f64, height_m: f64) -> (f64, f64, f64, f64) {
    let r = MEAN_EARTH_RADIUS_M + height_m;
    let decay = (-height_m / config.scale_height_m).exp();
    let n = 1.0 + config.n_s * decay * 1e-6;
    // −dn/dh = (N_s/H)·e^(−h/H)·1e-6  > 0 (refractivity falls with height).
    let neg_dndh = (config.n_s / config.scale_height_m) * decay * 1e-6;
    let cos_phi = (c / (n * r)).min(1.0);
    let sin_phi = (1.0 - cos_phi * cos_phi).max(0.0).sqrt();
    let bending = neg_dndh * (cos_phi / sin_phi) / n;
    let delay = (n - 1.0) / sin_phi;
    let dtheta = cos_phi / (r * sin_phi);
    let arc = 1.0 / sin_phi;
    (bending, delay, dtheta, arc)
}

/// Fixed-count composite-Simpson integration of the ray equations from the
/// station to `min(target_alt_m, ATMOSPHERE_TOP_M)`.
///
/// Precondition: `elevation_rad` is already clamped to the validity floor.
/// Returns zero integrals when the capped span is non-positive (target at or
/// below the station / atmosphere top).
fn integrate_ray(
    config: &RefractionConfig,
    elevation_rad: f64,
    station_alt_m: f64,
    target_alt_m: f64,
) -> RayIntegrals {
    let top = target_alt_m.min(ATMOSPHERE_TOP_M);
    if top <= station_alt_m {
        return RayIntegrals::default();
    }
    let r_s = MEAN_EARTH_RADIUS_M + station_alt_m;
    let c = refractive_index(config, station_alt_m) * r_s * elevation_rad.cos();
    let n = RAY_QUADRATURE_INTERVALS;
    let step = (top - station_alt_m) / n as f64;

    let (mut b, mut d, mut t, mut a) = ray_sample(config, c, station_alt_m);
    let end = ray_sample(config, c, top);
    b += end.0;
    d += end.1;
    t += end.2;
    a += end.3;
    for i in 1..n {
        let weight = if i % 2 == 1 { 4.0 } else { 2.0 };
        let s = ray_sample(config, c, station_alt_m + step * i as f64);
        b += weight * s.0;
        d += weight * s.1;
        t += weight * s.2;
        a += weight * s.3;
    }
    let scale = step / 3.0;
    RayIntegrals {
        bending_rad: b * scale,
        excess_delay_m: d * scale,
        geocentric_angle_rad: t * scale,
        arc_length_m: a * scale,
    }
}

/// Total ray-turning bending (rad) and excess range (m) for a station
/// observing a target, from the exponential profile.
///
/// The elevation is clamped to [`RefractionConfig::min_elevation_rad`] (the
/// validity floor). Bending is the ray-turning integral (ITU-R P.834-8 Eq. 5;
/// see the module docs on the embedded-target vs space-target regimes). Excess
/// range is the refractive delay `∫(n−1)/sin φ dh` (ITU-R P.834-8 Eq. 15) plus
/// the bent-path geometric excess `∫ds − chord` (second-order; a few cm above
/// ~10° elevation, growing toward the horizon). Both increase monotonically as
/// elevation decreases. Deterministic: identical inputs yield bitwise-identical
/// output.
///
/// When the capped integration span is non-positive — the target at or below
/// the station altitude, or the station above [`ATMOSPHERE_TOP_M`] — the ray
/// integral is empty and the result is exactly `(0.0, 0.0)`: no modeled
/// refraction, never a negative bias (per the module convention, refraction
/// only raises apparent elevation and lengthens range).
pub fn refraction_bending_and_excess(
    config: &RefractionConfig,
    elevation_rad: f64,
    station_alt_m: f64,
    target_alt_m: f64,
) -> (f64, f64) {
    config.validate();
    if target_alt_m.min(ATMOSPHERE_TOP_M) <= station_alt_m {
        return (0.0, 0.0);
    }
    let elevation = elevation_rad.max(config.min_elevation_rad);
    let ray = integrate_ray(config, elevation, station_alt_m, target_alt_m);
    let r_s = MEAN_EARTH_RADIUS_M + station_alt_m;
    let r_t = MEAN_EARTH_RADIUS_M + target_alt_m.min(ATMOSPHERE_TOP_M);
    let chord = (r_s * r_s + r_t * r_t - 2.0 * r_s * r_t * ray.geocentric_angle_rad.cos())
        .max(0.0)
        .sqrt();
    let excess = ray.excess_delay_m + (ray.arc_length_m - chord);
    (ray.bending_rad, excess)
}

/// Apply the modeled refraction bias to a true (geometric) measurement.
///
/// Returns `(apparent_elevation_rad, apparent_range_m)` = true values plus the
/// modeled bending and excess range (bias-before-noise; see
/// [`crate::propagation`]). Inverse: [`correct_refraction`].
pub fn apply_refraction(
    config: &RefractionConfig,
    true_elevation_rad: f64,
    true_range_m: f64,
    station_alt_m: f64,
    target_alt_m: f64,
) -> (f64, f64) {
    let (bending, excess) =
        refraction_bending_and_excess(config, true_elevation_rad, station_alt_m, target_alt_m);
    (true_elevation_rad + bending, true_range_m + excess)
}

/// Remove the modeled refraction bias from an apparent (measured) measurement.
///
/// One documented fixed-point step: the modeled bias is evaluated at the
/// *apparent* elevation (the true elevation is unknown) and subtracted —
/// `corrected = apparent − bias(apparent)`. Because the bias varies with
/// elevation, one step leaves a small residual that shrinks rapidly away from
/// the horizon (< 0.5″ in elevation and < 0.02 m in range for elevations
/// ≥ 20°); a bias-then-correct round trip with the true parameters recovers the
/// geometry to this quadrature/fixed-point tolerance. Inverse of
/// [`apply_refraction`].
pub fn correct_refraction(
    config: &RefractionConfig,
    apparent_elevation_rad: f64,
    apparent_range_m: f64,
    station_alt_m: f64,
    target_alt_m: f64,
) -> (f64, f64) {
    let (bending, excess) =
        refraction_bending_and_excess(config, apparent_elevation_rad, station_alt_m, target_alt_m);
    (apparent_elevation_rad - bending, apparent_range_m - excess)
}

/// Cheap 4/3-Earth tier: ray-turning bending (rad) from the classic
/// effective-radius model.
///
/// The linear-gradient (4/3-Earth) model fixes the refractivity gradient at
/// `|dn/dh| = (k−1)/(k·Rₑ)` ([`EFFECTIVE_EARTH_K`]); the physical ray
/// curvature it implies is `|dn/dh|·cos φ` (the horizontal-ray curvature
/// `(k−1)/(k·Rₑ)` times the obliquity factor `cos φ` — the constant form
/// without `cos φ` is only the near-horizontal limit). Integrated through
/// straight-line (`n ≈ 1`) geometry:
/// `τ = ∫ (k−1)/(k·Rₑ) · (cos φ/sin φ) dh` with
/// `cos φ = (Rₑ+h_s)·cos E/(Rₑ+h)`. Cross-checks the authoritative exponential
/// tier ([`refraction_bending_and_excess`]) at moderate elevations for a
/// low-altitude target, the regime where ITU-R P.834-8 §2 states the 4/3
/// approximation holds (heights below ~1 km). Elevation is clamped to the
/// validity floor.
pub fn effective_earth_bending(
    config: &RefractionConfig,
    elevation_rad: f64,
    station_alt_m: f64,
    target_alt_m: f64,
) -> f64 {
    config.validate();
    let elevation = elevation_rad.max(config.min_elevation_rad);
    let top = target_alt_m.min(ATMOSPHERE_TOP_M);
    if top <= station_alt_m {
        return 0.0;
    }
    let r_s = MEAN_EARTH_RADIUS_M + station_alt_m;
    let cos_e = elevation.cos();
    let inv_rho = (EFFECTIVE_EARTH_K - 1.0) / (EFFECTIVE_EARTH_K * MEAN_EARTH_RADIUS_M);
    let n = RAY_QUADRATURE_INTERVALS;
    let step = (top - station_alt_m) / n as f64;
    let sample = |height_m: f64| {
        let cos_phi = (r_s * cos_e / (MEAN_EARTH_RADIUS_M + height_m)).min(1.0);
        let sin_phi = (1.0 - cos_phi * cos_phi).max(0.0).sqrt();
        // Ray curvature is |dn/dh|·cos φ (obliquity factor); dh = ds·sin φ.
        inv_rho * cos_phi / sin_phi
    };
    let mut sum = sample(station_alt_m) + sample(top);
    for i in 1..n {
        let weight = if i % 2 == 1 { 4.0 } else { 2.0 };
        sum += weight * sample(station_alt_m + step * i as f64);
    }
    sum * step / 3.0
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    fn deg(rad: f64) -> f64 {
        rad.to_degrees()
    }
    fn rad(deg: f64) -> f64 {
        deg * PI / 180.0
    }

    // ── ITU-R P.834-8 Eq. (9): closed-form total refraction τ(h,θ) in degrees,
    // for h in km, θ in degrees. Authoritative for θ ≤ 10° (derived for that
    // range; §4.2, fetched 2026-07-16). Coefficients transcribed from the
    // fetched Recommendation and guarded by `constant_provenance_tripwire`.
    const EQ9: [f64; 7] = [1.314, 0.6437, 0.02869, 0.2305, 0.09428, 0.01096, 0.008583];
    fn itu_eq9_deg(h_km: f64, theta_deg: f64) -> f64 {
        let t = theta_deg;
        1.0 / (EQ9[0]
            + EQ9[1] * t
            + EQ9[2] * t * t
            + h_km * (EQ9[3] + EQ9[4] * t + EQ9[5] * t * t)
            + EQ9[6] * h_km * h_km)
    }

    /// Tripwire: cited constants must not drift from their fetched sources
    /// (module-doc / test citations, all fetched 2026-07-16). Any edit to these
    /// literals must re-fetch and re-cite.
    #[test]
    fn constant_provenance_tripwire() {
        assert_eq!(
            DEFAULT_SURFACE_REFRACTIVITY_N.to_bits(),
            313.0_f64.to_bits()
        );
        assert_eq!(CRPL_DECAY_CONSTANT_PER_KM.to_bits(), 0.143859_f64.to_bits());
        assert_eq!(ITU_SURFACE_REFRACTIVITY_N.to_bits(), 315.0_f64.to_bits());
        assert_eq!(ITU_DECAY_CONSTANT_PER_KM.to_bits(), 0.1361_f64.to_bits());
        assert_eq!(MEAN_EARTH_RADIUS_M.to_bits(), 6_371_000.0_f64.to_bits());
        assert_eq!(ATMOSPHERE_TOP_M.to_bits(), 80_000.0_f64.to_bits());
        assert_eq!(EFFECTIVE_EARTH_K.to_bits(), (4.0_f64 / 3.0).to_bits());
        // ITU-R P.834-8 Eq. (9) coefficients.
        let eq9_src = [
            1.314_f64, 0.6437, 0.02869, 0.2305, 0.09428, 0.01096, 0.008583,
        ];
        for (a, b) in EQ9.iter().zip(eq9_src.iter()) {
            assert_eq!(a.to_bits(), b.to_bits());
        }
        // Scale height is the reciprocal of the CRPL decay constant.
        assert_eq!(
            DEFAULT_SCALE_HEIGHT_M.to_bits(),
            (1000.0_f64 / 0.143859).to_bits()
        );
    }

    // Spec scenario: "Published worked values reproduced" — the ray-turning
    // bending reproduces ITU-R P.834-8 Eq. (9) for a space target (above the
    // modeled atmosphere), evaluated in the ITU reference atmosphere. Recorded
    // ratios (computed/Eq.9): 3°→0.969, 5°→0.978, 7°→1.002, 10°→1.054 — within
    // 6% over 3°–10°, the elevation band where Eq. (9) is authoritative (the
    // Recommendation notes it is derived for θ ≤ 10°; above that Eq. (9) is a
    // known-loose extrapolation and diverges from the integral by design).
    #[test]
    fn published_worked_values_bending() {
        let cfg = RefractionConfig::itu_p834_reference();
        // Target above the atmosphere: essentially the space-station regime.
        let target = 1.0e9;
        for &theta_deg in &[3.0, 5.0, 7.0, 10.0] {
            let (bending, _) = refraction_bending_and_excess(&cfg, rad(theta_deg), 0.0, target);
            let eq9_deg = itu_eq9_deg(0.0, theta_deg);
            let ratio = deg(bending) / eq9_deg;
            assert!(
                (0.94..=1.06).contains(&ratio),
                "θ={theta_deg}°: bending {:.5}° vs ITU Eq.9 {eq9_deg:.5}° (ratio {ratio:.4})",
                deg(bending)
            );
        }
    }

    // Spec scenario: "Published worked values reproduced" (range side) — the
    // zenith excess range equals the analytic column delay N_s·H·1e-6 (ITU-R
    // P.834-8 Eq. 20) to quadrature precision, and matches the canonical ~2.3 m
    // zenith tropospheric (hydrostatic) delay; at 45° the excess reproduces the
    // ITU-R P.834-8 Eq. (16) mapping (≈3.27 m). Recorded (ITU atmosphere):
    // zenith 2.3144 m, 45° 3.270 m.
    #[test]
    fn published_worked_values_excess_range() {
        let cfg = RefractionConfig::itu_p834_reference();
        let analytic_zenith = cfg.n_s * cfg.scale_height_m * 1e-6;
        let (_, zenith) = refraction_bending_and_excess(&cfg, rad(90.0), 0.0, 1.0e9);
        assert!(
            (zenith - analytic_zenith).abs() < 1e-3,
            "zenith excess {zenith:.6} m vs analytic N_s·H·1e-6 {analytic_zenith:.6} m"
        );
        assert!(
            (2.2..2.4).contains(&zenith),
            "zenith excess {zenith:.4} m outside the canonical ~2.3 m regime"
        );
        let (_, excess_45) = refraction_bending_and_excess(&cfg, rad(45.0), 0.0, 1.0e9);
        // ITU-R P.834-8 Eq. (16): ΔL_V/(sin θ·√(1+k·cot²θ)) ≈ 3.27 m at 45°.
        assert!(
            (excess_45 - 3.270).abs() < 0.05,
            "45° excess {excess_45:.4} m vs ITU Eq.16 ≈3.270 m"
        );
    }

    // Spec scenario: "Bending grows toward the horizon" — bending and excess
    // range both increase monotonically as elevation decreases from 45° to the
    // validity floor.
    #[test]
    fn bending_grows_toward_the_horizon() {
        let cfg = RefractionConfig::crpl();
        let mut prev_bending = 0.0;
        let mut prev_excess = 0.0;
        for &theta_deg in &[45.0, 30.0, 20.0, 15.0, 10.0, 7.0, 5.0, 3.0, 2.0, 1.0] {
            let (bending, excess) = refraction_bending_and_excess(&cfg, rad(theta_deg), 0.0, 1.0e9);
            assert!(
                bending > prev_bending,
                "bending not monotonic at {theta_deg}°: {bending} <= {prev_bending}"
            );
            assert!(
                excess > prev_excess,
                "excess not monotonic at {theta_deg}°: {excess} <= {prev_excess}"
            );
            prev_bending = bending;
            prev_excess = excess;
        }
    }

    // Spec scenario: "Tiers agree at moderate elevation" — the 4/3-Earth cheap
    // tier and the authoritative exponential tier agree (as ray-turning) for a
    // low-altitude target (h_t = 1 km, the regime where ITU-R P.834-8 §2 states
    // the 4/3 approximation holds). Both integrands carry the same cos φ
    // obliquity factor on the ray curvature, so the ratio is flat across
    // elevation. Recorded ratios (4/3 / exponential): 5°→0.9332, 10°→0.9352,
    // 15°→0.9356, 20°→0.9357, 30°→0.9358, 45°→0.9359. The residual is the
    // gradient difference: the 4/3 model's constant |dn/dh| = (k−1)/(k·Rₑ)
    // ≈ −39.2 N/km vs the CRPL exponential's mean gradient over the 0–1 km
    // path ≈ −41.9 N/km, giving 39.2/41.9 ≈ 0.936. Documented cross-check
    // band: [0.90, 0.98] over 5°–45°.
    #[test]
    fn tiers_agree_at_moderate_elevation() {
        let cfg = RefractionConfig::crpl();
        let target = 1000.0;
        for &theta_deg in &[5.0, 10.0, 15.0, 20.0, 30.0, 45.0] {
            let (exp_bending, _) = refraction_bending_and_excess(&cfg, rad(theta_deg), 0.0, target);
            let cheap = effective_earth_bending(&cfg, rad(theta_deg), 0.0, target);
            let ratio = cheap / exp_bending;
            assert!(
                (0.90..=0.98).contains(&ratio),
                "θ={theta_deg}°: 4/3 {} vs exp {} (ratio {ratio:.4}) outside band",
                deg(cheap) * 3600.0,
                deg(exp_bending) * 3600.0
            );
        }
    }

    // Spec scenario: "Below-floor evaluation clamps" — evaluating below the 1°
    // validity floor returns exactly the floor evaluation (bitwise), and the
    // API documents this behavior.
    #[test]
    fn below_floor_evaluation_clamps() {
        let cfg = RefractionConfig::crpl();
        let (bend_floor, exc_floor) = refraction_bending_and_excess(&cfg, rad(1.0), 0.0, 1.0e9);
        for &theta_deg in &[0.9, 0.5, 0.1, -1.0] {
            let (bend, exc) = refraction_bending_and_excess(&cfg, rad(theta_deg), 0.0, 1.0e9);
            assert_eq!(
                bend.to_bits(),
                bend_floor.to_bits(),
                "bending below floor at {theta_deg}° must equal the 1° evaluation"
            );
            assert_eq!(exc.to_bits(), exc_floor.to_bits());
        }
        // Effective-earth tier clamps identically.
        let cheap_floor = effective_earth_bending(&cfg, rad(1.0), 0.0, 1000.0);
        let cheap_below = effective_earth_bending(&cfg, rad(0.3), 0.0, 1000.0);
        assert_eq!(cheap_below.to_bits(), cheap_floor.to_bits());
    }

    // Spec requirement: repeated evaluation is bitwise deterministic (fixed
    // quadrature, no state).
    #[test]
    fn deterministic_repeated_evaluation() {
        let cfg = RefractionConfig::crpl();
        let a = refraction_bending_and_excess(&cfg, rad(4.2), 12.0, 42_000.0);
        let b = refraction_bending_and_excess(&cfg, rad(4.2), 12.0, 42_000.0);
        assert_eq!(a.0.to_bits(), b.0.to_bits());
        assert_eq!(a.1.to_bits(), b.1.to_bits());
        let ap = apply_refraction(&cfg, rad(4.2), 300_000.0, 12.0, 42_000.0);
        let ap2 = apply_refraction(&cfg, rad(4.2), 300_000.0, 12.0, 42_000.0);
        assert_eq!(ap.0.to_bits(), ap2.0.to_bits());
        assert_eq!(ap.1.to_bits(), ap2.1.to_bits());
    }

    // Spec scenario: "Round trip with true parameters" — biasing then
    // correcting with the same parameters recovers the geometry to the
    // documented quadrature/fixed-point tolerance. Recorded residuals: 20°
    // → 0.46″ elevation, 0.015 m range; 45° → 0.04″, 0.001 m (the one-step
    // fixed-point residual shrinks rapidly away from the horizon).
    #[test]
    fn round_trip_with_true_parameters() {
        let cfg = RefractionConfig::crpl();
        let true_range = 150_000.0;
        for &theta_deg in &[20.0, 30.0, 45.0] {
            let true_el = rad(theta_deg);
            let (app_el, app_range) = apply_refraction(&cfg, true_el, true_range, 0.0, 1.0e9);
            let (corr_el, corr_range) = correct_refraction(&cfg, app_el, app_range, 0.0, 1.0e9);
            let el_resid_arcsec = deg(corr_el - true_el).abs() * 3600.0;
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

    // Spec scenario: "Mismatched correction still helps" — the bias is very
    // nearly linear in N_s, so correcting with the surface refractivity
    // mis-set by ±10% leaves a residual of approximately the mis-set fraction
    // (≈10%) of the uncorrected bias: correcting removes ~90% of the bias, an
    // order of magnitude smaller than leaving it uncorrected. Recorded (CRPL,
    // θ=20°, space target): uncorrected bias ≈175.97″ / 6.321 m; +10% N_s
    // leaves ≈0.097×, −10% leaves ≈0.103×. The assert pins the residual ratio
    // to the [0.05, 0.13] band around the mis-set fraction — honest in both
    // directions (the correction cannot do better than the parameter error,
    // and must not do worse).
    #[test]
    fn mismatched_correction_still_helps() {
        let true_cfg = RefractionConfig::crpl();
        let true_el = rad(20.0);
        let true_range = 150_000.0;
        let (app_el, app_range) = apply_refraction(&true_cfg, true_el, true_range, 0.0, 1.0e9);
        let (uncorr_bend, uncorr_excess) =
            refraction_bending_and_excess(&true_cfg, true_el, 0.0, 1.0e9);

        for &frac in &[0.10, -0.10] {
            let misset =
                RefractionConfig::new(true_cfg.n_s * (1.0 + frac), true_cfg.scale_height_m);
            let (corr_el, corr_range) = correct_refraction(&misset, app_el, app_range, 0.0, 1.0e9);
            let el_ratio = (corr_el - true_el).abs() / uncorr_bend;
            let range_ratio = (corr_range - true_range).abs() / uncorr_excess;
            assert!(
                (0.05..0.13).contains(&el_ratio),
                "N_s{frac:+}: elevation residual ratio {el_ratio:.4} not ≈ the mis-set fraction"
            );
            assert!(
                (0.05..0.13).contains(&range_ratio),
                "N_s{frac:+}: range residual ratio {range_ratio:.4} not ≈ the mis-set fraction"
            );
            // The residual must beat leaving the bias uncorrected outright.
            assert!(el_ratio < 1.0 && range_ratio < 1.0);
        }
    }

    // Regression (adversarial-review finding): a non-positive integration span
    // — target at or below an elevated station, or station above the modeled
    // atmosphere — must yield exactly zero bending and zero excess, never a
    // negative excess. (Previously the zeroed geocentric angle made
    // `excess = 0 + (0 − |r_s − r_t|)`: station 2000 m over target 1000 m
    // produced excess = −1000 m, violating the module convention that
    // refraction only lengthens range.)
    #[test]
    fn target_below_station_yields_zero_bias() {
        let cfg = RefractionConfig::crpl();
        // Elevated radar looking down at a lower target: exactly zero bias.
        let (bending, excess) = refraction_bending_and_excess(&cfg, rad(5.0), 2000.0, 1000.0);
        assert_eq!(bending.to_bits(), 0.0_f64.to_bits());
        assert_eq!(excess.to_bits(), 0.0_f64.to_bits());
        // `apply_refraction` is then the identity on the geometry.
        let (app_el, app_range) = apply_refraction(&cfg, rad(5.0), 40_000.0, 2000.0, 1000.0);
        assert_eq!(app_el.to_bits(), rad(5.0).to_bits());
        assert_eq!(app_range.to_bits(), 40_000.0_f64.to_bits());
        // Sweep of station/target altitude orderings (below, equal, above,
        // and station above the modeled atmosphere): bending and excess are
        // never negative.
        for &station in &[0.0, 500.0, 1000.0, 2000.0, 5000.0, 90_000.0] {
            for &target in &[0.0, 500.0, 1000.0, 2000.0, 5000.0, 42_000.0, 1.0e9] {
                for &theta_deg in &[1.0, 5.0, 20.0, 60.0] {
                    let (b, e) =
                        refraction_bending_and_excess(&cfg, rad(theta_deg), station, target);
                    assert!(
                        b >= 0.0 && e >= 0.0,
                        "station {station} m, target {target} m, θ={theta_deg}°: \
                         bending {b} / excess {e} must be non-negative"
                    );
                }
            }
        }
    }

    /// Serde round trip; omitting fields takes the CRPL defaults.
    #[test]
    fn serde_defaults_to_crpl() {
        let parsed: RefractionConfig = serde_json::from_str("{}").unwrap();
        assert_eq!(parsed, RefractionConfig::crpl());
        let full = RefractionConfig::itu_p834_reference();
        let json = serde_json::to_string(&full).unwrap();
        let back: RefractionConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(back, full);
    }
    /// Serde-boundary validation: physically invalid parameters panic
    /// loudly at the first evaluation instead of producing NaN or
    /// wrong-signed biases.
    #[test]
    #[should_panic(expected = "surface refractivity")]
    fn negative_refractivity_panics() {
        let cfg = RefractionConfig::new(-1.0, 7000.0);
        refraction_bending_and_excess(&cfg, 0.2, 0.0, 10_000.0);
    }

    #[test]
    #[should_panic(expected = "scale height")]
    fn non_finite_scale_height_panics() {
        let cfg = RefractionConfig::new(313.0, f64::NAN);
        refraction_bending_and_excess(&cfg, 0.2, 0.0, 10_000.0);
    }
}
