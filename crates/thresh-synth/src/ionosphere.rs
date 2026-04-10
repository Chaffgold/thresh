//! Ionospheric propagation model for over-the-horizon radar (OTHR).
//!
//! Implements Chapman-layer electron density profiles, diurnal critical-frequency
//! variation, MUF/skip-zone calculations, virtual reflection heights, ionospheric
//! sounder simulation, and oblique ionogram generation.

use serde::{Deserialize, Serialize};

/// Earth radius in km (WGS-84 mean).
const EARTH_RADIUS_KM: f64 = 6371.0;

/// Ionospheric layer parameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IonosphereParams {
    /// F-layer critical frequency (MHz).
    pub fo_f2_mhz: f64,
    /// F-layer peak height (km).
    pub hm_f2_km: f64,
    /// F-layer scale height (km).
    pub scale_height_km: f64,
    /// E-layer critical frequency (MHz).
    pub fo_e_mhz: f64,
    /// E-layer height (km).
    pub hm_e_km: f64,
}

/// Ionospheric layer selector.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Layer {
    /// E-layer (~110 km).
    E,
    /// F-layer (~250-350 km).
    F,
}

// ---------------------------------------------------------------------------
// Task 2.1 — Chapman layer electron density
// ---------------------------------------------------------------------------

/// Chapman-layer electron density at a given height.
///
/// N(h) = N_max * exp(0.5 * (1 - z - exp(-z)))
/// where z = (h - hm) / H
pub fn chapman_density(height_km: f64, n_max: f64, hm_km: f64, scale_height_km: f64) -> f64 {
    let z = (height_km - hm_km) / scale_height_km;
    n_max * (0.5 * (1.0 - z - (-z).exp())).exp()
}

// ---------------------------------------------------------------------------
// Task 2.2 — Critical frequency with diurnal variation
// ---------------------------------------------------------------------------

/// foF2 with diurnal variation.
///
/// `solar_local_time_hours` in \[0, 24). Peak at local noon (12 h),
/// minimum near midnight.
///
/// Formula: `base * (0.6 + 0.4 * cos(2π * (slt - 12) / 24))`
pub fn fo_f2_diurnal(base_fo_f2_mhz: f64, solar_local_time_hours: f64) -> f64 {
    let phase = 2.0 * std::f64::consts::PI * (solar_local_time_hours - 12.0) / 24.0;
    base_fo_f2_mhz * (0.6 + 0.4 * phase.cos())
}

// ---------------------------------------------------------------------------
// Task 2.3 — Maximum Usable Frequency
// ---------------------------------------------------------------------------

/// Maximum Usable Frequency for a given ground range and F-layer height.
///
/// MUF = foF2 / cos(incidence_angle)
///
/// The incidence angle is derived from the triangle formed by the transmitter,
/// the reflection point, and the Earth centre.
pub fn muf(fo_f2_mhz: f64, ground_range_km: f64, virtual_height_km: f64) -> f64 {
    let half_range = ground_range_km / 2.0;
    let incidence = (half_range / (half_range.powi(2) + virtual_height_km.powi(2)).sqrt()).asin();
    // Secant law: MUF = foF2 * sec(incidence)
    fo_f2_mhz / incidence.cos()
}

// ---------------------------------------------------------------------------
// Task 2.4 — Skip zone
// ---------------------------------------------------------------------------

/// Minimum ground range (km) for single-hop F-layer propagation at `freq_mhz`.
///
/// When the operating frequency exceeds foF2 at vertical incidence the signal
/// must travel obliquely; this computes the minimum range at which reflection
/// can still occur.
pub fn skip_zone_range_km(freq_mhz: f64, params: &IonosphereParams) -> f64 {
    if freq_mhz <= params.fo_f2_mhz {
        return 0.0; // frequency below critical — no skip zone
    }
    // cos(incidence) = foF2 / freq  =>  incidence = acos(foF2 / freq)
    let cos_inc = params.fo_f2_mhz / freq_mhz;
    let inc = cos_inc.acos();
    // Ground range from geometry: half_range = h * tan(inc)
    let half_range = params.hm_f2_km * inc.tan();
    2.0 * half_range
}

// ---------------------------------------------------------------------------
// Task 2.5 — Virtual reflection height
// ---------------------------------------------------------------------------

/// Virtual reflection height (km) for a given ground range and layer.
///
/// E-layer: approximately constant at 110 km.
/// F-layer: increases with range, typically 250-350 km.
pub fn virtual_height_km(ground_range_km: f64, layer: Layer) -> f64 {
    match layer {
        Layer::E => 110.0,
        Layer::F => {
            // Linear model: base 250 km, increasing gently with range
            let base = 250.0;
            let slope = 0.02; // km per km of ground range
            let h = base + slope * ground_range_km;
            h.min(350.0) // cap at 350 km
        }
    }
}

// ---------------------------------------------------------------------------
// Task 2.6 — Ionospheric sounder model
// ---------------------------------------------------------------------------

/// Simulate a vertical-incidence ionospheric sounder measurement.
///
/// Returns estimated `(foF2_mhz, hmF2_km)` — in a simulation these are the
/// true params (a real sounder would add measurement noise).
pub fn sounder_measurement(params: &IonosphereParams) -> (f64, f64) {
    (params.fo_f2_mhz, params.hm_f2_km)
}

// ---------------------------------------------------------------------------
// Task 2.7 — Oblique ionogram
// ---------------------------------------------------------------------------

/// Compute an oblique ionogram: group path (km) vs frequency (MHz) for a
/// given ground range.
///
/// Returns a vector of `(frequency_mhz, group_path_km)` pairs.
/// Frequencies above the MUF for this range are omitted (no reflection).
pub fn oblique_ionogram(
    ground_range_km: f64,
    params: &IonosphereParams,
    freq_range: (f64, f64),
    n_points: usize,
) -> Vec<(f64, f64)> {
    let vh = virtual_height_km(ground_range_km, Layer::F);
    let muf_val = muf(params.fo_f2_mhz, ground_range_km, vh);

    let (f_lo, f_hi) = freq_range;
    let step = if n_points > 1 {
        (f_hi - f_lo) / (n_points - 1) as f64
    } else {
        0.0
    };

    let mut result = Vec::with_capacity(n_points);
    for i in 0..n_points {
        let f = f_lo + step * i as f64;
        if f > muf_val {
            continue; // no reflection above MUF
        }
        // Group path = slant path via virtual height
        let half = ground_range_km / 2.0;
        let slant = (half.powi(2) + vh.powi(2)).sqrt();
        // Near MUF the group path diverges; apply Breit-Tuve factor
        let ratio = f / muf_val;
        let factor = 1.0 / (1.0 - ratio.powi(2)).sqrt();
        let group_path = 2.0 * slant * factor;
        result.push((f, group_path));
    }
    result
}

// ---------------------------------------------------------------------------
// Helper: compute elevation angle from ground range and virtual height
// ---------------------------------------------------------------------------

/// Elevation angle (radians) for a ray reaching `ground_range_km` via
/// reflection at `virtual_height_km`, using spherical-Earth geometry.
pub fn elevation_angle_rad(ground_range_km: f64, virtual_height_km: f64) -> f64 {
    let half_angle = ground_range_km / (2.0 * EARTH_RADIUS_KM); // half subtended angle
    let r = EARTH_RADIUS_KM;
    let h = virtual_height_km;
    // Law of cosines in the triangle (centre, tx, reflection)
    let d = ((r + h).powi(2) + r.powi(2) - 2.0 * r * (r + h) * half_angle.cos()).sqrt();
    let elev = ((r + h).powi(2) - r.powi(2) - d.powi(2)) / (2.0 * r * d);
    elev.acos() - std::f64::consts::FRAC_PI_2
}

// =========================================================================
// Tests — Tasks 2.8-2.10
// =========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn default_params() -> IonosphereParams {
        IonosphereParams {
            fo_f2_mhz: 8.0,
            hm_f2_km: 300.0,
            scale_height_km: 80.0,
            fo_e_mhz: 3.5,
            hm_e_km: 110.0,
        }
    }

    // Task 2.8 — MUF increases with foF2
    #[test]
    fn muf_increases_with_fo_f2() {
        let range = 1000.0;
        let vh = 300.0;
        let muf_low = muf(6.0, range, vh);
        let muf_high = muf(10.0, range, vh);
        assert!(
            muf_high > muf_low,
            "MUF should increase with foF2: {muf_high} > {muf_low}"
        );
    }

    // Task 2.9 — Skip zone increases with frequency
    #[test]
    fn skip_zone_increases_with_frequency() {
        let params = default_params();
        let skip_10 = skip_zone_range_km(10.0, &params);
        let skip_15 = skip_zone_range_km(15.0, &params);
        assert!(
            skip_15 > skip_10,
            "Skip zone should increase with freq: {skip_15} > {skip_10}"
        );
    }

    // Task 2.10 — Diurnal foF2 peaks at noon
    #[test]
    fn diurnal_peaks_at_noon() {
        let base = 8.0;
        let noon = fo_f2_diurnal(base, 12.0);
        let midnight = fo_f2_diurnal(base, 0.0);
        let dawn = fo_f2_diurnal(base, 6.0);
        assert!(
            noon > midnight,
            "Noon ({noon}) should exceed midnight ({midnight})"
        );
        assert!(noon > dawn, "Noon ({noon}) should exceed dawn ({dawn})");
        // Noon should be the maximum (base * 1.0)
        assert!((noon - base).abs() < 1e-10, "Noon should equal base foF2");
    }

    #[test]
    fn chapman_peak_at_hm() {
        let n_max = 1e12;
        let hm = 300.0;
        let h_scale = 80.0;
        let density = chapman_density(hm, n_max, hm, h_scale);
        assert!(
            (density - n_max).abs() / n_max < 1e-10,
            "Density at hm should equal n_max"
        );
    }

    #[test]
    fn chapman_decreases_away_from_peak() {
        let n_max = 1e12;
        let hm = 300.0;
        let h_scale = 80.0;
        let at_peak = chapman_density(hm, n_max, hm, h_scale);
        let above = chapman_density(hm + 100.0, n_max, hm, h_scale);
        let below = chapman_density(hm - 100.0, n_max, hm, h_scale);
        assert!(at_peak > above);
        assert!(at_peak > below);
    }

    #[test]
    fn virtual_height_e_layer_constant() {
        assert!((virtual_height_km(500.0, Layer::E) - 110.0).abs() < 1e-10);
        assert!((virtual_height_km(2000.0, Layer::E) - 110.0).abs() < 1e-10);
    }

    #[test]
    fn virtual_height_f_layer_increases() {
        let h1 = virtual_height_km(500.0, Layer::F);
        let h2 = virtual_height_km(2000.0, Layer::F);
        assert!(h2 > h1);
        assert!(h1 >= 250.0);
        assert!(h2 <= 350.0);
    }

    #[test]
    fn oblique_ionogram_below_muf_only() {
        let params = default_params();
        let result = oblique_ionogram(1000.0, &params, (2.0, 30.0), 100);
        for &(f, _gp) in &result {
            let vh = virtual_height_km(1000.0, Layer::F);
            let m = muf(params.fo_f2_mhz, 1000.0, vh);
            assert!(f <= m, "Frequency {f} should not exceed MUF {m}");
        }
        assert!(!result.is_empty(), "Should have at least some valid points");
    }

    #[test]
    fn sounder_returns_params() {
        let params = default_params();
        let (f, h) = sounder_measurement(&params);
        assert!((f - params.fo_f2_mhz).abs() < 1e-10);
        assert!((h - params.hm_f2_km).abs() < 1e-10);
    }

    #[test]
    fn skip_zone_zero_below_critical() {
        let params = default_params();
        let skip = skip_zone_range_km(5.0, &params); // 5 < 8 MHz
        assert!(skip.abs() < 1e-10, "No skip zone below critical frequency");
    }
}
