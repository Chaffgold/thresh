//! Atmospheric density models and ballistic-coefficient drag.
//!
//! Two Earth-specific density models live here:
//!
//! * The **piecewise-exponential** table and lookup, moved verbatim from
//!   `thresh-synth/src/orbital.rs` (design Decision 1 of the
//!   `orbital-ballistic-filter-models` change) so that filter prediction and
//!   synthetic truth generation share one atmosphere (Vallado Table 8-4
//!   lineage). This remains the reentry path's model — nothing about it (or
//!   [`drag_acceleration`]) changed when the Harris-Priester model landed.
//! * The **Harris-Priester** min/max table with diurnal-bulge interpolation
//!   ([`harris_priester_density`]), added by the `orbit-propagation-fidelity`
//!   change (design Decision 4) as a deterministic, space-weather-free drag
//!   density for the 100–1000 km band.

use nalgebra::Vector3;

use super::gravity::GravityModel;
use crate::eci::EARTH_ROTATION_RATE;
use crate::geodetic::ecef_to_wgs84;

/// (base altitude km, nominal density kg/m³, scale height km)
pub const ATMOSPHERE_TABLE: &[(f64, f64, f64)] = &[
    (0.0, 1.225, 7.249),
    (25.0, 3.899e-2, 6.349),
    (30.0, 1.774e-2, 6.682),
    (40.0, 3.972e-3, 7.554),
    (50.0, 1.057e-3, 8.382),
    (60.0, 3.206e-4, 7.714),
    (70.0, 8.770e-5, 6.549),
    (80.0, 1.905e-5, 5.799),
    (90.0, 3.396e-6, 5.382),
    (100.0, 5.297e-7, 5.877),
    (110.0, 9.661e-8, 7.263),
    (120.0, 2.438e-8, 9.473),
    (130.0, 8.484e-9, 12.636),
    (140.0, 3.845e-9, 16.149),
    (150.0, 2.070e-9, 22.523),
    (180.0, 5.464e-10, 29.740),
    (200.0, 2.789e-10, 37.105),
    (250.0, 7.248e-11, 45.546),
    (300.0, 2.418e-11, 53.628),
    (350.0, 9.518e-12, 53.298),
    (400.0, 3.725e-12, 58.515),
    (450.0, 1.585e-12, 60.828),
    (500.0, 6.967e-13, 63.822),
    (600.0, 1.454e-13, 71.835),
    (700.0, 3.614e-14, 88.667),
    (800.0, 1.170e-14, 124.64),
    (900.0, 5.245e-15, 181.05),
    (1000.0, 3.019e-15, 268.00),
];

/// Compute atmospheric density (kg/m³) at a given geometric altitude (m)
/// using a piecewise-exponential model.
pub fn atmosphere_density(alt_m: f64) -> f64 {
    let alt_km = alt_m / 1000.0;
    if alt_km < 0.0 {
        return ATMOSPHERE_TABLE[0].1;
    }
    if alt_km > 1000.0 {
        return 0.0;
    }
    // Find the bracket
    let mut idx = 0;
    for (i, &(h, _, _)) in ATMOSPHERE_TABLE.iter().enumerate() {
        if h <= alt_km {
            idx = i;
        } else {
            break;
        }
    }
    let (h0, rho0, scale_h) = ATMOSPHERE_TABLE[idx];
    rho0 * (-((alt_km - h0) / scale_h)).exp()
}

/// Compute atmospheric drag acceleration for an ECI state.
///
/// Parameterized on the **inverse ballistic coefficient**
/// `inv_beta = C_d·A/m = 1/β` (m²/kg). Uses the co-rotating atmosphere
/// relative velocity `v_rel = v − ω_⊕ × r` (with
/// [`EARTH_ROTATION_RATE`]), so Coriolis/centrifugal drag effects are
/// captured without an ECEF state. Altitude for the density lookup is
/// geometric, `‖r‖ − equatorial_radius`, using
/// [`GravityModel::EARTH_WGS84`] — the atmosphere model is Earth-specific.
///
/// Returns zero outside the 0–1000 km altitude band of [`ATMOSPHERE_TABLE`]
/// and for negligible relative speed.
pub fn drag_acceleration(pos: &Vector3<f64>, vel: &Vector3<f64>, inv_beta: f64) -> Vector3<f64> {
    let r = (pos.x * pos.x + pos.y * pos.y + pos.z * pos.z).sqrt();
    let alt = r - GravityModel::EARTH_WGS84.equatorial_radius;
    if !(0.0..=1_000_000.0).contains(&alt) {
        return Vector3::zeros();
    }

    // Approximate atmospheric co-rotation velocity
    let v_atm = Vector3::new(
        -EARTH_ROTATION_RATE * pos.y,
        EARTH_ROTATION_RATE * pos.x,
        0.0,
    );
    let v_rel = Vector3::new(vel.x - v_atm.x, vel.y - v_atm.y, vel.z - v_atm.z);
    let v_rel_mag = (v_rel.x * v_rel.x + v_rel.y * v_rel.y + v_rel.z * v_rel.z).sqrt();

    if v_rel_mag < 1e-10 {
        return Vector3::zeros();
    }

    let rho = atmosphere_density(alt);
    let drag_factor = -0.5 * inv_beta * rho * v_rel_mag;

    Vector3::new(
        drag_factor * v_rel.x,
        drag_factor * v_rel.y,
        drag_factor * v_rel.z,
    )
}

// ---------------------------------------------------------------------------
// Harris-Priester density (orbit-propagation-fidelity, design Decision 4)
// ---------------------------------------------------------------------------

/// Harris-Priester min/max density table, one row per altitude node:
/// `(altitude m, ρ_min kg/m³, ρ_max kg/m³)`. Mean solar activity,
/// 100–1000 km.
///
/// # Provenance (fetched 2026-07-16)
///
/// Transcribed mechanically (script-parsed, not hand-copied) from Orekit's
/// `HarrisPriester.java` `ALT_RHO` table, commit
/// `77cfa41667457f0db879a0fc7e892e7248c1e516`
/// (<https://github.com/CS-SI/Orekit/blob/develop/src/main/java/org/orekit/models/earth/atmosphere/HarrisPriester.java>),
/// which cites Montenbruck & Gill, *Satellite Orbits*, Springer 2005. All
/// 50 rows cross-validated bit-for-bit against the independent
/// transcription in SatelliteToolboxAtmosphericModels.jl
/// `constants.jl` (`_HARRIS_PRIESTER_ALT_RHO`), commit
/// `268c2645923d5a1c4206785c9c98f1ff2eccd0e4`
/// (<https://github.com/JuliaSpace/SatelliteToolboxAtmosphericModels.jl/blob/main/src/harrispriester/constants.jl>),
/// and against the committed golden fixture
/// `test-data/golden/propagation/harris-priester-table.json`. Checksum
/// tripwire tests live in `harris_priester_tests`.
pub const HARRIS_PRIESTER_TABLE: &[(f64, f64, f64)] = &[
    (100000.0, 4.974e-07, 4.974e-07),
    (120000.0, 2.49e-08, 2.49e-08),
    (130000.0, 8.377e-09, 8.71e-09),
    (140000.0, 3.899e-09, 4.059e-09),
    (150000.0, 2.122e-09, 2.215e-09),
    (160000.0, 1.263e-09, 1.344e-09),
    (170000.0, 8.008e-10, 8.758e-10),
    (180000.0, 5.283e-10, 6.01e-10),
    (190000.0, 3.617e-10, 4.297e-10),
    (200000.0, 2.557e-10, 3.162e-10),
    (210000.0, 1.839e-10, 2.396e-10),
    (220000.0, 1.341e-10, 1.853e-10),
    (230000.0, 9.949e-11, 1.455e-10),
    (240000.0, 7.488e-11, 1.157e-10),
    (250000.0, 5.709e-11, 9.308e-11),
    (260000.0, 4.403e-11, 7.555e-11),
    (270000.0, 3.43e-11, 6.182e-11),
    (280000.0, 2.697e-11, 5.095e-11),
    (290000.0, 2.139e-11, 4.226e-11),
    (300000.0, 1.708e-11, 3.526e-11),
    (320000.0, 1.099e-11, 2.511e-11),
    (340000.0, 7.214e-12, 1.819e-11),
    (360000.0, 4.824e-12, 1.337e-11),
    (380000.0, 3.274e-12, 9.955e-12),
    (400000.0, 2.249e-12, 7.492e-12),
    (420000.0, 1.558e-12, 5.684e-12),
    (440000.0, 1.091e-12, 4.355e-12),
    (460000.0, 7.701e-13, 3.362e-12),
    (480000.0, 5.474e-13, 2.612e-12),
    (500000.0, 3.916e-13, 2.042e-12),
    (520000.0, 2.819e-13, 1.605e-12),
    (540000.0, 2.042e-13, 1.267e-12),
    (560000.0, 1.488e-13, 1.005e-12),
    (580000.0, 1.092e-13, 7.997e-13),
    (600000.0, 8.07e-14, 6.39e-13),
    (620000.0, 6.012e-14, 5.123e-13),
    (640000.0, 4.519e-14, 4.121e-13),
    (660000.0, 3.43e-14, 3.325e-13),
    (680000.0, 2.632e-14, 2.691e-13),
    (700000.0, 2.043e-14, 2.185e-13),
    (720000.0, 1.607e-14, 1.779e-13),
    (740000.0, 1.281e-14, 1.452e-13),
    (760000.0, 1.036e-14, 1.19e-13),
    (780000.0, 8.496e-15, 9.776e-14),
    (800000.0, 7.069e-15, 8.059e-14),
    (840000.0, 4.68e-15, 5.741e-14),
    (880000.0, 3.2e-15, 4.21e-14),
    (920000.0, 2.21e-15, 3.13e-14),
    (960000.0, 1.56e-15, 2.36e-14),
    (1000000.0, 1.15e-15, 1.81e-14),
];

/// Diurnal-bulge apex lag east of the subsolar point, degrees (the classic
/// Harris-Priester 30° ≈ 2 h lag; Orekit's `LAG = FastMath.toRadians(30.0)`
/// in the fetched source cited at [`HARRIS_PRIESTER_TABLE`]).
pub const HARRIS_PRIESTER_APEX_LAG_DEG: f64 = 30.0;

/// Lower edge of the Harris-Priester altitude band (m) — the first
/// [`HARRIS_PRIESTER_TABLE`] node.
pub const HARRIS_PRIESTER_MIN_ALTITUDE_M: f64 = 100_000.0;

/// Upper edge of the Harris-Priester altitude band (m) — the last
/// [`HARRIS_PRIESTER_TABLE`] node.
pub const HARRIS_PRIESTER_MAX_ALTITUDE_M: f64 = 1_000_000.0;

/// Harris-Priester atmospheric density (kg/m³) at geodetic height
/// `height_m` with diurnal-bulge angle cosine `cos_psi` (ψ = angle between
/// the position direction and the bulge apex; see
/// [`harris_priester_density_itrf`] for the apex construction).
///
/// The convention mirrors the fetched Orekit reference and the committed
/// golden fixture exactly (both cited at [`HARRIS_PRIESTER_TABLE`]):
///
/// * exponential interpolation of ρ_min and ρ_max between the bracketing
///   table nodes: `ρₓ(h) = ρₓ(hᵢ)·(ρₓ(hᵢ₊₁)/ρₓ(hᵢ))^((hᵢ−h)/(hᵢ−hᵢ₊₁))`;
/// * `ρ = ρ_min + (ρ_max − ρ_min)·cos²(ψ/2)` with
///   `cos²(ψ/2) = (1 + cos ψ)/2` — the bulge exponent is **fixed at
///   n = 2** (design Decision 4; the inclination-dependent 2–6 refinement
///   is deferred).
///
/// # Altitude clamping (documented per design Decision 4)
///
/// * Above [`HARRIS_PRIESTER_MAX_ALTITUDE_M`] the density is `0.0`.
/// * Below [`HARRIS_PRIESTER_MIN_ALTITUDE_M`] the height is clamped to the
///   band floor (the 100 km node, where ρ_min = ρ_max) — Harris-Priester
///   is not a reentry model; the piecewise-exponential
///   [`atmosphere_density`] remains the reentry path's model.
pub fn harris_priester_density(height_m: f64, cos_psi: f64) -> f64 {
    if height_m > HARRIS_PRIESTER_MAX_ALTITUDE_M {
        return 0.0;
    }
    let h = height_m.max(HARRIS_PRIESTER_MIN_ALTITUDE_M);
    let i = hp_bracket_index(h);
    let rho_min = hp_interpolate(h, i, 1);
    let rho_max = hp_interpolate(h, i, 2);
    // cos²(ψ/2) = (1 + cos ψ)/2, floored at zero exactly like the fetched
    // reference (`c2Psi2` in Orekit; defensive against cos_psi < −1 noise).
    let bulge = ((1.0 + cos_psi) / 2.0).max(0.0);
    rho_min + (rho_max - rho_min) * bulge
}

/// Index `i` of the table interval `[hᵢ, hᵢ₊₁]` bracketing `h`, matching
/// the fetched reference's search (the last interval also serves
/// `h = hᵢ₊₁` exactly).
///
/// Precondition: `h` is inside the clamped band (established by
/// [`harris_priester_density`]).
fn hp_bracket_index(h: f64) -> usize {
    let mut i = 0;
    while i < HARRIS_PRIESTER_TABLE.len() - 2 && h > HARRIS_PRIESTER_TABLE[i + 1].0 {
        i += 1;
    }
    i
}

/// Exponential node interpolation of table column `column` (1 = ρ_min,
/// 2 = ρ_max) over the interval starting at row `i`:
/// `ρₓ(h) = ρₓ(hᵢ)·(ρₓ(hᵢ₊₁)/ρₓ(hᵢ))^dH` with
/// `dH = (hᵢ−h)/(hᵢ−hᵢ₊₁)`.
fn hp_interpolate(h: f64, i: usize, column: usize) -> f64 {
    let (h_i, min_i, max_i) = HARRIS_PRIESTER_TABLE[i];
    let (h_j, min_j, max_j) = HARRIS_PRIESTER_TABLE[i + 1];
    let (rho_i, rho_j) = if column == 1 {
        (min_i, min_j)
    } else {
        (max_i, max_j)
    };
    let dh = (h_i - h) / (h_i - h_j);
    rho_i * (rho_j / rho_i).powf(dh)
}

/// The diurnal-bulge apex direction (unit vector, ITRF): the Sun direction
/// rotated by +[`HARRIS_PRIESTER_APEX_LAG_DEG`] about the ITRF +z axis
/// (right ascension + 30°, declination preserved — the fetched Orekit
/// `bulDir` construction and the fixture convention).
fn hp_bulge_apex(sun_itrf: &Vector3<f64>) -> Vector3<f64> {
    let s = sun_itrf.normalize();
    let (sin_lag, cos_lag) = HARRIS_PRIESTER_APEX_LAG_DEG.to_radians().sin_cos();
    Vector3::new(
        s.x * cos_lag - s.y * sin_lag,
        s.x * sin_lag + s.y * cos_lag,
        s.z,
    )
}

/// Harris-Priester density (kg/m³) at an **ITRF** position given the
/// **ITRF** Sun position: geodetic WGS-84 height via
/// [`ecef_to_wgs84`], bulge angle against
/// the apex 30° east of the subsolar direction (see
/// [`harris_priester_density`] for the interpolation/bulge convention and
/// the altitude clamps).
///
/// Both vectors must be in the same Earth-fixed frame; the
/// force-configuration layer rotates GCRF inputs through the
/// `FrameProvider` before calling this (spec: "Earth-fixed force legs use
/// the frame provider").
pub fn harris_priester_density_itrf(r_itrf: &Vector3<f64>, sun_itrf: &Vector3<f64>) -> f64 {
    let (_lat, _lon, height) = ecef_to_wgs84(r_itrf);
    let apex = hp_bulge_apex(sun_itrf);
    let cos_psi = r_itrf.normalize().dot(&apex);
    harris_priester_density(height, cos_psi)
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── Density lookup ───────────────────────────────────────────────────

    #[test]
    fn density_decreases_with_altitude() {
        let rho_200 = atmosphere_density(200_000.0);
        let rho_400 = atmosphere_density(400_000.0);
        let rho_800 = atmosphere_density(800_000.0);
        assert!(rho_200 > rho_400);
        assert!(rho_400 > rho_800);
        assert!(rho_200 > 0.0);
    }

    #[test]
    fn density_clamps_to_sea_level_below_zero_altitude() {
        assert_eq!(atmosphere_density(-100.0), ATMOSPHERE_TABLE[0].1);
    }

    #[test]
    fn density_is_zero_above_table_ceiling() {
        assert_eq!(atmosphere_density(1_000_001.0), 0.0);
    }

    #[test]
    fn density_equals_table_value_at_bracket_base() {
        // At exactly 100 km the exponential factor is exp(0) = 1, so the
        // lookup returns the table row's nominal density.
        assert_eq!(atmosphere_density(100_000.0), 5.297e-7);
    }

    // ── Drag acceleration ────────────────────────────────────────────────

    /// LEO-ish state inside the sensible atmosphere (~300 km altitude).
    fn leo_state() -> (Vector3<f64>, Vector3<f64>) {
        let r = GravityModel::EARTH_WGS84.equatorial_radius + 300_000.0;
        (Vector3::new(r, 0.0, 0.0), Vector3::new(0.0, 7_700.0, 0.0))
    }

    #[test]
    fn drag_opposes_corotating_relative_velocity() {
        let (pos, vel) = leo_state();
        let inv_beta = 2.2 * 20.0 / 500.0;
        let acc = drag_acceleration(&pos, &vel, inv_beta);

        let v_atm = Vector3::new(
            -EARTH_ROTATION_RATE * pos.y,
            EARTH_ROTATION_RATE * pos.x,
            0.0,
        );
        let v_rel = vel - v_atm;

        assert!(acc.norm() > 0.0, "drag should be nonzero at 300 km");
        assert!(acc.dot(&v_rel) < 0.0, "drag should oppose v_rel");
        // Anti-parallel: the cross product with v_rel vanishes.
        assert!(acc.cross(&v_rel).norm() < 1e-12 * acc.norm() * v_rel.norm());
    }

    #[test]
    fn drag_magnitude_matches_half_rho_v2_over_beta() {
        let (pos, vel) = leo_state();
        let inv_beta = 2.2 * 20.0 / 500.0;
        let acc = drag_acceleration(&pos, &vel, inv_beta);

        let v_atm = Vector3::new(
            -EARTH_ROTATION_RATE * pos.y,
            EARTH_ROTATION_RATE * pos.x,
            0.0,
        );
        let v_rel = vel - v_atm;
        let rho = atmosphere_density(pos.norm() - GravityModel::EARTH_WGS84.equatorial_radius);
        let expected = 0.5 * rho * v_rel.norm_squared() * inv_beta;

        assert!((acc.norm() - expected).abs() / expected < 1e-12);
    }

    #[test]
    fn drag_scales_linearly_with_inv_beta() {
        let (pos, vel) = leo_state();
        let a1 = drag_acceleration(&pos, &vel, 0.01);
        let a2 = drag_acceleration(&pos, &vel, 0.02);
        assert!((a2.norm() - 2.0 * a1.norm()).abs() < 1e-12 * a1.norm());
    }

    #[test]
    fn drag_vanishes_above_atmosphere_ceiling() {
        let r = GravityModel::EARTH_WGS84.equatorial_radius + 1_500_000.0;
        let acc = drag_acceleration(
            &Vector3::new(r, 0.0, 0.0),
            &Vector3::new(0.0, 7_000.0, 0.0),
            0.1,
        );
        assert_eq!(acc, Vector3::zeros());
    }

    #[test]
    fn drag_vanishes_when_moving_with_the_atmosphere() {
        let (pos, _) = leo_state();
        // Velocity exactly equal to the atmospheric co-rotation velocity.
        let v_atm = Vector3::new(
            -EARTH_ROTATION_RATE * pos.y,
            EARTH_ROTATION_RATE * pos.x,
            0.0,
        );
        let acc = drag_acceleration(&pos, &v_atm, 0.1);
        assert_eq!(acc, Vector3::zeros());
    }
}

#[cfg(test)]
mod harris_priester_tests {
    use super::*;
    use crate::geodetic::WGS84_A;

    // ── Transcription tripwires (nutation1980.rs pattern) ───────────────

    /// Exact-f64 column sums computed at parse time from the fetched Orekit
    /// source (identical in the SatelliteToolbox.jl cross-check and the
    /// committed fixture — see the table's provenance docs). Any dropped,
    /// duplicated, or mistyped value moves at least one sum.
    #[test]
    fn table_checksums_match_fetched_sources() {
        assert_eq!(HARRIS_PRIESTER_TABLE.len(), 50);
        let alt_sum: f64 = HARRIS_PRIESTER_TABLE.iter().map(|&(a, _, _)| a).sum();
        let min_sum: f64 = HARRIS_PRIESTER_TABLE.iter().map(|&(_, mn, _)| mn).sum();
        let max_sum: f64 = HARRIS_PRIESTER_TABLE.iter().map(|&(_, _, mx)| mx).sum();
        assert_eq!(alt_sum, 22_690_000.0);
        assert_eq!(min_sum, 5.40634717865e-07);
        assert_eq!(max_sum, 5.419957451600003e-07);
    }

    /// First and last rows verbatim against the fetched source lines
    /// `{  100000.0, 4.974e-07, 4.974e-07 }` and
    /// `{ 1000000.0, 1.150e-15, 1.810e-14 }`.
    #[test]
    fn boundary_rows_match_fetched_source() {
        assert_eq!(HARRIS_PRIESTER_TABLE[0], (100_000.0, 4.974e-7, 4.974e-7));
        assert_eq!(HARRIS_PRIESTER_TABLE[49], (1_000_000.0, 1.15e-15, 1.81e-14));
    }

    /// Structural sanity: altitudes strictly increase, ρ_min ≤ ρ_max
    /// everywhere, and the 100/120 km rows are isotropic (min = max).
    #[test]
    fn table_is_structurally_sound() {
        for window in HARRIS_PRIESTER_TABLE.windows(2) {
            assert!(window[0].0 < window[1].0);
        }
        for &(alt, rho_min, rho_max) in HARRIS_PRIESTER_TABLE {
            assert!(rho_min > 0.0, "non-positive density at {alt}");
            assert!(rho_min <= rho_max, "min > max at {alt}");
        }
        assert_eq!(HARRIS_PRIESTER_TABLE[0].1, HARRIS_PRIESTER_TABLE[0].2);
        assert_eq!(HARRIS_PRIESTER_TABLE[1].1, HARRIS_PRIESTER_TABLE[1].2);
        assert_eq!(HARRIS_PRIESTER_TABLE[0].0, HARRIS_PRIESTER_MIN_ALTITUDE_M);
        assert_eq!(HARRIS_PRIESTER_TABLE[49].0, HARRIS_PRIESTER_MAX_ALTITUDE_M);
    }

    // ── Spec: "Published table values reproduced" ────────────────────────

    /// At every table-node altitude, the apex (cos ψ = 1) density equals
    /// the node's ρ_max and the anti-apex (cos ψ = −1) density the node's
    /// ρ_min (to interpolation round-off: node hits land on a bracket edge
    /// where dH ∈ {0, 1}).
    #[test]
    fn table_node_densities_reproduce_published_values() {
        for &(alt, rho_min, rho_max) in HARRIS_PRIESTER_TABLE {
            let at_apex = harris_priester_density(alt, 1.0);
            let at_anti = harris_priester_density(alt, -1.0);
            assert!(
                (at_apex - rho_max).abs() <= 1e-12 * rho_max,
                "apex density {at_apex} != table max {rho_max} at {alt} m"
            );
            assert!(
                (at_anti - rho_min).abs() <= 1e-12 * rho_min,
                "anti-apex density {at_anti} != table min {rho_min} at {alt} m"
            );
        }
    }

    /// The committed fixture's worked examples
    /// (`test-data/golden/propagation/harris-priester-table.json`,
    /// generator-computed from the same fetched table + convention),
    /// including the off-node exponential-interpolation checks.
    #[test]
    fn fixture_worked_examples_reproduce() {
        let cases = [
            (400_000.0, 1.0, 7.492e-12),
            (400_000.0, -1.0, 2.249e-12),
            (400_000.0, 0.0, 4.8705e-12),
            (410_000.0, 1.0, 6.525682186561035e-12),
            (410_000.0, -1.0, 1.8718819407216898e-12),
        ];
        for (height, cos_psi, expected) in cases {
            let rho = harris_priester_density(height, cos_psi);
            assert!(
                (rho - expected).abs() <= 1e-12 * expected,
                "h = {height}, cos psi = {cos_psi}: {rho} != {expected}"
            );
        }
    }

    // ── Spec: "Density between bulge extremes" ───────────────────────────

    /// At fixed height the density runs from the max interpolation at the
    /// apex to the min at the anti-apex, strictly monotonically in ψ.
    #[test]
    fn bulge_interpolates_monotonically_between_extremes() {
        let height = 400_000.0;
        let apex = harris_priester_density(height, 1.0);
        let anti = harris_priester_density(height, -1.0);
        assert!(apex > anti, "400 km bulge should be anisotropic");

        let mut previous = f64::MAX;
        for step in 0..=12 {
            let psi = f64::from(step) * std::f64::consts::PI / 12.0;
            let rho = harris_priester_density(height, psi.cos());
            assert!(
                rho < previous,
                "density must decrease strictly with psi (step {step})"
            );
            assert!((anti..=apex).contains(&rho), "density outside extremes");
            previous = rho;
        }
    }

    // ── Documented clamps ────────────────────────────────────────────────

    #[test]
    fn clamps_outside_the_band_as_documented() {
        assert_eq!(harris_priester_density(1_000_001.0, 1.0), 0.0);
        // Below the band: clamped to the 100 km node, which is isotropic.
        let floor = harris_priester_density(50_000.0, 0.3);
        assert_eq!(floor, harris_priester_density(100_000.0, 0.3));
        assert!((floor - 4.974e-7).abs() <= 1e-12 * 4.974e-7);
    }

    // ── ITRF entry point: geodetic height + apex geometry ────────────────

    /// A satellite on the equator at the apex right ascension (Sun along
    /// +x, apex at RA 30°) sees exactly the apex density; 180° away it sees
    /// the anti-apex density. On the equator the geodetic height is exactly
    /// ‖r‖ − a.
    #[test]
    fn itrf_density_places_apex_thirty_degrees_east_of_the_sun() {
        let sun_itrf = Vector3::new(1.496e11, 0.0, 0.0);
        let radius = WGS84_A + 400_000.0;
        let (sin30, cos30) = 30.0_f64.to_radians().sin_cos();
        let at_apex = Vector3::new(radius * cos30, radius * sin30, 0.0);
        let at_anti = -at_apex;

        let rho_apex = harris_priester_density_itrf(&at_apex, &sun_itrf);
        let rho_anti = harris_priester_density_itrf(&at_anti, &sun_itrf);
        let expected_apex = harris_priester_density(400_000.0, 1.0);
        let expected_anti = harris_priester_density(400_000.0, -1.0);
        assert!(
            (rho_apex - expected_apex).abs() <= 1e-9 * expected_apex,
            "{rho_apex} != {expected_apex}"
        );
        assert!(
            (rho_anti - expected_anti).abs() <= 1e-9 * expected_anti,
            "{rho_anti} != {expected_anti}"
        );

        // The subsolar direction itself is NOT the apex: density there is
        // strictly below the apex density (30° lag does real work).
        let subsolar = Vector3::new(radius, 0.0, 0.0);
        let rho_subsolar = harris_priester_density_itrf(&subsolar, &sun_itrf);
        assert!(rho_subsolar < rho_apex);
        assert!(rho_subsolar > rho_anti);
    }
}
