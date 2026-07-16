//! Analytic Sun and Moon ephemerides (Meeus / Vallado low-precision series).
//!
//! Embedded-series geocentric positions for the two third bodies the force
//! stack needs (design Decision 5 of the `orbit-propagation-fidelity`
//! change): lunisolar third-body gravity, solar-radiation-pressure
//! direction, and the Harris-Priester diurnal bulge. No ephemeris files, no
//! network.
//!
//! # Models and provenance (coefficients fetched 2026-07-16)
//!
//! * **Sun** — Meeus, *Astronomical Algorithms*, 2nd ed., Ch. 25 "Solar
//!   Coordinates", the low-accuracy method (Eqs. 25.2–25.5; stated accuracy
//!   0.01°). Coefficients transcribed this session from the book-faithful
//!   `soniakeys/meeus` v3 Go implementation (`solar/solar.go`, which cites
//!   each Meeus equation number); spot values below are its
//!   `solar_test.go` Example 25.a expectations.
//! * **Moon** — Vallado, *Fundamentals of Astrodynamics and Applications*,
//!   Algorithm 31 "Moon" (3rd ed. pp. 290–291), the truncated ELP-lineage
//!   series (~0.3°-class). Coefficients transcribed this session from
//!   Vallado's companion `moon.m` (CelesTrak astrodynamics software,
//!   `Spacecraft-Code/Vallado` GitHub mirror, `Matlab/moon.m`); the 3rd-ed
//!   consolidated errata (`celestrak.org/software/vallado/errataver3.pdf`)
//!   confirms the corrected `481267.8813` longitude rate and supplies the
//!   Example 5-3 position used as a spot test.
//!
//! The accuracy figures above are the published classes; the *measured*
//! accuracy against an independent authority (astropy) is recorded with the
//! golden fixtures under `test-data/golden/propagation/` (change task 4.2).
//!
//! # Frame chain and conventions
//!
//! Both series produce spherical coordinates on the **ecliptic of date**
//! referred to the **mean equinox of date**. Rotating about the x axis by
//! the IAU-1980 mean obliquity ([`crate::frames::legs::mean_obliquity_iau1980`])
//! gives the mean-equator-of-date frame ([`crate::frames::Frame::Mod`]);
//! the transpose of the IAU-76 precession leg
//! ([`crate::frames::legs::precession_matrix_iau76`]) then carries MOD into
//! GCRF — the same legs the [`crate::frames`] provider composes, so the
//! chain here is consistent with every other frame transform in the crate.
//!
//! Epochs are the time-scale-aware [`Epoch`]; the series' time argument is
//! Julian centuries of **TT** since J2000.0
//! ([`crate::frames::legs::julian_centuries_tt`]). Meeus specifies TD (= TT)
//! and Vallado TDB; |TT − TDB| < 2 ms, metres at lunar velocity — far below
//! both series' accuracy classes.
//!
//! Positions are **geometric** (no light-time or aberration corrections),
//! which is the physically correct choice for force evaluation. All returned
//! distances are metres.

use nalgebra::Vector3;

use crate::frames::legs::{
    julian_centuries_tt, mean_obliquity_iau1980, precession_matrix_iau76, rotation_x,
};
use crate::orbital::gravity::GravityModel;
use crate::time::Epoch;

/// Astronomical unit in metres — exactly 149 597 870 700 m by definition
/// (IAU 2012 Resolution B2, XXVIIIth General Assembly; confirmed against the
/// IAU/Observatoire de Paris announcement this session).
pub const ASTRONOMICAL_UNIT_M: f64 = 149_597_870_700.0;

/// Degrees to radians.
const DEG_TO_RAD: f64 = std::f64::consts::PI / 180.0;

/// Earth equatorial radius (m) used by the lunar parallax → distance
/// conversion, `r = R⊕ / sin π`. Vallado's `ex5_3.m` converts Earth radii
/// with 6378.137 km — exactly the WGS-84 value already embedded in
/// [`GravityModel::EARTH_WGS84`], which is reused here.
const EARTH_EQUATORIAL_RADIUS_M: f64 = GravityModel::EARTH_WGS84.equatorial_radius;

/// One periodic term `amplitude · trig(phase + rate·T)` of a low-precision
/// series, all angles in degrees, `T` in Julian centuries of TT since
/// J2000.0.
#[derive(Debug, Clone, Copy, PartialEq)]
struct SeriesTerm {
    /// Amplitude, degrees.
    amplitude_deg: f64,
    /// Phase at J2000.0, degrees.
    phase_deg: f64,
    /// Argument rate, degrees per Julian century (TT).
    rate_deg_per_century: f64,
}

/// Shorthand for the [`SeriesTerm`] rows below.
const fn term(amplitude_deg: f64, phase_deg: f64, rate_deg_per_century: f64) -> SeriesTerm {
    SeriesTerm {
        amplitude_deg,
        phase_deg,
        rate_deg_per_century,
    }
}

// ---------------------------------------------------------------------
// Series tables. Sources (both fetched 2026-07-16, see the module docs):
//   * Sun — Meeus AA 2nd ed. Ch. 25 low-accuracy method, coefficient-exact
//     per the soniakeys/meeus v3 `solar/solar.go` transcription.
//   * Moon — Vallado Alg. 31 per the companion `moon.m`
//     (Spacecraft-Code/Vallado mirror), longitude rate 481267.8813 as
//     corrected by the 3rd-ed consolidated errata (pp. 290–291 entry).
// Tripwire checksums over every table live in the test module below.
// ---------------------------------------------------------------------

/// Equation-of-center rows of the Meeus low-accuracy solar series:
/// `(k, [c₀, c₁, c₂])` evaluated as `(c₀ + c₁T + c₂T²) · sin(k·M)` degrees,
/// with `M` the Sun's mean anomaly (Meeus AA 2nd ed. Ch. 25, the unnumbered
/// "equation of the center" expression between Eqs. 25.4 and 25.5).
const SUN_EQUATION_OF_CENTER: [(f64, [f64; 3]); 3] = [
    (1.0, [1.914_602, -0.004_817, -0.000_014]),
    (2.0, [0.019_993, -0.000_101, 0.0]),
    (3.0, [0.000_289, 0.0, 0.0]),
];

/// Periodic terms of the Moon's ecliptic longitude (Vallado Alg. 31 via
/// `moon.m`); added to the mean-longitude polynomial in
/// [`moon_ecliptic_of_date`]. Degrees.
const MOON_LONGITUDE_TERMS: [SeriesTerm; 6] = [
    term(6.29, 134.9, 477_198.85),
    term(-1.27, 259.2, -413_335.38),
    term(0.66, 235.7, 890_534.23),
    term(0.21, 269.9, 954_397.70),
    term(-0.19, 357.5, 35_999.05),
    term(-0.11, 186.6, 966_404.05),
];

/// Periodic terms of the Moon's ecliptic latitude (Vallado Alg. 31 via
/// `moon.m`). Degrees; the latitude has no secular part.
const MOON_LATITUDE_TERMS: [SeriesTerm; 4] = [
    term(5.13, 93.3, 483_202.03),
    term(0.28, 228.2, 960_400.87),
    term(-0.28, 318.3, 6_003.18),
    term(-0.17, 217.6, -407_332.20),
];

/// Cosine terms of the Moon's horizontal parallax (Vallado Alg. 31 via
/// `moon.m`); added to the 0.9508° constant in [`moon_ecliptic_of_date`].
/// Degrees.
const MOON_PARALLAX_TERMS: [SeriesTerm; 4] = [
    term(0.0518, 134.9, 477_198.85),
    term(0.0095, 259.2, -413_335.38),
    term(0.0078, 235.7, 890_534.23),
    term(0.0028, 269.9, 954_397.70),
];

/// Evaluates the quadratic `c₀ + c₁·t + c₂·t²` (Horner form) — the shape of
/// every polynomial in the two series.
fn horner3(t: f64, c: [f64; 3]) -> f64 {
    c[0] + t * (c[1] + t * c[2])
}

/// Sums a low-precision series table:
/// `Σ amplitude · trig(phase + rate·T)`, arguments converted from degrees;
/// the result is in the amplitude's unit (degrees here). `trig` is
/// [`f64::sin`] for longitude/latitude terms, [`f64::cos`] for parallax.
fn evaluate_series(terms: &[SeriesTerm], t_tt: f64, trig: fn(f64) -> f64) -> f64 {
    terms
        .iter()
        .map(|term| {
            term.amplitude_deg
                * trig((term.phase_deg + term.rate_deg_per_century * t_tt) * DEG_TO_RAD)
        })
        .sum()
}

// ---------------------------------------------------------------------
// Sun (Meeus Ch. 25 low-accuracy method) — phase helpers.
// ---------------------------------------------------------------------

/// Sun state on the ecliptic of date: geometric true longitude referred to
/// the mean equinox of date, plus the Earth–Sun distance.
///
/// The low-accuracy method takes the Sun's geometric ecliptic latitude as
/// zero (the neglected latitude is arcsecond-scale, well below the series'
/// 0.01° class).
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct SunEcliptic {
    /// Geometric true longitude, radians in `[0, 2π)`, mean equinox of date.
    pub longitude_rad: f64,
    /// Earth–Sun distance, metres.
    pub distance_m: f64,
}

/// Phase 1 (Sun): evaluate the Meeus Ch. 25 low-accuracy series at `t_tt`
/// Julian centuries of TT since J2000.0.
///
/// Phases: mean longitude L₀ (Eq. 25.2) → mean anomaly M (Eq. 25.3) →
/// eccentricity e (Eq. 25.4) → equation of center C (table) → true
/// longitude L₀ + C and radius vector (Eq. 25.5).
pub(crate) fn sun_ecliptic_of_date(t_tt: f64) -> SunEcliptic {
    // Meeus Eq. 25.2 — geometric mean longitude, mean equinox of date.
    let mean_longitude_deg = horner3(t_tt, [280.466_46, 36_000.769_83, 0.000_303_2]);
    // Meeus Eq. 25.3 — mean anomaly.
    let mean_anomaly_deg = horner3(t_tt, [357.529_11, 35_999.050_29, -0.000_153_7]);
    // Meeus Eq. 25.4 — eccentricity of Earth's orbit.
    let eccentricity = horner3(t_tt, [0.016_708_634, -0.000_042_037, -0.000_000_126_7]);
    let center_deg = sun_equation_of_center_deg(t_tt, mean_anomaly_deg);
    let true_longitude_deg = mean_longitude_deg + center_deg;
    let true_anomaly_rad = (mean_anomaly_deg + center_deg) * DEG_TO_RAD;
    // Meeus Eq. 25.5 — radius vector in AU.
    let distance_au = 1.000_001_018 * (1.0 - eccentricity * eccentricity)
        / (1.0 + eccentricity * true_anomaly_rad.cos());
    SunEcliptic {
        longitude_rad: true_longitude_deg.rem_euclid(360.0) * DEG_TO_RAD,
        distance_m: distance_au * ASTRONOMICAL_UNIT_M,
    }
}

/// Equation of center C (degrees): the [`SUN_EQUATION_OF_CENTER`] rows
/// summed at mean anomaly `mean_anomaly_deg`, coefficients evaluated at
/// `t_tt`.
fn sun_equation_of_center_deg(t_tt: f64, mean_anomaly_deg: f64) -> f64 {
    let m = mean_anomaly_deg * DEG_TO_RAD;
    SUN_EQUATION_OF_CENTER
        .iter()
        .map(|(multiple, coefficients)| horner3(t_tt, *coefficients) * (multiple * m).sin())
        .sum()
}

// ---------------------------------------------------------------------
// Moon (Vallado Alg. 31 truncated series) — phase helpers.
// ---------------------------------------------------------------------

/// Moon state on the ecliptic of date, mean equinox of date.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct MoonEcliptic {
    /// Ecliptic longitude, radians in `[0, 2π)`, mean equinox of date.
    pub longitude_rad: f64,
    /// Ecliptic latitude, radians (bounded by the series amplitude sum,
    /// |β| < 5.86°).
    pub latitude_rad: f64,
    /// Earth–Moon distance, metres, from the horizontal parallax:
    /// `r = R⊕ / sin π`.
    pub distance_m: f64,
}

/// Phase 1 (Moon): evaluate the Vallado Alg. 31 truncated series at `t_tt`
/// Julian centuries of TT since J2000.0.
///
/// Phases: mean longitude polynomial + longitude series → latitude series →
/// horizontal-parallax series → parallax-to-distance conversion.
pub(crate) fn moon_ecliptic_of_date(t_tt: f64) -> MoonEcliptic {
    // Mean longitude 218.32 + 481267.8813·T (deg; rate per the 3rd-ed
    // errata) plus the periodic terms.
    let longitude_deg =
        218.32 + 481_267.881_3 * t_tt + evaluate_series(&MOON_LONGITUDE_TERMS, t_tt, f64::sin);
    let latitude_deg = evaluate_series(&MOON_LATITUDE_TERMS, t_tt, f64::sin);
    let parallax_deg = 0.9508 + evaluate_series(&MOON_PARALLAX_TERMS, t_tt, f64::cos);
    MoonEcliptic {
        longitude_rad: longitude_deg.rem_euclid(360.0) * DEG_TO_RAD,
        latitude_rad: latitude_deg * DEG_TO_RAD,
        distance_m: EARTH_EQUATORIAL_RADIUS_M / (parallax_deg * DEG_TO_RAD).sin(),
    }
}

// ---------------------------------------------------------------------
// Frame chain: ecliptic of date → MOD → GCRF.
// ---------------------------------------------------------------------

/// Phase 2: spherical ecliptic-of-date coordinates (mean equinox of date) →
/// cartesian **MOD** position, by rotating about x through the IAU-1980
/// mean obliquity (the `frames` leg — same convention as ERFA `rx`:
/// `v_mod = R1(−ε̄) · v_ecliptic`).
pub(crate) fn ecliptic_of_date_to_mod(
    longitude_rad: f64,
    latitude_rad: f64,
    distance_m: f64,
    t_tt: f64,
) -> Vector3<f64> {
    let (sin_lon, cos_lon) = longitude_rad.sin_cos();
    let (sin_lat, cos_lat) = latitude_rad.sin_cos();
    let ecliptic = Vector3::new(
        distance_m * cos_lat * cos_lon,
        distance_m * cos_lat * sin_lon,
        distance_m * sin_lat,
    );
    rotation_x(-mean_obliquity_iau1980(t_tt)) * ecliptic
}

/// Phase 3: **MOD → GCRF** via the transpose of the IAU-76 precession leg
/// (the `frames` provider's GCRF→MOD rotation is exactly this matrix, so
/// the chain stays consistent with every other transform in the crate —
/// asserted in the tests below).
pub(crate) fn mod_to_gcrf(position_mod: &Vector3<f64>, t_tt: f64) -> Vector3<f64> {
    precession_matrix_iau76(t_tt).transpose() * position_mod
}

/// Geocentric Sun position in the **MOD** frame (metres) at `t_tt` Julian
/// centuries of TT since J2000.0 — the pre-precession intermediate, exposed
/// for tests and of-date consumers.
pub(crate) fn sun_position_mod(t_tt: f64) -> Vector3<f64> {
    let sun = sun_ecliptic_of_date(t_tt);
    ecliptic_of_date_to_mod(sun.longitude_rad, 0.0, sun.distance_m, t_tt)
}

/// Geocentric Moon position in the **MOD** frame (metres) at `t_tt` Julian
/// centuries of TT since J2000.0 — the frame Vallado's Example 5-3 answer
/// is expressed in (spot-tested below).
pub(crate) fn moon_position_mod(t_tt: f64) -> Vector3<f64> {
    let moon = moon_ecliptic_of_date(t_tt);
    ecliptic_of_date_to_mod(moon.longitude_rad, moon.latitude_rad, moon.distance_m, t_tt)
}

/// Geocentric geometric Sun position in **GCRF**, metres.
///
/// Meeus Ch. 25 low-accuracy series (~0.01°-class; see the module docs for
/// provenance and the golden fixtures for measured accuracy), evaluated at
/// the epoch's TT and carried ecliptic-of-date → MOD → GCRF through the
/// `frames` legs.
///
/// # Example
///
/// ```
/// use thresh_core::orbital::ephemeris::{sun_position_gcrf, ASTRONOMICAL_UNIT_M};
/// use thresh_core::time::Epoch;
///
/// let epoch = Epoch::from_gregorian_utc(2024, 1, 1, 0, 0, 0, 0);
/// let r_sun = sun_position_gcrf(&epoch);
/// // The Earth–Sun distance stays within ~1.7% of one AU (orbital
/// // eccentricity ≈ 0.0167).
/// assert!((r_sun.norm() / ASTRONOMICAL_UNIT_M - 1.0).abs() < 0.02);
/// ```
pub fn sun_position_gcrf(epoch: &Epoch) -> Vector3<f64> {
    let t_tt = julian_centuries_tt(epoch);
    mod_to_gcrf(&sun_position_mod(t_tt), t_tt)
}

/// Geocentric geometric Moon position in **GCRF**, metres.
///
/// Vallado Algorithm 31 truncated series (~0.3°-class; see the module docs
/// for provenance and the golden fixtures for measured accuracy), evaluated
/// at the epoch's TT and carried ecliptic-of-date → MOD → GCRF through the
/// `frames` legs.
///
/// # Example
///
/// ```
/// use thresh_core::orbital::ephemeris::moon_position_gcrf;
/// use thresh_core::time::Epoch;
///
/// let epoch = Epoch::from_gregorian_utc(2024, 1, 1, 0, 0, 0, 0);
/// let r_moon = moon_position_gcrf(&epoch);
/// // Lunar distance stays between perigee and apogee bounds.
/// assert!(r_moon.norm() > 3.56e8 && r_moon.norm() < 4.07e8);
/// ```
pub fn moon_position_gcrf(epoch: &Epoch) -> Vector3<f64> {
    let t_tt = julian_centuries_tt(epoch);
    mod_to_gcrf(&moon_position_mod(t_tt), t_tt)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::frames::{Frame, rotation_between};

    /// Kilometres to metres.
    const KM: f64 = 1_000.0;

    /// Scaled-integer column sum for the tripwire checksums: exact for the
    /// fixed-decimal source values regardless of f64 representation.
    fn scaled_sum<'a>(values: impl Iterator<Item = &'a f64>, scale: f64) -> i64 {
        values.map(|v| (v * scale).round() as i64).sum()
    }

    // ── Transcription tripwires (nutation1980.rs pattern) ───────────────

    /// Checksums computed from the fetched sources this session (see the
    /// module docs): any dropped, duplicated, or mistyped coefficient moves
    /// at least one column sum. Scales match each column's printed decimal
    /// places, so the rounded sums are exact.
    #[test]
    fn series_table_checksums_match_fetched_sources() {
        let amp = |t: &'static [SeriesTerm]| t.iter().map(|x| &x.amplitude_deg);
        let phase = |t: &'static [SeriesTerm]| t.iter().map(|x| &x.phase_deg);
        let rate = |t: &'static [SeriesTerm]| t.iter().map(|x| &x.rate_deg_per_century);

        // moon.m longitude series: Σamp = 5.59, Σphase = 1443.8,
        // Σrate = 2 911 198.50.
        assert_eq!(scaled_sum(amp(&MOON_LONGITUDE_TERMS), 100.0), 559);
        assert_eq!(scaled_sum(phase(&MOON_LONGITUDE_TERMS), 10.0), 14_438);
        assert_eq!(scaled_sum(rate(&MOON_LONGITUDE_TERMS), 100.0), 291_119_850);

        // moon.m latitude series: Σamp = 4.96, Σphase = 857.4,
        // Σrate = 1 042 273.88.
        assert_eq!(scaled_sum(amp(&MOON_LATITUDE_TERMS), 100.0), 496);
        assert_eq!(scaled_sum(phase(&MOON_LATITUDE_TERMS), 10.0), 8_574);
        assert_eq!(scaled_sum(rate(&MOON_LATITUDE_TERMS), 100.0), 104_227_388);

        // moon.m parallax series: Σamp = 0.0719, Σphase = 899.7,
        // Σrate = 1 908 795.40.
        assert_eq!(scaled_sum(amp(&MOON_PARALLAX_TERMS), 10_000.0), 719);
        assert_eq!(scaled_sum(phase(&MOON_PARALLAX_TERMS), 10.0), 8_997);
        assert_eq!(scaled_sum(rate(&MOON_PARALLAX_TERMS), 100.0), 190_879_540);

        // Meeus equation of center (solar.go): Σc₀ = 1.934884,
        // Σc₁ = −0.004918, Σc₂ = −0.000014; multiples 1+2+3.
        let column = |i: usize| SUN_EQUATION_OF_CENTER.iter().map(move |(_, c)| &c[i]);
        assert_eq!(scaled_sum(column(0), 1e6), 1_934_884);
        assert_eq!(scaled_sum(column(1), 1e6), -4_918);
        assert_eq!(scaled_sum(column(2), 1e6), -14);
        let multiples: f64 = SUN_EQUATION_OF_CENTER.iter().map(|(k, _)| k).sum();
        assert_eq!(multiples, 6.0);
    }

    /// The leading term of each table, spot-checked verbatim against the
    /// fetched sources (moon.m rows 1; solar.go equation-of-center row 1).
    #[test]
    fn largest_terms_match_fetched_sources() {
        assert_eq!(MOON_LONGITUDE_TERMS[0], term(6.29, 134.9, 477_198.85));
        assert_eq!(MOON_LATITUDE_TERMS[0], term(5.13, 93.3, 483_202.03));
        assert_eq!(MOON_PARALLAX_TERMS[0], term(0.0518, 134.9, 477_198.85));
        assert_eq!(
            SUN_EQUATION_OF_CENTER[0],
            (1.0, [1.914_602, -0.004_817, -0.000_014])
        );
    }

    // ── Per-phase helpers ────────────────────────────────────────────────

    /// `evaluate_series` forms `amp · trig(phase + rate·T)` per term: a
    /// rate-free term reproduces `amp·sin(phase)` at any T, and a pure-rate
    /// term picks up the argument at the expected T.
    #[test]
    fn evaluate_series_forms_phase_plus_rate_arguments() {
        let constant = [term(2.0, 90.0, 0.0)];
        assert!((evaluate_series(&constant, 0.7, f64::sin) - 2.0).abs() < 1e-15);
        let rate_only = [term(1.0, 0.0, 360.0)];
        // At T = 0.25 the argument is 90°.
        assert!((evaluate_series(&rate_only, 0.25, f64::sin) - 1.0).abs() < 1e-15);
        assert!(evaluate_series(&rate_only, 0.25, f64::cos).abs() < 1e-15);
    }

    /// `horner3` is the quadratic it claims to be.
    #[test]
    fn horner3_evaluates_quadratic() {
        assert_eq!(horner3(2.0, [1.0, 10.0, 100.0]), 421.0);
    }

    // ── Fetched published spot values ────────────────────────────────────

    /// Spec "Analytic Sun and Moon ephemerides with measured accuracy" —
    /// Meeus Example 25.a (JDE 2448908.5 TD = 1992 Oct 13.0): true
    /// longitude ☉ = 199.90987°, R = 0.99766 AU, e = 0.016711668
    /// (expected values from the fetched `solar_test.go` / book example).
    /// Implementation reproduces them to ~3e-6° / 2e-6 AU; tolerances cover
    /// the source's print rounding.
    #[test]
    fn sun_series_matches_meeus_example_25a() {
        let epoch = Epoch::from_jde_tt(2_448_908.5);
        let t_tt = julian_centuries_tt(&epoch);
        assert!(
            (t_tt - -0.072_183_436).abs() < 1e-9,
            "T mismatch: {t_tt}, Meeus prints -0.072183436"
        );

        let sun = sun_ecliptic_of_date(t_tt);
        let longitude_deg = sun.longitude_rad.to_degrees();
        assert!(
            (longitude_deg - 199.909_87).abs() < 1e-4,
            "true longitude {longitude_deg} != 199.90987"
        );
        let distance_au = sun.distance_m / ASTRONOMICAL_UNIT_M;
        assert!(
            (distance_au - 0.997_66).abs() < 1e-5,
            "radius vector {distance_au} != 0.99766 AU"
        );
    }

    /// Spec "Analytic Sun and Moon ephemerides with measured accuracy" —
    /// Vallado Example 5-3 (jd 2449470.5 = 1994 Apr 28.0, input JD from the
    /// fetched `ex5_3.m`): mean-of-date equatorial position
    /// (−134240.626, −311571.590, −126693.785) km per the fetched 3rd-ed
    /// consolidated errata (pp. 290–291 entry). Measured reproduction is
    /// 0.35 m; the 2 m tolerance covers the errata's 3-decimal-km print
    /// rounding (≤ 0.87 m) plus the ERFA-vs-Vallado mean-obliquity
    /// truncation (~0.7 m at lunar distance).
    #[test]
    fn moon_series_matches_vallado_example_5_3() {
        let epoch = Epoch::from_jde_tt(2_449_470.5);
        let t_tt = julian_centuries_tt(&epoch);
        let r_mod = moon_position_mod(t_tt);
        let expected = Vector3::new(-134_240.626 * KM, -311_571.590 * KM, -126_693.785 * KM);
        assert!(
            (r_mod - expected).norm() < 2.0,
            "Ex 5-3 mismatch: {} m",
            (r_mod - expected).norm()
        );
    }

    /// The truncated Moon series stays within its documented ~0.3° class of
    /// the full-series truth: Meeus Example 47.a (1992 Apr 12.0 TD,
    /// JDE 2448724.5) per the fetched `moonposition_test.go` —
    /// λ = 133.162655°, β = −3.229126°, Δ = 368409.7 km. Measured deltas:
    /// +0.0823° / −0.0782° / −340 km; tolerances are ~2× those.
    #[test]
    fn moon_series_within_class_of_meeus_full_series_example_47a() {
        let epoch = Epoch::from_jde_tt(2_448_724.5);
        let moon = moon_ecliptic_of_date(julian_centuries_tt(&epoch));
        let longitude_deg = moon.longitude_rad.to_degrees();
        let latitude_deg = moon.latitude_rad.to_degrees();
        assert!(
            (longitude_deg - 133.162_655).abs() < 0.15,
            "longitude {longitude_deg} vs full-series 133.162655"
        );
        assert!(
            (latitude_deg - -3.229_126).abs() < 0.15,
            "latitude {latitude_deg} vs full-series -3.229126"
        );
        assert!(
            (moon.distance_m - 368_409.7 * KM).abs() < 700.0 * KM,
            "distance {} km vs full-series 368409.7 km",
            moon.distance_m / KM
        );
    }

    // ── Structural bounds ────────────────────────────────────────────────

    /// Sun distance stays within 2% of one AU over a daily 10-year sweep,
    /// and the ±1.67% eccentricity signal is actually visible (perihelion
    /// (1−e)·AU ≈ 0.9833, aphelion ≈ 1.0167).
    #[test]
    fn sun_distance_within_two_percent_of_an_au() {
        let base = Epoch::from_gregorian_utc(2020, 1, 1, 0, 0, 0, 0);
        let mut min_au = f64::MAX;
        let mut max_au = f64::MIN;
        for day in 0..3_653 {
            let r_au =
                sun_position_gcrf(&(base + f64::from(day) * 86_400.0)).norm() / ASTRONOMICAL_UNIT_M;
            assert!(
                (0.98..=1.02).contains(&r_au),
                "day {day}: {r_au} AU outside 1 AU ± 2%"
            );
            min_au = min_au.min(r_au);
            max_au = max_au.max(r_au);
        }
        assert!(min_au < 0.984, "perihelion not reached: {min_au}");
        assert!(max_au > 1.016, "aphelion not reached: {max_au}");
    }

    /// Moon distance and ecliptic latitude stay in physical bounds over a
    /// weekly 50-year sweep: distance within the 356–407 Mm perigee/apogee
    /// band, |β| under the series amplitude sum 5.86°, and the ±5°
    /// inclination excursion actually exercised.
    #[test]
    fn moon_distance_and_latitude_stay_in_physical_bounds() {
        let base = Epoch::from_gregorian_utc(2000, 1, 1, 0, 0, 0, 0);
        let mut max_lat_deg = 0.0_f64;
        let mut min_r = f64::MAX;
        let mut max_r = f64::MIN;
        for week in 0..2_609 {
            let epoch = base + f64::from(week) * 7.0 * 86_400.0;
            let moon = moon_ecliptic_of_date(julian_centuries_tt(&epoch));
            assert!(
                (3.56e8..=4.07e8).contains(&moon.distance_m),
                "week {week}: {} m outside the lunar distance band",
                moon.distance_m
            );
            let lat_deg = moon.latitude_rad.to_degrees().abs();
            assert!(lat_deg < 5.9, "week {week}: |β| = {lat_deg}° exceeds bound");
            max_lat_deg = max_lat_deg.max(lat_deg);
            min_r = min_r.min(moon.distance_m);
            max_r = max_r.max(moon.distance_m);
        }
        assert!(
            max_lat_deg > 4.5,
            "latitude never exercised: {max_lat_deg}°"
        );
        assert!(min_r < 3.65e8, "perigee never approached: {min_r} m");
        assert!(max_r > 4.00e8, "apogee never approached: {max_r} m");
    }

    /// The Sun's ecliptic latitude is zero by construction; rotating the
    /// GCRF result back through the chain recovers a vector in the ecliptic
    /// plane — validating the direction and order of both rotations.
    #[test]
    fn sun_ecliptic_latitude_is_zero_through_the_chain() {
        for &(year, month) in &[(2010, 3), (2024, 1), (2030, 9)] {
            let epoch = Epoch::from_gregorian_utc(year, month, 15, 6, 0, 0, 0);
            let t_tt = julian_centuries_tt(&epoch);
            let r_gcrf = sun_position_gcrf(&epoch);
            let r_mod = precession_matrix_iau76(t_tt) * r_gcrf;
            let r_ecliptic = rotation_x(mean_obliquity_iau1980(t_tt)) * r_mod;
            assert!(
                r_ecliptic.z.abs() < 1e-9 * r_ecliptic.norm(),
                "{year}-{month}: ecliptic z = {} m",
                r_ecliptic.z
            );
        }
    }

    /// The module's MOD→GCRF leg agrees with the `frames` provider's
    /// rotation for the same pair — the ephemerides ride the same reduction
    /// chain as every other transform in the crate (spec: "delivered in
    /// GCRF via the reference-frame machinery").
    #[test]
    fn gcrf_chain_matches_frames_provider() {
        let epoch = Epoch::from_gregorian_utc(2024, 3, 20, 12, 0, 0, 0);
        let t_tt = julian_centuries_tt(&epoch);
        let rotation = rotation_between(Frame::Mod, Frame::Gcrf, &epoch);

        let moon_mod = moon_position_mod(t_tt);
        let via_provider = rotation.rotate_position(&moon_mod);
        let via_module = moon_position_gcrf(&epoch);
        assert!((via_module - via_provider).norm() < 1e-9 * via_provider.norm());

        let sun_mod = sun_position_mod(t_tt);
        let via_provider = rotation.rotate_position(&sun_mod);
        let via_module = sun_position_gcrf(&epoch);
        assert!((via_module - via_provider).norm() < 1e-9 * via_provider.norm());
    }

    /// GCRF and MOD positions differ by accumulated precession —
    /// arcminute-scale for modern epochs, so the rotation is doing real
    /// work (guards against an accidentally-identity precession leg).
    #[test]
    fn precession_leg_visibly_rotates_modern_epochs() {
        let epoch = Epoch::from_gregorian_utc(2024, 6, 1, 0, 0, 0, 0);
        let t_tt = julian_centuries_tt(&epoch);
        let moon_mod = moon_position_mod(t_tt);
        let moon_gcrf = moon_position_gcrf(&epoch);
        let angle = (moon_mod.dot(&moon_gcrf) / (moon_mod.norm() * moon_gcrf.norm()))
            .clamp(-1.0, 1.0)
            .acos();
        // ~0.3° of accumulated precession over 24.4 years.
        assert!(
            angle > 1e-3 && angle < 1e-2,
            "precession angle {angle} rad outside the expected band"
        );
    }
}
