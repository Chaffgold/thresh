//! Truncated EGM96 spherical-harmonic gravity (12×12), evaluated in the
//! Earth-fixed (ITRF) frame.
//!
//! The public entry point is [`egm96_acceleration_itrf`]: it takes an ITRF
//! position and returns the **total** ITRF gravitational acceleration
//! (central term plus the selected harmonics), mirroring the contract of
//! `gravity::j2_acceleration`. Frame rotation (GCRF ↔ ITRF through the
//! `FrameProvider`) is the force-configuration layer's job, not this
//! module's — everything here is Earth-fixed in, Earth-fixed out.
//!
//! # Provenance (coefficients and constants)
//!
//! The 88 normalized coefficient pairs (degree 2..=12, all orders) were
//! transcribed mechanically (script-parsed, not hand-copied) from the NASA
//! GSFC/NIMA EGM96 distribution file `egm96_to360.ascii` (original
//! distribution: NASA CDDIS, <https://cddis.nasa.gov/926/egm96/>), fetched
//! 2026-07-16 from the Sandia National Laboratories mirror
//! <https://github.com/sandialabs/phoenix> (file
//! `phoenix-astrodynamics/src/main/resources/gov/sandia/phoenix/solarsystem/egm96_to360.ascii`).
//! All 88 rows were cross-validated bit-for-bit against an independently
//! hosted copy, the ICGEM `EGM96.gfc` model file
//! (<https://icgem.gfz-potsdam.de/>), fetched the same session. Reference:
//! Lemoine et al., *The Development of the Joint NASA GSFC and NIMA
//! Geopotential Model EGM96*, NASA/TP-1998-206861 (1998).
//!
//! [`EGM96_MU`] and [`EGM96_RADIUS`] come from the ICGEM `EGM96.gfc` header
//! (`earth_gravity_constant 0.3986004415E+15`, `radius 0.6378136300E+07`,
//! `tide_system tide_free`).
//!
//! # Constant-set discipline
//!
//! EGM96's GM and scale radius differ slightly from WGS-84's
//! (`GravityModel::EARTH_WGS84`): 3.986004415e14 vs 3.986004418e14 m³/s²
//! and 6378136.3 vs 6378137.0 m. The harmonic evaluation always uses the
//! EGM96 set — coefficients are only meaningful with the constants they
//! were estimated with. Consistency identities against the closed-form
//! zonal accelerations therefore evaluate **both** sides with
//! [`egm96_gravity_model`] (see the `c20_only_reproduces_analytic_j2`
//! test), never with a mixed constant set.
//!
//! # Evaluation method
//!
//! Fully normalized associated Legendre functions via the standard
//! forward-column recursion (Holmes & Featherstone 2002 lineage; the
//! `a_nm`/`b_nm` coefficient expressions as given in Li et al., *Realizing
//! the Calculation of a Fully Normalized Associated Legendre Function Based
//! on an FPGA*, Sensors 24(22):7358, 2024,
//! <https://pmc.ncbi.nlm.nih.gov/articles/PMC11598212/>, and the sectoral
//! seed factor √((2m+1)/(2m)) as also documented for the normalized
//! Montenbruck & Gill §3.2 evaluation in
//! <https://www.space.t.u-tokyo.ac.jp/s2e-documents/Specifications/Disturbance/Spec_GeoPotential.html>),
//! then spherical potential partials assembled to Cartesian by the chain
//! rule. The recursion, its derivative, and the assembly are each verified
//! by executable identities in the tests: the addition theorem
//! Σₘ P̄ₙₘ² = 2n+1, agreement with an independent unnormalized-recursion ×
//! explicit-normalization path, central-difference derivative checks, and
//! acceleration ≡ numeric gradient of the potential.

use nalgebra::Vector3;

use super::gravity::GravityModel;

/// EGM96 gravitational parameter GM (m³/s²).
///
/// From the ICGEM `EGM96.gfc` header (`earth_gravity_constant
/// 0.3986004415E+15`), fetched 2026-07-16. Note this is *not* the WGS-84
/// value used by `GravityModel::EARTH_WGS84` (3.986004418e14).
pub const EGM96_MU: f64 = 3.986_004_415e14;

/// EGM96 reference (scaling) radius a (m).
///
/// From the ICGEM `EGM96.gfc` header (`radius 0.6378136300E+07`), fetched
/// 2026-07-16. Note this is *not* the WGS-84 equatorial radius (6378137.0).
pub const EGM96_RADIUS: f64 = 6_378_136.3;

/// Highest degree and order embedded in [`egm96_acceleration_itrf`]'s
/// coefficient table. Requests beyond it are clamped (documented there).
pub const EGM96_MAX_DEGREE: u32 = 12;

/// Table dimensions: indices run 0..=12 (degree-major).
const NMAX: usize = 12;
const DIM: usize = NMAX + 1;

/// A per-evaluation table of P̄ₙₘ (or dP̄ₙₘ/dt) values indexed `[n][m]`.
type LegendreTable = [[f64; DIM]; DIM];

/// Fully normalized EGM96 coefficients to degree and order 12, one row per
/// coefficient pair: `(n, m, C̄ₙₘ, S̄ₙₘ)` (dimensionless, tide-free system).
///
/// Row order matches the source file (degree-major, order-minor). See the
/// module docs for provenance; the transcription tripwires live in
/// `table_checksums_match_fetched_source`.
pub(crate) const EGM96_COEFFICIENTS: [(u8, u8, f64, f64); 88] = [
    // degree 2
    (2, 0, -0.000484165371736, 0.0),
    (2, 1, -1.86987635955e-10, 1.19528012031e-09),
    (2, 2, 2.43914352398e-06, -1.40016683654e-06),
    // degree 3
    (3, 0, 9.57254173792e-07, 0.0),
    (3, 1, 2.02998882184e-06, 2.48513158716e-07),
    (3, 2, 9.04627768605e-07, -6.19025944205e-07),
    (3, 3, 7.21072657057e-07, 1.41435626958e-06),
    // degree 4
    (4, 0, 5.39873863789e-07, 0.0),
    (4, 1, -5.36321616971e-07, -4.73440265853e-07),
    (4, 2, 3.50694105785e-07, 6.6267157254e-07),
    (4, 3, 9.90771803829e-07, -2.00928369177e-07),
    (4, 4, -1.88560802735e-07, 3.08853169333e-07),
    // degree 5
    (5, 0, 6.8532347563e-08, 0.0),
    (5, 1, -6.21012128528e-08, -9.44226127525e-08),
    (5, 2, 6.52438297612e-07, -3.23349612668e-07),
    (5, 3, -4.51955406071e-07, -2.14847190624e-07),
    (5, 4, -2.95301647654e-07, 4.96658876769e-08),
    (5, 5, 1.74971983203e-07, -6.69384278219e-07),
    // degree 6
    (6, 0, -1.49957994714e-07, 0.0),
    (6, 1, -7.60879384947e-08, 2.62890545501e-08),
    (6, 2, 4.81732442832e-08, -3.73728201347e-07),
    (6, 3, 5.71730990516e-08, 9.02694517163e-09),
    (6, 4, -8.62142660109e-08, -4.71408154267e-07),
    (6, 5, -2.6713332549e-07, -5.36488432483e-07),
    (6, 6, 9.67616121092e-09, -2.37192006935e-07),
    // degree 7
    (7, 0, 9.0978937145e-08, 0.0),
    (7, 1, 2.79872910488e-07, 9.54336911867e-08),
    (7, 2, 3.29743816488e-07, 9.30667596042e-08),
    (7, 3, 2.50398657706e-07, -2.17198608738e-07),
    (7, 4, -2.75114355257e-07, -1.23800392323e-07),
    (7, 5, 1.93765507243e-09, 1.77377719872e-08),
    (7, 6, -3.58856860645e-07, 1.51789817739e-07),
    (7, 7, 1.09185148045e-09, 2.44415707993e-08),
    // degree 8
    (8, 0, 4.96711667324e-08, 0.0),
    (8, 1, 2.33422047893e-08, 5.90060493411e-08),
    (8, 2, 8.02978722615e-08, 6.54175425859e-08),
    (8, 3, -1.91877757009e-08, -8.63454445021e-08),
    (8, 4, -2.44600105471e-07, 7.00233016934e-08),
    (8, 5, -2.55352403037e-08, 8.91462164788e-08),
    (8, 6, -6.57361610961e-08, 3.09238461807e-07),
    (8, 7, 6.72811580072e-08, 7.47440473633e-08),
    (8, 8, -1.24092493016e-07, 1.20533165603e-07),
    // degree 9
    (9, 0, 2.76714300853e-08, 0.0),
    (9, 1, 1.43387502749e-07, 2.16834947618e-08),
    (9, 2, 2.22288318564e-08, -3.22196647116e-08),
    (9, 3, -1.60811502143e-07, -7.42287409462e-08),
    (9, 4, -9.00179225336e-09, 1.94666779475e-08),
    (9, 5, -1.66165092924e-08, -5.41113191483e-08),
    (9, 6, 6.26941938248e-08, 2.22903525945e-07),
    (9, 7, -1.18366323475e-07, -9.65152667886e-08),
    (9, 8, 1.88436022794e-07, -3.08566220421e-09),
    (9, 9, -4.77475386132e-08, 9.66412847714e-08),
    // degree 10
    (10, 0, 5.26222488569e-08, 0.0),
    (10, 1, 8.35115775652e-08, -1.31314331796e-07),
    (10, 2, -9.42413882081e-08, -5.1579165739e-08),
    (10, 3, -6.89895048176e-09, -1.53768828694e-07),
    (10, 4, -8.40764549716e-08, -7.92806255331e-08),
    (10, 5, -4.93395938185e-08, -5.05370221897e-08),
    (10, 6, -3.75885236598e-08, -7.95667053872e-08),
    (10, 7, 8.11460540925e-09, -3.36629641314e-09),
    (10, 8, 4.04927981694e-08, -9.18705975922e-08),
    (10, 9, 1.25491334939e-07, -3.76516222392e-08),
    (10, 10, 1.00538634409e-07, -2.4014844952e-08),
    // degree 11
    (11, 0, -5.09613707522e-08, 0.0),
    (11, 1, 1.51687209933e-08, -2.68604146166e-08),
    (11, 2, 1.86309749878e-08, -9.90693862047e-08),
    (11, 3, -3.09871239854e-08, -1.4813180426e-07),
    (11, 4, -3.89580205051e-08, -6.3666651198e-08),
    (11, 5, 3.77848029452e-08, 4.94736238169e-08),
    (11, 6, -1.18676592395e-09, 3.44769584593e-08),
    (11, 7, 4.11565188074e-09, -8.98252808977e-08),
    (11, 8, -5.984108413e-09, 2.43989612237e-08),
    (11, 9, -3.14231072723e-08, 4.17731829829e-08),
    (11, 10, -5.21882681927e-08, -1.83364561788e-08),
    (11, 11, 4.60344448746e-08, -6.96662308185e-08),
    // degree 12
    (12, 0, 3.77252636558e-08, 0.0),
    (12, 1, -5.40654977836e-08, -4.35675748979e-08),
    (12, 2, 1.42979642253e-08, 3.20975937619e-08),
    (12, 3, 3.93995876403e-08, 2.44264863505e-08),
    (12, 4, -6.86908127934e-08, 4.15081109011e-09),
    (12, 5, 3.0941112873e-08, 7.82536279033e-09),
    (12, 6, 3.41523275208e-09, 3.91765484449e-08),
    (12, 7, -1.86909958587e-08, 3.56131849382e-08),
    (12, 8, -2.53769398865e-08, 1.69361024629e-08),
    (12, 9, 4.22880630662e-08, 2.52692598301e-08),
    (12, 10, -6.17619654902e-09, 3.08375794212e-08),
    (12, 11, 1.12502994122e-08, -6.37946501558e-09),
    (12, 12, -2.4953260739e-09, -1.117806019e-08),
];

/// Look up the fully normalized EGM96 coefficient pair `(C̄ₙₘ, S̄ₙₘ)` for
/// the given degree and order, or `None` outside the embedded 2..=12 table.
pub fn egm96_coefficient(degree: u32, order: u32) -> Option<(f64, f64)> {
    EGM96_COEFFICIENTS
        .iter()
        .find(|&&(n, m, _, _)| u32::from(n) == degree && u32::from(m) == order)
        .map(|&(_, _, c, s)| (c, s))
}

/// The conventional (unnormalized) zonal coefficient Jₙ derived from the
/// embedded table: Jₙ = −√(2n+1) · C̄ₙ₀ (the m = 0 normalization factor is
/// √(2n+1); see e.g. the normalization definition in the module-doc
/// sources). `None` outside degree 2..=12.
///
/// `egm96_jn(2)` ≈ 1.0826266835531513e-3, the familiar J2 (tide-free).
pub fn egm96_jn(degree: u32) -> Option<f64> {
    egm96_coefficient(degree, 0).map(|(c, _)| -f64::from(2 * degree + 1).sqrt() * c)
}

/// The [`GravityModel`] carrying EGM96's own constant set: [`EGM96_MU`],
/// [`EGM96_RADIUS`], and J2 derived from the embedded C̄₂₀.
///
/// Use this — not `GravityModel::EARTH_WGS84` — when comparing the harmonic
/// evaluation against the closed-form zonal accelerations, so both sides
/// share one constant set (see the module docs on constant-set discipline).
pub fn egm96_gravity_model() -> GravityModel {
    GravityModel {
        mu: EGM96_MU,
        j2: egm96_jn(2).expect("degree 2 is always in the embedded table"),
        equatorial_radius: EGM96_RADIUS,
    }
}

/// Positions closer to the polar axis than this fraction of the radius are
/// evaluated on a floored meridian ring (see [`egm96_acceleration_itrf`]).
const POLAR_AXIS_FLOOR: f64 = 1e-9;

/// Total gravitational acceleration (m/s², ITRF) at an ITRF position from
/// the EGM96 model truncated to `degree` × `order`: the central μ/r² term
/// plus all harmonics with n ≤ `degree` and m ≤ `order`.
///
/// * Evaluated with the EGM96 constant set ([`EGM96_MU`], [`EGM96_RADIUS`])
///   — see the module docs for why the constants and coefficients travel
///   together.
/// * `degree` is clamped to [`EGM96_MAX_DEGREE`] and `order` to `degree`
///   (the table stops at 12×12; larger requests evaluate the full table).
/// * `(degree, order) = (2, 0)` is exactly the C̄₂₀ (J2) field;
///   `(degree, 0)` is zonal-only; `(0, 0)` or `(1, _)` is pure two-body.
/// * The spherical-gradient formulation is singular on the polar axis, so
///   positions within 1e-9 × r of the axis (`POLAR_AXIS_FLOOR`) are
///   evaluated at an off-axis distance floored to that value (≈ 6 mm at Earth's
///   surface). The result stays finite; the error is bounded by the
///   (tiny) tesseral contribution itself. Zonal terms are unaffected.
pub fn egm96_acceleration_itrf(pos_itrf: &Vector3<f64>, degree: u32, order: u32) -> Vector3<f64> {
    let degree = degree.min(EGM96_MAX_DEGREE) as usize;
    let order = (order.min(EGM96_MAX_DEGREE) as usize).min(degree);
    let geometry = SphericalGeometry::new(pos_itrf);
    let (p, dp) = normalized_legendre(geometry.t, geometry.u);
    let partials = potential_partials(&geometry, degree, order, &p, &dp);
    assemble_cartesian(&geometry, &partials)
}

/// Spherical geometry of an ITRF position, computed once per evaluation.
struct SphericalGeometry {
    x: f64,
    y: f64,
    z: f64,
    r: f64,
    r2: f64,
    /// Distance from the polar axis, floored at [`POLAR_AXIS_FLOOR`] · r.
    rho: f64,
    /// sin(geocentric latitude) = z/r — the Legendre argument.
    t: f64,
    /// cos(geocentric latitude) = ρ/r.
    u: f64,
    /// Longitude λ = atan2(y, x).
    lambda: f64,
}

impl SphericalGeometry {
    fn new(pos: &Vector3<f64>) -> Self {
        let (x, y, z) = (pos.x, pos.y, pos.z);
        let r2 = x * x + y * y + z * z;
        let r = r2.sqrt();
        let rho = (x * x + y * y).sqrt().max(POLAR_AXIS_FLOOR * r);
        Self {
            x,
            y,
            z,
            r,
            r2,
            rho,
            t: z / r,
            u: rho / r,
            lambda: y.atan2(x),
        }
    }
}

/// Fully normalized associated Legendre functions P̄ₙₘ(t) and their
/// derivatives dP̄ₙₘ/dt for all n, m ≤ [`NMAX`], where t = sin(latitude)
/// and u = cos(latitude) = √(1 − t²).
///
/// Forward-column recursion (module docs cite the fetched sources): seeds
/// P̄₀₀ = 1, P̄₁₀ = √3·t, P̄₁₁ = √3·u; sectoral and column phases below.
/// Derivatives propagate by differentiating each recurrence in t — no
/// additional coefficient formulas — and are verified against central
/// differences in the tests. The derivative seeds divide by u, so the
/// caller keeps u off zero (the polar-axis floor).
fn normalized_legendre(t: f64, u: f64) -> (LegendreTable, LegendreTable) {
    let mut p: LegendreTable = [[0.0; DIM]; DIM];
    let mut dp: LegendreTable = [[0.0; DIM]; DIM];
    let sqrt3 = 3.0_f64.sqrt();
    p[0][0] = 1.0;
    p[1][0] = sqrt3 * t;
    dp[1][0] = sqrt3;
    p[1][1] = sqrt3 * u;
    dp[1][1] = -sqrt3 * t / u;
    seed_sectorals(t, u, &mut p, &mut dp);
    fill_columns(t, &mut p, &mut dp);
    (p, dp)
}

/// Sectoral phase: P̄ₘₘ = √((2m+1)/(2m)) · u · P̄ₘ₋₁,ₘ₋₁ for m ≥ 2, and
/// its t-derivative (du/dt = −t/u).
fn seed_sectorals(t: f64, u: f64, p: &mut LegendreTable, dp: &mut LegendreTable) {
    for m in 2..DIM {
        let c = ((2 * m + 1) as f64 / (2 * m) as f64).sqrt();
        p[m][m] = c * u * p[m - 1][m - 1];
        dp[m][m] = c * (u * dp[m - 1][m - 1] - (t / u) * p[m - 1][m - 1]);
    }
}

/// Column phase: P̄ₙₘ = aₙₘ·t·P̄ₙ₋₁,ₘ − bₙₘ·P̄ₙ₋₂,ₘ with
/// aₙₘ = √((2n−1)(2n+1)/((n−m)(n+m))) and
/// bₙₘ = √((2n+1)(n+m−1)(n−m−1)/((n−m)(n+m)(2n−3))).
///
/// At n = m+1 the bₙₘ factor is exactly zero (its (n−m−1) term), so the
/// zero-initialized P̄ₙ₋₂ entries below the diagonal never contribute.
fn fill_columns(t: f64, p: &mut LegendreTable, dp: &mut LegendreTable) {
    for m in 0..DIM {
        for n in (m + 1).max(2)..DIM {
            let (nf, mf) = (n as f64, m as f64);
            let a = ((2.0 * nf - 1.0) * (2.0 * nf + 1.0) / ((nf - mf) * (nf + mf))).sqrt();
            let b = ((2.0 * nf + 1.0) * (nf + mf - 1.0) * (nf - mf - 1.0)
                / ((nf - mf) * (nf + mf) * (2.0 * nf - 3.0)))
                .sqrt();
            p[n][m] = a * t * p[n - 1][m] - b * p[n - 2][m];
            dp[n][m] = a * (p[n - 1][m] + t * dp[n - 1][m]) - b * dp[n - 2][m];
        }
    }
}

/// Partials of the geopotential W = (μ/r)·(1 + Σₙ Σₘ (a/r)ⁿ P̄ₙₘ(t) ·
/// (C̄ₙₘ cos mλ + S̄ₙₘ sin mλ)) with respect to (r, φ, λ).
struct PotentialPartials {
    du_dr: f64,
    du_dphi: f64,
    du_dlambda: f64,
}

/// Accumulate the potential partials over the coefficient table, truncated
/// to n ≤ `degree`, m ≤ `order`. The central term seeds ∂W/∂r = −μ/r².
fn potential_partials(
    geometry: &SphericalGeometry,
    degree: usize,
    order: usize,
    p: &LegendreTable,
    dp: &LegendreTable,
) -> PotentialPartials {
    let mu_r = EGM96_MU / geometry.r;
    let mu_r2 = EGM96_MU / geometry.r2;
    let ratio_powers = scaled_radius_powers(geometry.r);
    let mut partials = PotentialPartials {
        du_dr: -mu_r2,
        du_dphi: 0.0,
        du_dlambda: 0.0,
    };
    let selected = EGM96_COEFFICIENTS
        .iter()
        .filter(|&&(n, m, _, _)| usize::from(n) <= degree && usize::from(m) <= order);
    for &(n, m, c, s) in selected {
        let (ni, mi) = (usize::from(n), usize::from(m));
        let (sin_ml, cos_ml) = (f64::from(m) * geometry.lambda).sin_cos();
        let outer = c * cos_ml + s * sin_ml;
        let arn = ratio_powers[ni];
        partials.du_dr -= mu_r2 * (f64::from(n) + 1.0) * arn * p[ni][mi] * outer;
        // dP̄/dφ = (dP̄/dt)·cos φ since t = sin φ.
        partials.du_dphi += mu_r * arn * dp[ni][mi] * geometry.u * outer;
        partials.du_dlambda += mu_r * arn * f64::from(m) * p[ni][mi] * (s * cos_ml - c * sin_ml);
    }
    partials
}

/// Powers (a/r)⁰ ..= (a/r)¹² of the EGM96 scale radius over the position
/// radius.
fn scaled_radius_powers(r: f64) -> [f64; DIM] {
    let ratio = EGM96_RADIUS / r;
    let mut powers = [1.0; DIM];
    for n in 1..DIM {
        powers[n] = powers[n - 1] * ratio;
    }
    powers
}

/// Chain-rule assembly of the Cartesian acceleration from the spherical
/// partials: a = (∂W/∂r)∇r + (∂W/∂φ)∇φ + (∂W/∂λ)∇λ with
/// ∇r = (x, y, z)/r, ∇φ = (−xz, −yz, ρ²)/(r²ρ), ∇λ = (−y, x, 0)/ρ².
/// Verified against a numeric gradient of the potential in the tests.
fn assemble_cartesian(geometry: &SphericalGeometry, partials: &PotentialPartials) -> Vector3<f64> {
    let radial = partials.du_dr / geometry.r;
    let phi = partials.du_dphi / (geometry.r2 * geometry.rho);
    let lambda = partials.du_dlambda / (geometry.rho * geometry.rho);
    Vector3::new(
        radial * geometry.x - phi * geometry.z * geometry.x - lambda * geometry.y,
        radial * geometry.y - phi * geometry.z * geometry.y + lambda * geometry.x,
        radial * geometry.z + phi * geometry.rho * geometry.rho,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::orbital::gravity::{
        EARTH_J3, EARTH_J4, j2_acceleration, j3_acceleration, j4_acceleration,
        two_body_acceleration,
    };

    /// A LEO-magnitude general position (r ≈ 6778 km, all components
    /// nonzero) used by the identity tests.
    fn leo_position() -> Vector3<f64> {
        Vector3::new(5_300_000.0, -3_400_000.0, 2_500_000.0)
    }

    fn relative_norm_error(actual: &Vector3<f64>, reference: &Vector3<f64>) -> f64 {
        (actual - reference).norm() / reference.norm()
    }

    // ── Transcription tripwires (the nutation1980 pattern) ──────────────

    /// Checksums computed from the fetched `egm96_to360.ascii` (n ≤ 12
    /// rows) by the parse script that generated the table (run this
    /// session): Σn = 726, Σm = 363, and the exact left-to-right f64 sums
    /// of the C̄ and S̄ columns. Any dropped, duplicated, or mistyped row
    /// moves at least one of these sums.
    #[test]
    fn table_checksums_match_fetched_source() {
        let n_sum: u32 = EGM96_COEFFICIENTS.iter().map(|&(n, ..)| u32::from(n)).sum();
        let m_sum: u32 = EGM96_COEFFICIENTS
            .iter()
            .map(|&(_, m, ..)| u32::from(m))
            .sum();
        let c_sum: f64 = EGM96_COEFFICIENTS.iter().map(|&(_, _, c, _)| c).sum();
        let s_sum: f64 = EGM96_COEFFICIENTS.iter().map(|&(_, _, _, s)| s).sum();
        assert_eq!(n_sum, 726);
        assert_eq!(m_sum, 363);
        assert_eq!(c_sum, -0.00047612893965528997);
        assert_eq!(s_sum, -2.963217996370348e-06);
    }

    /// First and last rows spot-checked against the fetched source lines
    /// `2 0 -0.484165371736E-03 0.0…` and
    /// `12 12 -0.249532607390E-08 -0.111780601900E-07`.
    #[test]
    fn boundary_rows_match_fetched_source() {
        assert_eq!(EGM96_COEFFICIENTS[0], (2, 0, -0.000484165371736, 0.0));
        assert_eq!(
            EGM96_COEFFICIENTS[87],
            (12, 12, -2.4953260739e-09, -1.117806019e-08)
        );
    }

    #[test]
    fn coefficient_lookup_covers_exactly_the_table() {
        assert_eq!(
            egm96_coefficient(2, 0),
            Some((-0.000484165371736, 0.0)),
            "C̄₂₀ lookup"
        );
        assert_eq!(egm96_coefficient(1, 0), None);
        assert_eq!(egm96_coefficient(13, 0), None);
        assert_eq!(egm96_coefficient(5, 6), None);
    }

    // ── Normalization contract on the Legendre recursion ────────────────

    /// For 4π (geodesy) full normalization the addition theorem gives
    /// Σₘ₌₀ⁿ P̄ₙₘ(t)² = 2n + 1 at every t — a wrong normalization factor
    /// anywhere in the recursion breaks this for some (n, t).
    #[test]
    fn legendre_satisfies_addition_theorem() {
        for &t in &[0.0_f64, 0.3, -0.7, 0.95] {
            let u = (1.0 - t * t).sqrt();
            let (p, _) = normalized_legendre(t, u);
            for (n, row) in p.iter().enumerate() {
                let sum: f64 = row.iter().take(n + 1).map(|value| value * value).sum();
                let expected = (2 * n + 1) as f64;
                assert!(
                    (sum - expected).abs() <= 1e-11 * expected,
                    "addition theorem broken at n={n}, t={t}: {sum} vs {expected}"
                );
            }
        }
    }

    /// Independent reference path: classic unnormalized three-term
    /// recurrences (no Condon–Shortley phase) times the explicit
    /// normalization factor √((2 − δ₀ₘ)(2n+1)(n−m)!/(n+m)!). Both paths
    /// must produce identical P̄ₙₘ.
    // The three-row recurrence indexes p[n][m], p[n-1][m], p[n-2][m]; plain
    // index loops are the clearest transcription of the reference formulas.
    #[allow(clippy::needless_range_loop)]
    fn unnormalized_legendre(t: f64, u: f64) -> LegendreTable {
        let mut p: LegendreTable = [[0.0; DIM]; DIM];
        p[0][0] = 1.0;
        for m in 1..DIM {
            p[m][m] = (2 * m - 1) as f64 * u * p[m - 1][m - 1];
        }
        for n in 1..DIM {
            for m in 0..n {
                let (nf, mf) = (n as f64, m as f64);
                let prev2 = if n >= m + 2 { p[n - 2][m] } else { 0.0 };
                p[n][m] =
                    ((2.0 * nf - 1.0) * t * p[n - 1][m] - (nf + mf - 1.0) * prev2) / (nf - mf);
            }
        }
        p
    }

    fn normalization_factor(n: usize, m: usize) -> f64 {
        // (n+m)!/(n−m)! as a running product of the integers n−m+1 ..= n+m.
        let mut factorial_ratio = 1.0;
        for k in (n - m + 1)..=(n + m) {
            factorial_ratio *= k as f64;
        }
        let delta = if m == 0 { 1.0 } else { 2.0 };
        (delta * (2 * n + 1) as f64 / factorial_ratio).sqrt()
    }

    #[test]
    fn normalized_recursion_matches_unnormalized_reference() {
        for &t in &[0.2_f64, -0.5, 0.9] {
            let u = (1.0 - t * t).sqrt();
            let (p, _) = normalized_legendre(t, u);
            let reference = unnormalized_legendre(t, u);
            for n in 0..DIM {
                for m in 0..=n {
                    let expected = normalization_factor(n, m) * reference[n][m];
                    assert!(
                        (p[n][m] - expected).abs() <= 1e-12 * expected.abs().max(1e-30),
                        "P̄({n},{m}) at t={t}: {} vs {}",
                        p[n][m],
                        expected
                    );
                }
            }
        }
    }

    #[test]
    fn legendre_derivative_matches_central_difference() {
        let h = 1e-6;
        for &t in &[0.1_f64, -0.6, 0.8] {
            let u = (1.0 - t * t).sqrt();
            let (_, dp) = normalized_legendre(t, u);
            let (plus, _) = normalized_legendre(t + h, (1.0 - (t + h) * (t + h)).sqrt());
            let (minus, _) = normalized_legendre(t - h, (1.0 - (t - h) * (t - h)).sqrt());
            for n in 0..DIM {
                for m in 0..=n {
                    let numeric = (plus[n][m] - minus[n][m]) / (2.0 * h);
                    assert!(
                        (dp[n][m] - numeric).abs() <= 1e-4 * numeric.abs().max(1.0),
                        "dP̄({n},{m})/dt at t={t}: {} vs {}",
                        dp[n][m],
                        numeric
                    );
                }
            }
        }
    }

    // ── The acceleration is the gradient of the potential ───────────────

    /// The geopotential itself, same truncation and conventions as
    /// `potential_partials` — test-only, for numeric differentiation.
    fn egm96_potential_itrf(pos: &Vector3<f64>, degree: usize, order: usize) -> f64 {
        let geometry = SphericalGeometry::new(pos);
        let (p, _) = normalized_legendre(geometry.t, geometry.u);
        let powers = scaled_radius_powers(geometry.r);
        let mu_r = EGM96_MU / geometry.r;
        let mut potential = mu_r;
        let selected = EGM96_COEFFICIENTS
            .iter()
            .filter(|&&(n, m, _, _)| usize::from(n) <= degree && usize::from(m) <= order);
        for &(n, m, c, s) in selected {
            let (ni, mi) = (usize::from(n), usize::from(m));
            let (sin_ml, cos_ml) = (f64::from(m) * geometry.lambda).sin_cos();
            potential += mu_r * powers[ni] * p[ni][mi] * (c * cos_ml + s * sin_ml);
        }
        potential
    }

    #[test]
    fn acceleration_is_gradient_of_potential() {
        let pos = leo_position();
        let h = 100.0;
        let acc = egm96_acceleration_itrf(&pos, 12, 12);
        for i in 0..3 {
            let mut plus = pos;
            let mut minus = pos;
            plus[i] += h;
            minus[i] -= h;
            let numeric = (egm96_potential_itrf(&plus, 12, 12)
                - egm96_potential_itrf(&minus, 12, 12))
                / (2.0 * h);
            assert!(
                (acc[i] - numeric).abs() <= 1e-7 * numeric.abs(),
                "component {i}: {} vs {}",
                acc[i],
                numeric
            );
        }
    }

    // ── Consistency identities against the closed-form zonals ───────────

    /// Spec scenario "C20-only harmonics reproduce analytic J2": the
    /// harmonic evaluation restricted to C̄₂₀ must match the closed-form
    /// `j2_acceleration` — both sides on the EGM96 constant set — to
    /// relative 1e-12 (the spec floor is 1e-10).
    #[test]
    fn c20_only_reproduces_analytic_j2() {
        let model = egm96_gravity_model();
        for pos in [
            leo_position(),
            Vector3::new(1_000_000.0, 2_000_000.0, 6_400_000.0),
            Vector3::new(-6_700_000.0, 900_000.0, -400_000.0),
        ] {
            let harmonic = egm96_acceleration_itrf(&pos, 2, 0);
            let analytic = j2_acceleration(&pos, &model);
            assert!(
                relative_norm_error(&harmonic, &analytic) <= 1e-12,
                "C̄₂₀-only vs analytic J2 at {pos:?}: rel {}",
                relative_norm_error(&harmonic, &analytic)
            );
        }
    }

    /// Zonal-only harmonics to degree 4 must equal analytic J2 plus the
    /// closed-form J3/J4 perturbations, with J3/J4 derived from the same
    /// table (Jₙ = −√(2n+1)·C̄ₙ₀) and the EGM96 constant set on both sides.
    #[test]
    fn zonal_only_reproduces_closed_form_j3_j4() {
        let model = egm96_gravity_model();
        let j3 = egm96_jn(3).expect("C̄₃₀ present");
        let j4 = egm96_jn(4).expect("C̄₄₀ present");
        for pos in [
            leo_position(),
            Vector3::new(1_000_000.0, 2_000_000.0, 6_400_000.0),
        ] {
            let harmonic = egm96_acceleration_itrf(&pos, 4, 0);
            let closed_form = j2_acceleration(&pos, &model)
                + j3_acceleration(&pos, &model, j3)
                + j4_acceleration(&pos, &model, j4);
            assert!(
                relative_norm_error(&harmonic, &closed_form) <= 1e-12,
                "zonal 4×0 vs closed form at {pos:?}: rel {}",
                relative_norm_error(&harmonic, &closed_form)
            );
        }
    }

    /// The gravity module's J3/J4 constants are exactly the table-derived
    /// values — a tripwire tying the closed-form constants to the fetched
    /// coefficient file.
    #[test]
    fn zonal_constants_tie_to_the_table() {
        assert_eq!(egm96_jn(3), Some(EARTH_J3));
        assert_eq!(egm96_jn(4), Some(EARTH_J4));
    }

    /// EGM96's derived J2 agrees with the (untouched) WGS-84 J2 constant to
    /// ~3e-6 relative — the two constant sets are close but not identical.
    #[test]
    fn egm96_j2_agrees_with_wgs84_j2_value() {
        let j2 = egm96_jn(2).expect("C̄₂₀ present");
        let wgs84 = GravityModel::EARTH_WGS84.j2;
        assert!(
            (j2 - wgs84).abs() / wgs84 < 1e-5,
            "J2 {j2} vs WGS-84 {wgs84}"
        );
    }

    // ── Truncation and clamping behavior ────────────────────────────────

    #[test]
    fn central_term_only_matches_two_body() {
        let pos = leo_position();
        let model = egm96_gravity_model();
        for (degree, order) in [(0, 0), (1, 1)] {
            let acc = egm96_acceleration_itrf(&pos, degree, order);
            assert!(relative_norm_error(&acc, &two_body_acceleration(&pos, &model)) <= 1e-15);
        }
    }

    #[test]
    fn degree_and_order_clamp_to_embedded_table() {
        let pos = leo_position();
        assert_eq!(
            egm96_acceleration_itrf(&pos, 20, 20),
            egm96_acceleration_itrf(&pos, 12, 12)
        );
        // Order clamps to degree: (4, 12) is the same field as (4, 4).
        assert_eq!(
            egm96_acceleration_itrf(&pos, 4, 12),
            egm96_acceleration_itrf(&pos, 4, 4)
        );
    }

    #[test]
    fn tesseral_orders_contribute() {
        let pos = leo_position();
        let full = egm96_acceleration_itrf(&pos, 12, 12);
        let zonal = egm96_acceleration_itrf(&pos, 12, 0);
        let difference = (full - zonal).norm();
        assert!(
            difference > 1e-9,
            "tesseral terms should contribute measurably, got {difference}"
        );
    }

    /// On the polar axis the spherical-gradient formulation would divide
    /// by zero; the floored meridian ring keeps the result finite and
    /// within the (tiny) tesseral contribution of the zonal-only field.
    #[test]
    fn polar_axis_evaluation_is_finite() {
        let pos = Vector3::new(0.0, 0.0, 6_900_000.0);
        let acc = egm96_acceleration_itrf(&pos, 12, 12);
        assert!(acc.iter().all(|c| c.is_finite()));
        let zonal = egm96_acceleration_itrf(&pos, 12, 0);
        assert!((acc - zonal).norm() < 1e-4);
    }
}
