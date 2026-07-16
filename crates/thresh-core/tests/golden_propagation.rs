//! Golden-vector envelope tests for the high-fidelity orbit-propagation
//! stack (task 4.3 of `orbit-propagation-fidelity`, design Decision 7).
//!
//! Fixtures are committed under `test-data/golden/propagation/` (sources,
//! measured sensitivities, and regeneration commands in that directory's
//! `PROVENANCE.md`) and resolved `CARGO_MANIFEST_DIR`-relative per the
//! repo's tier-3 golden pattern. Everything here runs under plain
//! `cargo test` with **default features** — no network, no Python — and
//! the code under test is pure Rust math, so repeated runs are bitwise
//! identical (asserted below).
//!
//! Two fixture families:
//!
//! * **Component goldens** — analytic Sun/Moon positions against the
//!   astropy authority (spec "Sun and Moon positions match the independent
//!   authority"), EGM96 12×12 spot accelerations against an independent
//!   evaluation of the same fetched coefficient file (spec "Harmonic
//!   acceleration matches independent evaluation"), and the published
//!   Harris-Priester table plus its tool-computed worked examples.
//! * **Trajectory goldens** — three arcs integrated by an independently
//!   written Python force stack under scipy DOP853 at `rtol = atol =
//!   1e-12`, reproduced here by [`propagate_on_grid`] on each fixture's
//!   exact sample grid with the fixture's force configuration mirrored
//!   field by field (every shared constant asserted, most of them bitwise)
//!   within the fixture's measured per-arc tolerance (spec "Golden arcs
//!   reproduced within measured tolerance"), plus the LEO full-stack vs
//!   J2-only fidelity demonstration (spec "Higher fidelity than the J2
//!   baseline").
//!
//! Arc propagation uses [`DpConfig::default`] (abs/rel tolerances 1e-9,
//! 300 s step ceiling — the same ceiling the generator ran with). The
//! budgets carry no separate Rust-integrator term; what covers it is
//! recorded in each fixture: a measured scipy integrator-sensitivity term
//! (rtol 1e-12 vs 1e-10) plus an informational rtol-1e-9 re-run delta
//! showing what the 1e-9 class contributes (LEO 1.2e-2 m, MEO 1.6e-7 m,
//! SRP 3.4e-2 m — at or below the budgeted term on every arc), all inside
//! the 3× margin on the budget sum plus the 1 m floor. That margin is the
//! cover on the MEO arc, where both integrator numbers are sub-micrometre
//! and the budget is ephemeris-swap dominated.

use std::path::PathBuf;

use nalgebra::Vector3;
use serde::Deserialize;
use thresh_core::frames::Iau76Fk5Provider;
use thresh_core::orbital::atmosphere::{HARRIS_PRIESTER_TABLE, harris_priester_density};
use thresh_core::orbital::dormand_prince::{DpConfig, DpSolution, propagate_on_grid};
use thresh_core::orbital::egm96::{
    EGM96_MU, EGM96_RADIUS, egm96_acceleration_itrf, egm96_gravity_model,
};
use thresh_core::orbital::ephemeris::{moon_position_gcrf, sun_position_gcrf};
use thresh_core::orbital::force_config::{
    DragConfig, ForceModelConfig, GravityFidelity, SrpConfig, ThirdBodyConfig,
};
use thresh_core::orbital::srp::{
    SHADOW_CYLINDER_RADIUS_M, SOLAR_PRESSURE_1AU_N_M2, cylindrical_shadow_factor,
};
use thresh_core::orbital::third_body::{GM_MOON, GM_SUN};
use thresh_core::time::Epoch;

/// `test-data/golden/propagation/` at the workspace root, resolved
/// `CARGO_MANIFEST_DIR`-relative (repo precedent: `golden_frames.rs`).
fn golden_propagation_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("test-data")
        .join("golden")
        .join("propagation")
}

fn load_json<T: serde::de::DeserializeOwned>(file_name: &str) -> T {
    let path = golden_propagation_dir().join(file_name);
    let text =
        std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    serde_json::from_str(&text).unwrap_or_else(|e| panic!("parse {}: {e}", path.display()))
}

/// Angle between two vectors in arcseconds.
fn angular_separation_arcsec(a: &Vector3<f64>, b: &Vector3<f64>) -> f64 {
    let cos = (a.dot(b) / (a.norm() * b.norm())).clamp(-1.0, 1.0);
    cos.acos().to_degrees() * 3600.0
}

/// A fixture-recorded shared constant must equal the Rust constant.
///
/// Compared within 1 ulp rather than bitwise because serde_json's default
/// float parsing is not guaranteed correctly rounded (measured: the
/// fixtures' 17-digit `j2` parses 1 ulp off; Python parses the same text
/// to exactly the Rust-computed value). The genuine bit-level identities
/// are asserted against rustc-parsed literals in the `srp`, `third_body`,
/// and `egm96` unit tests; enabling serde_json's `float_roundtrip`
/// workspace-wide is out of scope for an everything-additive change.
fn assert_recorded_constant(recorded: f64, expected: f64, case: &str, what: &str) {
    assert!(
        (recorded - expected).abs() <= f64::EPSILON * expected.abs(),
        "{case}: recorded {what} {recorded:e} differs from the Rust constant {expected:e}"
    );
}

// ---------------------------------------------------------------------------
// Fixture schemas (matching test-data/golden/propagation/PROVENANCE.md)
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
struct SunMoonFixture {
    epochs: Vec<SunMoonEpoch>,
    tolerances: SunMoonTolerances,
}

#[derive(Deserialize)]
struct SunMoonEpoch {
    epoch_utc: String,
    sun_gcrf_m: [f64; 3],
    moon_gcrf_m: [f64; 3],
}

#[derive(Deserialize)]
struct SunMoonTolerances {
    sun_angular_arcsec: f64,
    moon_angular_arcsec: f64,
    sun_distance_rel: f64,
    moon_distance_rel: f64,
}

#[derive(Deserialize)]
struct Egm96Fixture {
    model: Egm96Model,
    tolerance: Egm96Tolerance,
    spots: Vec<Egm96Spot>,
}

#[derive(Deserialize)]
struct Egm96Model {
    degree: u32,
    order: u32,
    gm_m3_s2: f64,
    reference_radius_m: f64,
}

#[derive(Deserialize)]
struct Egm96Tolerance {
    relative_per_component: f64,
}

#[derive(Deserialize)]
struct Egm96Spot {
    name: String,
    position_itrf_m: [f64; 3],
    acceleration_total_m_s2: [f64; 3],
}

#[derive(Deserialize)]
struct HpFixture {
    convention: HpConvention,
    table: Vec<HpRow>,
    worked_examples: HpWorkedExamples,
}

#[derive(Deserialize)]
struct HpConvention {
    bulge_exponent_n: u32,
}

#[derive(Deserialize)]
struct HpRow {
    altitude_m: f64,
    rho_min_kg_m3: f64,
    rho_max_kg_m3: f64,
}

#[derive(Deserialize)]
struct HpWorkedExamples {
    values: Vec<HpExample>,
}

#[derive(Deserialize)]
struct HpExample {
    height_m: f64,
    cos_psi: f64,
    density_kg_m3: f64,
    note: String,
}

#[derive(Deserialize)]
struct ArcFixture {
    case: String,
    epoch_utc: String,
    frame: String,
    initial_state: ArcInitialState,
    force_config: ArcForceConfig,
    tolerance: ArcTolerance,
    samples: Vec<ArcSample>,
    shadow_crossings: Option<ArcShadowCrossings>,
    j2_only_comparison: Option<ArcJ2Comparison>,
}

#[derive(Deserialize)]
struct ArcInitialState {
    r_m: [f64; 3],
    v_mps: [f64; 3],
}

#[derive(Deserialize)]
struct ArcForceConfig {
    gravity: ArcGravity,
    drag: Option<ArcDrag>,
    srp: Option<ArcSrp>,
    third_body: ArcThirdBody,
    j2_baseline_for_fidelity_comparison: ArcJ2Baseline,
}

#[derive(Deserialize)]
struct ArcGravity {
    model: String,
    degree: Option<u32>,
    order: Option<u32>,
    gm_m3_s2: f64,
    reference_radius_m: Option<f64>,
}

#[derive(Deserialize)]
struct ArcDrag {
    model: String,
    inv_beta_m2_per_kg: f64,
    bulge_exponent_n: u32,
    apex_lag_deg: f64,
}

#[derive(Deserialize)]
struct ArcSrp {
    model: String,
    cr_area_over_mass_m2_per_kg: f64,
    pressure_at_1au_n_m2: f64,
    shadow: ArcShadow,
}

#[derive(Deserialize)]
struct ArcShadow {
    model: String,
    radius_m: f64,
}

#[derive(Deserialize)]
struct ArcThirdBody {
    sun: bool,
    moon: bool,
    gm_sun_m3_s2: Option<f64>,
    gm_moon_m3_s2: Option<f64>,
}

#[derive(Deserialize)]
struct ArcJ2Baseline {
    mu_m3_s2: f64,
    equatorial_radius_m: f64,
    j2: f64,
}

#[derive(Deserialize)]
struct ArcTolerance {
    position_m: f64,
    velocity_mps: f64,
}

#[derive(Deserialize)]
struct ArcSample {
    t_s: f64,
    r_m: [f64; 3],
    v_mps: [f64; 3],
}

#[derive(Deserialize)]
struct ArcShadowCrossings {
    count: usize,
}

#[derive(Deserialize)]
struct ArcJ2Comparison {
    max_position_delta_m: f64,
    final_position_delta_m: f64,
}

const ARC_FILES: [&str; 3] = [
    "leo-full-force.json",
    "meo-harmonics-lunisolar.json",
    "srp-shadow-crossing.json",
];

// ---------------------------------------------------------------------------
// Component-golden runners (shared by the envelope and determinism tests)
// ---------------------------------------------------------------------------

/// One ephemeris body against its authority position: angular separation
/// within the fixture tolerance, distance within the relative tolerance.
fn assert_body_within(
    body: &str,
    epoch_utc: &str,
    ours: &Vector3<f64>,
    authority: &[f64; 3],
    angular_tol_arcsec: f64,
    distance_rel_tol: f64,
) {
    let authority = Vector3::from(*authority);
    let separation = angular_separation_arcsec(ours, &authority);
    let distance_rel = (ours.norm() - authority.norm()).abs() / authority.norm();
    assert!(
        separation <= angular_tol_arcsec,
        "{body} @ {epoch_utc}: {separation:.1}\" from the authority \
         (tolerance {angular_tol_arcsec}\")"
    );
    assert!(
        distance_rel <= distance_rel_tol,
        "{body} @ {epoch_utc}: relative distance error {distance_rel:.2e} \
         (tolerance {distance_rel_tol:.2e})"
    );
}

/// Sun/Moon GCRF positions at every fixture epoch against the astropy
/// authority, within the fixture's measured per-body tolerances; every
/// computed component is appended to `sink` for the determinism check.
fn run_sun_moon_golden(sink: &mut Vec<f64>) {
    let fixture: SunMoonFixture = load_json("sun-moon-ephemeris.json");
    assert!(!fixture.epochs.is_empty(), "sun-moon fixture has no epochs");
    for row in &fixture.epochs {
        let epoch = Epoch::from_iso8601(&row.epoch_utc).expect("fixture epoch parses");
        let sun = sun_position_gcrf(&epoch);
        let moon = moon_position_gcrf(&epoch);
        assert_body_within(
            "sun",
            &row.epoch_utc,
            &sun,
            &row.sun_gcrf_m,
            fixture.tolerances.sun_angular_arcsec,
            fixture.tolerances.sun_distance_rel,
        );
        assert_body_within(
            "moon",
            &row.epoch_utc,
            &moon,
            &row.moon_gcrf_m,
            fixture.tolerances.moon_angular_arcsec,
            fixture.tolerances.moon_distance_rel,
        );
        sink.extend(sun.iter().chain(moon.iter()));
    }
}

/// EGM96 spot accelerations against the generator's independent evaluation
/// of the same fetched coefficient file, per component within the fixture's
/// relative tolerance.
fn run_egm96_golden(sink: &mut Vec<f64>) {
    let fixture: Egm96Fixture = load_json("egm96-spot-accelerations.json");
    // The shared constant set is the executable contract.
    assert_recorded_constant(fixture.model.gm_m3_s2, EGM96_MU, "egm96", "GM");
    assert_recorded_constant(
        fixture.model.reference_radius_m,
        EGM96_RADIUS,
        "egm96",
        "reference radius",
    );
    assert!(!fixture.spots.is_empty(), "EGM96 fixture has no spots");

    let tol = fixture.tolerance.relative_per_component;
    for spot in &fixture.spots {
        let accel = egm96_acceleration_itrf(
            &Vector3::from(spot.position_itrf_m),
            fixture.model.degree,
            fixture.model.order,
        );
        for i in 0..3 {
            let expected = spot.acceleration_total_m_s2[i];
            assert_ne!(
                expected, 0.0,
                "{}: component {i} unexpectedly zero",
                spot.name
            );
            let rel = (accel[i] - expected).abs() / expected.abs();
            assert!(
                rel <= tol,
                "{} component {i}: {} vs {expected} (relative {rel:.2e}, tolerance {tol:.0e})",
                spot.name,
                accel[i]
            );
        }
        sink.extend(accel.iter());
    }
}

/// Published Harris-Priester table nodes bit-identical to the embedded
/// table, and the fixture's tool-computed worked examples reproduced
/// through [`harris_priester_density`].
fn run_harris_priester_golden(sink: &mut Vec<f64>) {
    let fixture: HpFixture = load_json("harris-priester-table.json");
    assert_eq!(
        fixture.convention.bulge_exponent_n, 2,
        "bulge exponent is fixed at n = 2 (design Decision 4)"
    );
    assert_eq!(
        fixture.table.len(),
        HARRIS_PRIESTER_TABLE.len(),
        "table row count"
    );
    for (row, (alt, rho_min, rho_max)) in fixture.table.iter().zip(HARRIS_PRIESTER_TABLE) {
        assert_eq!(row.altitude_m.to_bits(), alt.to_bits());
        assert_eq!(row.rho_min_kg_m3.to_bits(), rho_min.to_bits());
        assert_eq!(row.rho_max_kg_m3.to_bits(), rho_max.to_bits());
        // Node reproduction through the interpolation itself (spec
        // "Published table values reproduced"): the bulge extremes at a
        // node are the node's own min/max densities.
        let at_max = harris_priester_density(row.altitude_m, 1.0);
        let at_min = harris_priester_density(row.altitude_m, -1.0);
        assert!((at_max - row.rho_max_kg_m3).abs() <= 1e-12 * row.rho_max_kg_m3);
        assert!((at_min - row.rho_min_kg_m3).abs() <= 1e-12 * row.rho_min_kg_m3);
        sink.push(at_max);
        sink.push(at_min);
    }
    assert!(
        !fixture.worked_examples.values.is_empty(),
        "H-P fixture has no worked examples"
    );
    for example in &fixture.worked_examples.values {
        let rho = harris_priester_density(example.height_m, example.cos_psi);
        let rel = (rho - example.density_kg_m3).abs() / example.density_kg_m3;
        assert!(
            rel <= 1e-12,
            "H-P worked example ({}): {rho} vs {} (relative {rel:.2e})",
            example.note,
            example.density_kg_m3
        );
        sink.push(rho);
    }
}

// ---------------------------------------------------------------------------
// Trajectory-golden runners
// ---------------------------------------------------------------------------

/// Translate the fixture's gravity block, asserting the shared constants
/// (bit-identical — the two stacks recorded the same numbers by contract).
fn arc_gravity(gravity: &ArcGravity, case: &str) -> GravityFidelity {
    assert_recorded_constant(gravity.gm_m3_s2, EGM96_MU, case, "gravity GM");
    match gravity.model.as_str() {
        "two_body" => GravityFidelity::TwoBody,
        "egm96" => {
            let radius = gravity
                .reference_radius_m
                .unwrap_or_else(|| panic!("{case}: no reference radius"));
            assert_recorded_constant(radius, EGM96_RADIUS, case, "EGM96 reference radius");
            GravityFidelity::Harmonics {
                degree: gravity
                    .degree
                    .unwrap_or_else(|| panic!("{case}: no degree")),
                order: gravity.order.unwrap_or_else(|| panic!("{case}: no order")),
            }
        }
        other => panic!("{case}: unknown gravity model {other:?}"),
    }
}

/// Translate the fixture's drag block (Harris-Priester only in the golden
/// set), asserting the recorded bulge convention.
fn arc_drag(drag: &ArcDrag, case: &str) -> DragConfig {
    assert_eq!(drag.model, "harris_priester", "{case}: drag model");
    assert_eq!(drag.bulge_exponent_n, 2, "{case}: bulge exponent");
    assert_eq!(drag.apex_lag_deg, 30.0, "{case}: apex lag");
    DragConfig::HarrisPriester {
        inv_beta: drag.inv_beta_m2_per_kg,
    }
}

/// Translate the fixture's SRP block, asserting the solar-pressure
/// constant bit-for-bit (the design.md open-question resolution) and the
/// shadow-cylinder convention.
fn arc_srp(srp: &ArcSrp, case: &str) -> SrpConfig {
    assert_eq!(
        srp.model, "cannonball_cylindrical_shadow",
        "{case}: SRP model"
    );
    // The bit-identity of P(1 AU) across the stacks is asserted against
    // the rustc-parsed literal in the `srp` unit tests; here the recorded
    // value goes through serde_json (see `assert_recorded_constant`).
    assert_recorded_constant(
        srp.pressure_at_1au_n_m2,
        SOLAR_PRESSURE_1AU_N_M2,
        case,
        "P(1 AU)",
    );
    assert_eq!(srp.shadow.model, "cylinder", "{case}: shadow model");
    assert_recorded_constant(
        srp.shadow.radius_m,
        SHADOW_CYLINDER_RADIUS_M,
        case,
        "shadow-cylinder radius",
    );
    SrpConfig {
        cr_area_over_mass: srp.cr_area_over_mass_m2_per_kg,
    }
}

/// Translate the fixture's third-body block, asserting the JPL DE440 GM
/// values bit-for-bit where enabled.
fn arc_third_body(third_body: &ArcThirdBody, case: &str) -> ThirdBodyConfig {
    if third_body.sun {
        let gm = third_body
            .gm_sun_m3_s2
            .unwrap_or_else(|| panic!("{case}: no GM_Sun"));
        assert_recorded_constant(gm, GM_SUN, case, "GM_Sun");
    }
    if third_body.moon {
        let gm = third_body
            .gm_moon_m3_s2
            .unwrap_or_else(|| panic!("{case}: no GM_Moon"));
        assert_recorded_constant(gm, GM_MOON, case, "GM_Moon");
    }
    ThirdBodyConfig {
        sun: third_body.sun,
        moon: third_body.moon,
    }
}

/// Mirror the fixture's `force_config` block field by field into the Rust
/// [`ForceModelConfig`], with every shared constant asserted.
fn arc_force_model(config: &ArcForceConfig, case: &str) -> ForceModelConfig {
    // The J2 baseline recorded for the fidelity comparison is the same
    // constant set the Rust default configuration evaluates with.
    let baseline = egm96_gravity_model();
    let recorded = &config.j2_baseline_for_fidelity_comparison;
    assert_recorded_constant(recorded.mu_m3_s2, baseline.mu, case, "J2-baseline mu");
    assert_recorded_constant(
        recorded.equatorial_radius_m,
        baseline.equatorial_radius,
        case,
        "J2-baseline radius",
    );
    assert_recorded_constant(recorded.j2, baseline.j2, case, "J2-baseline J2");

    ForceModelConfig {
        gravity: arc_gravity(&config.gravity, case),
        drag: config.drag.as_ref().map(|d| arc_drag(d, case)),
        srp: config.srp.as_ref().map(|s| arc_srp(s, case)),
        third_body: arc_third_body(&config.third_body, case),
    }
}

/// Propagate the fixture's initial state on the fixture's exact sample
/// grid under `forces` (GCRF, epoch-anchored, zero-EOP provider — the
/// fixture's stated frame convention).
fn run_arc_with(fixture: &ArcFixture, forces: ForceModelConfig) -> DpSolution {
    assert_eq!(fixture.frame, "GCRF", "{}: arcs are GCRF", fixture.case);
    let epoch = Epoch::from_iso8601(&fixture.epoch_utc).expect("fixture epoch parses");
    let provider = Iau76Fk5Provider::default();
    let accel = forces.build(epoch, &provider);
    let offsets: Vec<f64> = fixture.samples.iter().map(|s| s.t_s).collect();
    propagate_on_grid(
        &Vector3::from(fixture.initial_state.r_m),
        &Vector3::from(fixture.initial_state.v_mps),
        &offsets,
        &DpConfig::default(),
        accel,
    )
}

/// Propagate the arc with its own fixture force configuration.
fn run_arc(fixture: &ArcFixture) -> DpSolution {
    run_arc_with(
        fixture,
        arc_force_model(&fixture.force_config, &fixture.case),
    )
}

/// Worst position/velocity deltas against the golden samples, asserting
/// the grid is reproduced sample for sample (bitwise offsets — the
/// grid-exact sampling contract).
fn max_deltas(fixture: &ArcFixture, solution: &DpSolution) -> (f64, f64) {
    assert_eq!(
        solution.samples.len(),
        fixture.samples.len(),
        "{}: sample count",
        fixture.case
    );
    let mut max_dr = 0.0_f64;
    let mut max_dv = 0.0_f64;
    for (golden, ours) in fixture.samples.iter().zip(&solution.samples) {
        assert_eq!(
            ours.offset_s.to_bits(),
            golden.t_s.to_bits(),
            "{}: sample offset",
            fixture.case
        );
        max_dr = max_dr.max((ours.position - Vector3::from(golden.r_m)).norm());
        max_dv = max_dv.max((ours.velocity - Vector3::from(golden.v_mps)).norm());
    }
    (max_dr, max_dv)
}

/// Every sampled state within the fixture's measured tolerance (spec
/// "Golden arcs reproduced within measured tolerance").
fn assert_arc_within_tolerance(fixture: &ArcFixture, solution: &DpSolution) {
    let (max_dr, max_dv) = max_deltas(fixture, solution);
    assert!(
        max_dr <= fixture.tolerance.position_m,
        "{}: max position delta {max_dr:.3} m exceeds the measured tolerance {} m",
        fixture.case,
        fixture.tolerance.position_m
    );
    assert!(
        max_dv <= fixture.tolerance.velocity_mps,
        "{}: max velocity delta {max_dv:.6} m/s exceeds the measured tolerance {} m/s",
        fixture.case,
        fixture.tolerance.velocity_mps
    );
}

/// When the fixture records shadow crossings, the cylindrical shadow
/// factor evaluated along our own trajectory (with our own Sun ephemeris)
/// transitions exactly as many times on the sample grid.
fn assert_shadow_crossings(fixture: &ArcFixture, solution: &DpSolution) {
    let Some(crossings) = &fixture.shadow_crossings else {
        return;
    };
    let epoch = Epoch::from_iso8601(&fixture.epoch_utc).expect("fixture epoch parses");
    let mut transitions = 0_usize;
    let mut previous: Option<f64> = None;
    for sample in &solution.samples {
        let sun = sun_position_gcrf(&(epoch + sample.offset_s));
        let factor = cylindrical_shadow_factor(&sample.position, &sun);
        if previous.is_some_and(|p| p != factor) {
            transitions += 1;
        }
        previous = Some(factor);
    }
    assert_eq!(
        transitions, crossings.count,
        "{}: shadow-factor transitions on the sample grid",
        fixture.case
    );
}

/// Full envelope for one arc; appends every sampled component to `sink`.
fn run_arc_envelope(file: &str, sink: &mut Vec<f64>) {
    let fixture: ArcFixture = load_json(file);
    let solution = run_arc(&fixture);
    assert_arc_within_tolerance(&fixture, &solution);
    assert_shadow_crossings(&fixture, &solution);
    for sample in &solution.samples {
        sink.extend(sample.position.iter().chain(sample.velocity.iter()));
    }
}

// ---------------------------------------------------------------------------
// Envelope tests
// ---------------------------------------------------------------------------

/// Spec "Sun and Moon positions match the independent authority": the
/// analytic ephemerides stay within the fixture's measured (3× margin)
/// per-body angular and distance tolerances at all six epochs.
#[test]
fn sun_and_moon_positions_match_the_independent_authority() {
    run_sun_moon_golden(&mut Vec::new());
}

/// Spec "Harmonic acceleration matches independent evaluation": EGM96
/// 12×12 accelerations at the four ITRF spots agree per component to the
/// fixture's 1e-9 relative tolerance.
#[test]
fn egm96_spot_accelerations_match_independent_evaluation() {
    run_egm96_golden(&mut Vec::new());
}

/// The published Harris-Priester table is embedded bit-for-bit and the
/// fixture's worked interpolation/bulge examples reproduce through the
/// Rust density function.
#[test]
fn harris_priester_table_and_worked_examples_match_the_fixture() {
    run_harris_priester_golden(&mut Vec::new());
}

/// Spec "Golden arcs reproduced within measured tolerance", LEO arc:
/// EGM96 12×12 + Harris-Priester drag + SRP with shadow + lunisolar third
/// body over two revolutions, every sample within 3.3 m / 3.5 mm/s
/// (measured on this implementation: 6.4e-2 m / 6.1e-5 m/s — 51× margin).
#[test]
fn leo_full_force_arc_reproduced_within_measured_tolerance() {
    run_arc_envelope("leo-full-force.json", &mut Vec::new());
}

/// Spec "Golden arcs reproduced within measured tolerance", MEO arc:
/// harmonics + lunisolar over 12 h, every sample within 34 m / 5.6 mm/s.
/// The budget is ephemeris-swap dominated (fixture sensitivity 10.8 m for
/// substituting the analytic series this stack implements — Meeus Ch. 25
/// Sun + Vallado Alg 31 Moon) and the measured delta — 10.6 m /
/// 1.5e-3 m/s — sits exactly on that term, as designed.
#[test]
fn meo_harmonics_lunisolar_arc_reproduced_within_measured_tolerance() {
    run_arc_envelope("meo-harmonics-lunisolar.json", &mut Vec::new());
}

/// Spec "Golden arcs reproduced within measured tolerance", SRP arc:
/// two-body + cannonball SRP crossing Earth shadow twice, every sample
/// within 1.6 m / 1.5 mm/s (measured: 1.9e-1 m / 1.6e-4 m/s — 8× margin),
/// with the two crossings reproduced on the sample grid.
#[test]
fn srp_shadow_crossing_arc_reproduced_within_measured_tolerance() {
    run_arc_envelope("srp-shadow-crossing.json", &mut Vec::new());
}

/// Spec "Higher fidelity than the J2 baseline": on the LEO golden arc the
/// full force stack tracks the fixture at least an order of magnitude
/// closer than the J2-only baseline, whose miss also reproduces the
/// generator's recorded J2-only comparison (same force model on both
/// stacks — the residual is integrator-only). Measured: full stack
/// 6.4e-2 m max vs J2-only 463.7 m max / 385.4 m final — a ~7200×
/// fidelity factor, asserted at ≥ 10×.
#[test]
fn full_stack_tracks_the_golden_closer_than_j2_only() {
    let fixture: ArcFixture = load_json("leo-full-force.json");
    let full = run_arc(&fixture);
    let (full_dr, _) = max_deltas(&fixture, &full);

    let j2_only = run_arc_with(&fixture, ForceModelConfig::default());
    let (j2_dr, _) = max_deltas(&fixture, &j2_only);
    let golden_final = Vector3::from(fixture.samples.last().expect("samples").r_m);
    let j2_final = (j2_only.samples.last().expect("samples").position - golden_final).norm();

    // Our J2-only run lands where the generator's J2-only run landed
    // (same closed-form force on both stacks, so the residual is
    // integrator-only: measured 0.02 m on 385.4 m, asserted at ~100×).
    let recorded = fixture
        .j2_only_comparison
        .as_ref()
        .expect("LEO fixture records the J2-only comparison");
    assert!(
        (j2_final - recorded.final_position_delta_m).abs() <= 2.0,
        "J2-only final miss {j2_final:.3} m vs the recorded {} m",
        recorded.final_position_delta_m
    );
    assert!(
        (j2_dr - recorded.max_position_delta_m).abs() <= 2.0,
        "J2-only max miss {j2_dr:.3} m vs the recorded {} m",
        recorded.max_position_delta_m
    );

    // The fidelity demonstration, numeric: full stack within tolerance,
    // J2-only misses by ≥ 20× tolerance (the generation-time headroom
    // assertion), i.e. the full stack is ≥ 10× closer.
    assert!(full_dr <= fixture.tolerance.position_m);
    assert!(
        j2_dr >= 20.0 * fixture.tolerance.position_m,
        "J2-only max miss {j2_dr:.1} m lost its fidelity headroom \
         (tolerance {} m)",
        fixture.tolerance.position_m
    );
    assert!(
        full_dr * 10.0 <= j2_dr,
        "full stack ({full_dr:.3} m) must beat J2-only ({j2_dr:.3} m) by ≥ 10×"
    );
}

/// Repeated runs of the entire golden suite — fixture load from disk
/// included — are bitwise identical (the trajectory-golden requirement's
/// determinism clause; no network and no Python are structural: the tests
/// only read committed JSON and call pure Rust math).
#[test]
fn two_runs_are_bitwise_identical() {
    let run_all = || {
        let mut sink = Vec::new();
        run_sun_moon_golden(&mut sink);
        run_egm96_golden(&mut sink);
        run_harris_priester_golden(&mut sink);
        for file in ARC_FILES {
            run_arc_envelope(file, &mut sink);
        }
        sink
    };
    let first = run_all();
    let second = run_all();

    assert_eq!(first.len(), second.len());
    assert!(!first.is_empty());
    for (i, (a, b)) in first.iter().zip(&second).enumerate() {
        assert_eq!(
            a.to_bits(),
            b.to_bits(),
            "component {i} differs between runs: {a:?} vs {b:?}"
        );
    }
}
