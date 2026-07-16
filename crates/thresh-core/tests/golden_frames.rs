//! Golden-vector envelope tests for the IAU-76/FK5 frame-transform chain
//! (task 3.2 of `astro-time-and-frames`, design Decision 7).
//!
//! Fixtures are committed under `test-data/golden/frames/` (provenance,
//! sources, and regeneration commands in that directory's `PROVENANCE.md`)
//! and resolved `CARGO_MANIFEST_DIR`-relative per the repo's tier-3 golden
//! pattern. Everything here runs under plain `cargo test` with **default
//! features** — no network, no Python, no non-default feature — and the
//! transforms under test are pure Rust math, so two runs are bitwise
//! identical (asserted below; spec "Deterministic CI execution").
//!
//! Two fixture families:
//!
//! * `vallado-example-3-15.json` — Vallado's published worked IAU-76/FK5
//!   reduction (AAS 06-675 LEO test case / AIAA 2006-6753 Rev 3 TEME row)
//!   with its published EOP inputs. Every leg must reproduce the published
//!   position to ≤ 1 m (spec "Vallado reduction reproduced").
//! * `zero-eop-*.json` — astropy-computed TEME/GCRF/ITRF vectors at three
//!   epochs under the provider's zero-EOP defaults (UT1 = UTC, no polar
//!   motion), each comparison carrying its own documented tolerance
//!   (1.5 × the measured IAU-76-vs-IAU-2006/2000A theory delta + 1 m; see
//!   `PROVENANCE.md`).

use std::collections::HashMap;
use std::path::PathBuf;

use nalgebra::Vector3;
use serde::Deserialize;
use thresh_core::frames::{Frame, FrameProvider, Iau76Fk5Provider};
use thresh_core::time::Epoch;

/// Arcseconds to radians (EOP bulletins publish polar motion in arcseconds).
const ARCSEC: f64 = std::f64::consts::PI / (180.0 * 3600.0);

/// `test-data/golden/frames/` at the workspace root, resolved
/// `CARGO_MANIFEST_DIR`-relative (repo precedent:
/// `thresh_data::golden::golden_orbital_dir`).
fn golden_frames_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("test-data")
        .join("golden")
        .join("frames")
}

fn load_json<T: serde::de::DeserializeOwned>(file_name: &str) -> T {
    let path = golden_frames_dir().join(file_name);
    let text =
        std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    serde_json::from_str(&text).unwrap_or_else(|e| panic!("parse {}: {e}", path.display()))
}

/// Fixture frame names are uppercase display names ("GCRF", "ITRF", …).
fn parse_frame(name: &str) -> Frame {
    match name {
        "GCRF" => Frame::Gcrf,
        "MOD" => Frame::Mod,
        "TOD" => Frame::Tod,
        "TEME" => Frame::Teme,
        "PEF" => Frame::Pef,
        "ITRF" => Frame::Itrf,
        other => panic!("fixture names unknown frame {other:?}"),
    }
}

// ---------------------------------------------------------------------------
// Fixture schema (matching test-data/golden/frames/PROVENANCE.md)
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
struct ValladoFixture {
    epoch_utc: String,
    eop: Eop,
    input: InputState,
    legs: Vec<Leg>,
}

#[derive(Deserialize)]
struct Eop {
    dut1_s: f64,
    xp_arcsec: f64,
    yp_arcsec: f64,
}

#[derive(Deserialize)]
struct InputState {
    frame: String,
    r_m: [f64; 3],
    v_mps: [f64; 3],
}

#[derive(Deserialize)]
struct Leg {
    frame: String,
    published: PublishedState,
    tolerance_m: f64,
    tolerance_mps: f64,
}

#[derive(Deserialize)]
struct PublishedState {
    r_m: [f64; 3],
    v_mps: [f64; 3],
}

#[derive(Deserialize)]
struct ZeroEopFixture {
    case: String,
    epoch_utc: String,
    eop: Eop,
    states: HashMap<String, ZeroEopState>,
    comparisons: Vec<Comparison>,
}

#[derive(Deserialize)]
struct ZeroEopState {
    r_m: [f64; 3],
    v_mps: [f64; 3],
}

#[derive(Deserialize)]
struct Comparison {
    from_frame: String,
    to_frame: String,
    tolerance_m: f64,
    tolerance_mps: f64,
}

const ZERO_EOP_FILES: [&str; 3] = [
    "zero-eop-2015-06-30.json",
    "zero-eop-2020-01-01.json",
    "zero-eop-2024-03-20.json",
];

// ---------------------------------------------------------------------------
// Transform runners (shared by the envelope and determinism tests)
// ---------------------------------------------------------------------------

/// Runs every Vallado leg (ITRF input → PEF/TOD/MOD/GCRF/TEME with the
/// fixture EOP), asserting each against its published state and tolerance,
/// and appends every output component to `sink` for the determinism check.
fn run_vallado_legs(sink: &mut Vec<f64>) {
    let fixture: ValladoFixture = load_json("vallado-example-3-15.json");
    let epoch = Epoch::from_iso8601(&fixture.epoch_utc).expect("fixture epoch parses");
    let provider = Iau76Fk5Provider {
        delta_ut1: fixture.eop.dut1_s,
        polar_motion: Some((
            fixture.eop.xp_arcsec * ARCSEC,
            fixture.eop.yp_arcsec * ARCSEC,
        )),
    };

    assert_eq!(fixture.input.frame, "ITRF", "Vallado input must be ITRF");
    let r_itrf = Vector3::from(fixture.input.r_m);
    let v_itrf = Vector3::from(fixture.input.v_mps);
    assert!(!fixture.legs.is_empty(), "Vallado fixture has no legs");

    for leg in &fixture.legs {
        let rot = provider
            .rotation(Frame::Itrf, parse_frame(&leg.frame), &epoch)
            .expect("IAU-76/FK5 provider supports every pair");
        let (r, v) = rot.rotate_state(&r_itrf, &v_itrf);
        let dr = (r - Vector3::from(leg.published.r_m)).norm();
        let dv = (v - Vector3::from(leg.published.v_mps)).norm();
        assert!(
            dr <= leg.tolerance_m,
            "Vallado ITRF->{}: position off by {dr:.4} m (tolerance {} m)",
            leg.frame,
            leg.tolerance_m
        );
        assert!(
            dv <= leg.tolerance_mps,
            "Vallado ITRF->{}: velocity off by {dv:.6} m/s (tolerance {} m/s)",
            leg.frame,
            leg.tolerance_mps
        );
        sink.extend(r.iter().chain(v.iter()));
    }
}

/// Runs every comparison of every zero-EOP fixture with the default
/// (zero-EOP) provider, asserting each against its documented tolerance,
/// and appends every output component to `sink`.
fn run_zero_eop_comparisons(sink: &mut Vec<f64>) {
    for file in ZERO_EOP_FILES {
        let fixture: ZeroEopFixture = load_json(file);
        // Guard the premise: these fixtures were generated at the provider's
        // documented zero defaults (UT1 = UTC, no polar motion).
        assert_eq!(fixture.eop.dut1_s, 0.0, "{}: non-zero dUT1", fixture.case);
        assert_eq!(fixture.eop.xp_arcsec, 0.0, "{}: non-zero xp", fixture.case);
        assert_eq!(fixture.eop.yp_arcsec, 0.0, "{}: non-zero yp", fixture.case);

        let epoch = Epoch::from_iso8601(&fixture.epoch_utc).expect("fixture epoch parses");
        let provider = Iau76Fk5Provider::default();
        assert!(
            !fixture.comparisons.is_empty(),
            "{}: no comparisons",
            fixture.case
        );

        for cmp in &fixture.comparisons {
            let input = &fixture.states[&cmp.from_frame];
            let expected = &fixture.states[&cmp.to_frame];
            let rot = provider
                .rotation(
                    parse_frame(&cmp.from_frame),
                    parse_frame(&cmp.to_frame),
                    &epoch,
                )
                .expect("IAU-76/FK5 provider supports every pair");
            let (r, v) = rot.rotate_state(&Vector3::from(input.r_m), &Vector3::from(input.v_mps));
            let dr = (r - Vector3::from(expected.r_m)).norm();
            let dv = (v - Vector3::from(expected.v_mps)).norm();
            assert!(
                dr <= cmp.tolerance_m,
                "{} {}->{}: position off by {dr:.4} m (tolerance {} m)",
                fixture.case,
                cmp.from_frame,
                cmp.to_frame,
                cmp.tolerance_m
            );
            assert!(
                dv <= cmp.tolerance_mps,
                "{} {}->{}: velocity off by {dv:.6} m/s (tolerance {} m/s)",
                fixture.case,
                cmp.from_frame,
                cmp.to_frame,
                cmp.tolerance_mps
            );
            sink.extend(r.iter().chain(v.iter()));
        }
    }
}

// ---------------------------------------------------------------------------
// Envelope tests
// ---------------------------------------------------------------------------

/// Spec "Vallado reduction reproduced": the committed Vallado fixture's ITRF
/// state transformed to PEF, TOD, MOD, GCRF, and TEME with the fixture's
/// EOP values matches each published row within 1 m / 0.01 m/s.
#[test]
fn vallado_reduction_reproduced_within_1_m_per_leg() {
    run_vallado_legs(&mut Vec::new());
}

/// Exit-criterion measurement (task 5.2): at the Vallado fixture epoch the
/// TEME→GCRF relabeling this change makes real moves a LEO position by more
/// than 1 km — the error the old GMST-only conflation silently ignored.
#[test]
fn teme_gcrf_displacement_at_vallado_epoch_exceeds_1_km() {
    let fixture: ValladoFixture = load_json("vallado-example-3-15.json");
    let epoch = Epoch::from_iso8601(&fixture.epoch_utc).expect("fixture epoch parses");

    // Published rows first: the GCRF and TEME states of the same physical
    // vector differ by ~10 km at this epoch.
    let legs: HashMap<&str, &Leg> = fixture.legs.iter().map(|l| (l.frame.as_str(), l)).collect();
    let r_gcrf = Vector3::from(legs["GCRF"].published.r_m);
    let r_teme = Vector3::from(legs["TEME"].published.r_m);
    let published_displacement = (r_gcrf - r_teme).norm();
    assert!(
        published_displacement > 1_000.0,
        "published TEME/GCRF displacement {published_displacement:.1} m"
    );

    // And our own chain agrees: transforming the published TEME state to
    // GCRF moves it by the same kilometre-scale displacement.
    let (r, _v) = thresh_core::frames::teme_to_gcrf(
        &r_teme,
        &Vector3::from(legs["TEME"].published.v_mps),
        &epoch,
    );
    let displacement = (r - r_teme).norm();
    assert!(
        displacement > 1_000.0,
        "transform TEME/GCRF displacement {displacement:.1} m"
    );
    assert!(
        (displacement - published_displacement).abs() < 10.0,
        "chain displacement {displacement:.1} m vs published {published_displacement:.1} m"
    );
}

/// Zero-EOP fixtures (three epochs, astropy targets, Skyfield-cross-checked)
/// reproduce within their documented per-comparison tolerances under the
/// provider's zero-EOP defaults.
#[test]
fn zero_eop_fixtures_within_documented_tolerances() {
    run_zero_eop_comparisons(&mut Vec::new());
}

/// Spec "Deterministic CI execution": running the full golden suite twice —
/// fixture load from disk included — produces bitwise-identical results.
/// (No network and no Python are structural: this test only reads committed
/// JSON and calls pure Rust math.)
#[test]
fn two_runs_are_bitwise_identical() {
    let mut first = Vec::new();
    run_vallado_legs(&mut first);
    run_zero_eop_comparisons(&mut first);

    let mut second = Vec::new();
    run_vallado_legs(&mut second);
    run_zero_eop_comparisons(&mut second);

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
