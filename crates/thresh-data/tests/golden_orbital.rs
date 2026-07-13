//! Golden tier-3 envelope test (task 8.2, design Decision 8 of the
//! `orbital-ballistic-filter-models` change).
//!
//! `KeplerJ2` initialized from each committed SGP4 fixture's t₀ state must
//! stay within ≤ 5 km position error of the SGP4 samples out to 600 s. The
//! envelope is deliberately generous: SGP4 carries drag and short-periodic
//! terms that Kepler+J2 osculating physics legitimately lacks — this tier
//! certifies "right physics regime"; golden tiers 1 (element conversions,
//! `thresh-core`) and 2 (Vallado propagation, `thresh-filter`) certify
//! exactness.
//!
//! Spec "Deterministic CI execution": this test runs under plain
//! `cargo test` with **default features** — it only deserializes the
//! committed JSON (no `sgp4`, no `orbital` feature, no network, no Python)
//! and the prediction it checks is pure Rust math, bit-identical across
//! repeated runs (asserted below).
//!
//! Why this test lives in `thresh-data` rather than the umbrella `thresh`
//! crate: the tier-3 generator (`src/bin/gen_sgp4_fixtures.rs`) and the
//! fixture schema (`src/golden.rs`) are already here, so generator and
//! consumer stay co-located; the only new edge is a `thresh-filter`
//! dev-dependency, which points in the same direction as the existing
//! `thresh-data → thresh-tracker → thresh-filter` chain and adds no new
//! compilation unit. The umbrella-crate alternative would have needed a new
//! `serde_json` dev-dependency there and split the generator/test pair
//! across crates.
//!
//! Fixtures are resolved `CARGO_MANIFEST_DIR`-relative per the
//! `thresh-tracker/src/tracker.rs` test-data precedent.

use nalgebra::{DVector, Vector3};
use thresh_core::orbital::GravityModel;
use thresh_data::golden::{Sgp4Fixture, golden_orbital_dir};
use thresh_filter::models::kepler_j2::KeplerJ2;
use thresh_filter::traits::MotionModel;

/// Documented tier-3 envelope: maximum allowed position error vs the SGP4
/// samples at every sampled offset up to and including 600 s.
const ENVELOPE_M: f64 = 5_000.0;

/// Load every committed `*.json` fixture from `test-data/golden/orbital/`,
/// sorted by file name for deterministic iteration order.
fn load_fixtures() -> Vec<(String, Sgp4Fixture)> {
    let dir = golden_orbital_dir();
    let entries = std::fs::read_dir(&dir)
        .unwrap_or_else(|e| panic!("read golden fixture dir {}: {e}", dir.display()));

    let mut paths: Vec<_> = entries
        .map(|e| e.expect("read dir entry").path())
        .filter(|p| p.extension().is_some_and(|ext| ext == "json"))
        .collect();
    paths.sort();

    paths
        .into_iter()
        .map(|path| {
            let text = std::fs::read_to_string(&path)
                .unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
            let fixture: Sgp4Fixture = serde_json::from_str(&text)
                .unwrap_or_else(|e| panic!("parse {}: {e}", path.display()));
            (path.display().to_string(), fixture)
        })
        .collect()
}

/// Interleaved `[x, vx, y, vy, z, vz]` filter state from the fixture's
/// t₀ sample (the sample with `offset_s == 0`).
fn t0_state(fixture: &Sgp4Fixture) -> DVector<f64> {
    let t0 = fixture
        .samples
        .iter()
        .find(|s| s.offset_s == 0.0)
        .unwrap_or_else(|| panic!("{}: no t0 sample", fixture.object));
    let p = t0.position_m;
    let v = t0.velocity_m_s;
    DVector::from_column_slice(&[p[0], v[0], p[1], v[1], p[2], v[2]])
}

/// Position error (metres) between a `KeplerJ2` prediction from the t₀
/// state over `offset_s` and the fixture's SGP4 sample at that offset.
fn position_error_m(model: &KeplerJ2, x0: &DVector<f64>, fixture: &Sgp4Fixture, idx: usize) -> f64 {
    let sample = &fixture.samples[idx];
    let predicted = model.predict(x0, sample.offset_s);
    let pos = Vector3::new(predicted[0], predicted[2], predicted[4]);
    let sgp4_pos = Vector3::from(sample.position_m);
    (pos - sgp4_pos).norm()
}

/// Spec scenario: `KeplerJ2` from each fixture's t₀ state stays within the
/// 5 km envelope of the SGP4 samples at every offset through 600 s.
///
/// Observed errors at fixture-generation time (sgp4 1.2.2, informational —
/// the gate is the 5 km envelope): 600 s position error was 5.8 m (ISS),
/// 10.7 m (06251), 3.1 m (28057). Over 600 s the SGP4-vs-J2-osculating
/// divergence is dominated by drag and higher-order short-periodics, both
/// tiny on that horizon; the generous envelope guards regime-level breakage
/// (wrong frame, wrong mu, integrator blowup), not metre-level drift.
#[test]
fn kepler_j2_stays_within_sgp4_envelope() {
    let fixtures = load_fixtures();
    assert!(
        fixtures.len() >= 2,
        "expected at least 2 committed SGP4 fixtures, found {} — \
         regenerate with `cargo run -p thresh-data --features orbital --bin gen-sgp4-fixtures`",
        fixtures.len()
    );

    let model = KeplerJ2::new(GravityModel::EARTH_WGS84);
    for (path, fixture) in &fixtures {
        assert_eq!(fixture.frame, "TEME", "{path}: unexpected frame");
        let x0 = t0_state(fixture);
        let mut saw_600 = false;
        for (idx, sample) in fixture.samples.iter().enumerate() {
            let err = position_error_m(&model, &x0, fixture, idx);
            assert!(
                err <= ENVELOPE_M,
                "{}: position error {err:.1} m at t0+{} s exceeds the {ENVELOPE_M} m envelope",
                fixture.object,
                sample.offset_s
            );
            saw_600 |= sample.offset_s == 600.0;
        }
        assert!(saw_600, "{}: fixture has no 600 s sample", fixture.object);
    }
}

/// Spec "Deterministic CI execution", tier-3 leg: repeated runs of the
/// envelope computation are bit-identical (tiers 1/2 assert the same in
/// `thresh-core` / `thresh-filter` unit tests). Everything here is pure
/// Rust float math over committed JSON — no Python, no network, no
/// non-default features.
#[test]
fn envelope_check_is_bit_identical_across_runs() {
    let fixtures = load_fixtures();
    let model = KeplerJ2::new(GravityModel::EARTH_WGS84);

    for (_, fixture) in &fixtures {
        let x0 = t0_state(fixture);
        for idx in 0..fixture.samples.len() {
            let a = position_error_m(&model, &x0, fixture, idx);
            let b = position_error_m(&model, &x0, fixture, idx);
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "{}: repeated envelope evaluation differs",
                fixture.object
            );
        }
    }
}
