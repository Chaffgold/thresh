//! `gen-sgp4-fixtures` — SGP4 golden-fixture generator (tier 3, Decision 8
//! of the `orbital-ballistic-filter-models` change).
//!
//! Run **manually, never in CI**: the output is committed under
//! `test-data/golden/orbital/`, so CI only ever reads the fixtures (plain
//! `cargo test`, default features, no `sgp4`, no network). Regenerate with:
//!
//! ```text
//! cargo run -p thresh-data --features orbital --bin gen-sgp4-fixtures
//! ```
//!
//! The generator propagates three LEO objects with the `sgp4` crate (whose
//! propagation is verified against the AIAA-2006-6753 test suite) and
//! writes TEME position/velocity samples at t₀ and t₀ + {60, 300, 600} s as
//! JSON, plus a `PROVENANCE.md` recording the TLEs, the `sgp4` crate
//! version (read from the workspace `Cargo.lock`), and this regeneration
//! command. All TLEs are inlined below — no network access, ever.
//!
//! Output is bit-identical across repeated runs at the same `sgp4` version:
//! the TLEs are constants and `serde_json` float formatting is
//! deterministic (shortest round-trip).

use std::fs;
use std::path::Path;
use std::process::ExitCode;

use thresh_data::golden::{Sgp4Fixture, Sgp4Sample, golden_orbital_dir};
use thresh_data::orbital::{Tle, parse_tle, propagate_tle};

/// Sample offsets from the TLE epoch, in seconds (task 8.1).
const OFFSETS_S: [f64; 4] = [0.0, 60.0, 300.0, 600.0];

/// One object to fixture: name, verbatim TLE lines, output file stem, and a
/// provenance note describing where the TLE comes from.
struct FixtureObject {
    name: &'static str,
    line1: &'static str,
    line2: &'static str,
    file_stem: &'static str,
    provenance: &'static str,
}

/// The committed fixture set: three LEO objects spanning distinct regimes
/// (ISS-band 51.6° at ~420 km, 58° moderate-drag at ~400 km perigee, and a
/// sun-synchronous near-circular orbit at ~840 km).
const OBJECTS: [FixtureObject; 3] = [
    FixtureObject {
        name: "ISS (ZARYA)",
        // Copied verbatim from crates/thresh-data/scenarios/orbital-iss.tle
        // (the cached TLE the orbital CI benchmark already uses).
        line1: "1 25544U 98067A   24001.00000000  .00016717  00000-0  10270-3 0  9026",
        line2: "2 25544  51.6400 208.9163 0006703  30.1579 330.0018 15.49560455    18",
        file_stem: "sgp4-25544-iss",
        provenance: "cached repo TLE `crates/thresh-data/scenarios/orbital-iss.tle`",
    },
    FixtureObject {
        name: "AIAA object 06251 (DELTA 1 DEB)",
        // Copied verbatim from the AIAA-2006-6753 SGP4 verification TLE set
        // ("near Earth normal drag case", perigee 377.26 km), as shipped in
        // the sgp4 crate's test_cases.toml.
        line1: "1 06251U 62025E   06176.82412014  .00008885  00000-0  12808-3 0  3985",
        line2: "2 06251  58.0579  54.0425 0030035 139.1568 221.1854 15.56387291  6774",
        file_stem: "sgp4-06251-delta1deb",
        provenance: "AIAA-2006-6753 verification TLE set (sgp4 crate `test_cases.toml`)",
    },
    FixtureObject {
        name: "AIAA object 28057 (sun-synchronous LEO)",
        // Copied verbatim from the AIAA-2006-6753 SGP4 verification TLE set
        // ("near Earth normal drag case but with low eccentricity"), as
        // shipped in the sgp4 crate's test_cases.toml.
        line1: "1 28057U 03049A   06177.78615833  .00000060  00000-0  35940-4 0  1836",
        line2: "2 28057  98.4283 247.6961 0000884  88.1964 271.9322 14.35478080140550",
        file_stem: "sgp4-28057-sunsync",
        provenance: "AIAA-2006-6753 verification TLE set (sgp4 crate `test_cases.toml`)",
    },
];

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(err) => {
            eprintln!("gen-sgp4-fixtures: {err}");
            ExitCode::from(1)
        }
    }
}

/// Top-level phase sequence: resolve the output directory, build and write
/// one fixture per object, then write the provenance record.
fn run() -> Result<(), String> {
    let out_dir = golden_orbital_dir();
    fs::create_dir_all(&out_dir).map_err(|e| format!("create {}: {e}", out_dir.display()))?;

    for obj in &OBJECTS {
        let fixture = build_fixture(obj)?;
        write_fixture(&out_dir, obj.file_stem, &fixture)?;
    }

    let sgp4_version = sgp4_version_from_lock()?;
    write_provenance(&out_dir, &sgp4_version)?;
    println!(
        "wrote {} fixtures + PROVENANCE.md to {} (sgp4 {sgp4_version})",
        OBJECTS.len(),
        out_dir.display()
    );
    Ok(())
}

/// Parse one object's TLE and propagate it with SGP4 at the fixed offsets,
/// converting the km / km·s⁻¹ crate output to metres / m·s⁻¹.
fn build_fixture(obj: &FixtureObject) -> Result<Sgp4Fixture, String> {
    let tle = parse_single_tle(obj)?;

    let times_min: Vec<f64> = OFFSETS_S.iter().map(|s| s / 60.0).collect();
    let states = propagate_tle(&tle, &times_min)
        .map_err(|e| format!("{}: SGP4 propagation failed: {e}", obj.name))?;

    let samples = OFFSETS_S
        .iter()
        .zip(&states)
        .map(|(&offset_s, st)| Sgp4Sample {
            offset_s,
            position_m: st.position_km.map(|c| c * 1000.0),
            velocity_m_s: st.velocity_km_s.map(|c| c * 1000.0),
        })
        .collect();

    Ok(Sgp4Fixture {
        object: obj.name.to_string(),
        norad_id: tle.norad_id,
        tle_line1: obj.line1.to_string(),
        tle_line2: obj.line2.to_string(),
        epoch_jd: tle.epoch_jd(),
        frame: "TEME".to_string(),
        samples,
    })
}

/// Run the object's two TLE lines through the crate's standard 2LE parser
/// and attach the display name.
fn parse_single_tle(obj: &FixtureObject) -> Result<Tle, String> {
    let text = format!("{}\n{}\n", obj.line1, obj.line2);
    let mut parsed =
        parse_tle(&text).map_err(|e| format!("{}: TLE parse failed: {e}", obj.name))?;
    let mut tle = parsed
        .pop()
        .ok_or_else(|| format!("{}: TLE parse yielded no entries", obj.name))?;
    tle.name = obj.name.to_string();
    Ok(tle)
}

/// Serialize one fixture as pretty JSON (trailing newline for
/// committed-file hygiene).
fn write_fixture(out_dir: &Path, file_stem: &str, fixture: &Sgp4Fixture) -> Result<(), String> {
    let path = out_dir.join(format!("{file_stem}.json"));
    let mut json = serde_json::to_string_pretty(fixture)
        .map_err(|e| format!("{file_stem}: serialize failed: {e}"))?;
    json.push('\n');
    fs::write(&path, json).map_err(|e| format!("write {}: {e}", path.display()))?;
    println!("wrote {}", path.display());
    Ok(())
}

/// Read the `sgp4` package version out of the workspace `Cargo.lock` so the
/// provenance record always matches the crate that actually generated the
/// fixtures.
fn sgp4_version_from_lock() -> Result<String, String> {
    let lock_path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("Cargo.lock");
    let lock =
        fs::read_to_string(&lock_path).map_err(|e| format!("read {}: {e}", lock_path.display()))?;

    let mut lines = lock.lines();
    while let Some(line) = lines.next() {
        if line.trim() != "name = \"sgp4\"" {
            continue;
        }
        let version_line = lines
            .next()
            .ok_or_else(|| "Cargo.lock: truncated sgp4 package entry".to_string())?;
        return version_line
            .trim()
            .strip_prefix("version = \"")
            .and_then(|v| v.strip_suffix('"'))
            .map(str::to_string)
            .ok_or_else(|| format!("Cargo.lock: unexpected sgp4 version line: {version_line}"));
    }
    Err("Cargo.lock: no sgp4 package entry found".to_string())
}

/// Write `PROVENANCE.md`: the TLEs verbatim, the sgp4 crate version, and
/// the regeneration command (task 8.1).
fn write_provenance(out_dir: &Path, sgp4_version: &str) -> Result<(), String> {
    let mut md = String::new();
    md.push_str(
        "# SGP4 Golden Fixture Provenance\n\n\
         Tier-3 golden vectors for design Decision 8 of the\n\
         `orbital-ballistic-filter-models` change: TEME position/velocity samples\n\
         at t0 and t0 + {60, 300, 600} s, propagated by the AIAA-verified [`sgp4`\n\
         crate](https://crates.io/crates/sgp4) and committed so CI never\n\
         regenerates them. Units in the JSON files are metres and metres/second.\n\n\
         Consumed by the default-feature envelope test\n\
         `crates/thresh-data/tests/golden_orbital.rs`: `KeplerJ2` initialized from\n\
         each fixture's t0 state must stay within 5 km position error of the SGP4\n\
         samples out to 600 s. The envelope is deliberately generous — SGP4\n\
         carries drag and short-periodic terms that Kepler+J2 osculating physics\n\
         legitimately lacks; exactness is certified by golden tiers 1 and 2.\n\n",
    );

    md.push_str("## Generation record\n\n");
    md.push_str(&format!(
        "- `sgp4` crate version: **{sgp4_version}** (from the workspace `Cargo.lock` at generation time)\n\
         - Generator: `crates/thresh-data/src/bin/gen_sgp4_fixtures.rs` (feature-gated, never in CI)\n\
         - Regeneration command (run from the workspace root, then commit the output):\n\n\
         ```sh\n\
         cargo run -p thresh-data --features orbital --bin gen-sgp4-fixtures\n\
         ```\n\n\
         Repeated runs at the same `sgp4` version are bit-identical (constant\n\
         inline TLEs, deterministic `serde_json` float formatting).\n\n",
    ));

    md.push_str("## Objects and TLEs (verbatim)\n\n");
    for obj in &OBJECTS {
        md.push_str(&format!(
            "### {} — `{}.json`\n\nSource: {}.\n\n```text\n{}\n{}\n```\n\n",
            obj.name, obj.file_stem, obj.provenance, obj.line1, obj.line2
        ));
    }

    let path = out_dir.join("PROVENANCE.md");
    fs::write(&path, md).map_err(|e| format!("write {}: {e}", path.display()))?;
    println!("wrote {}", path.display());
    Ok(())
}
