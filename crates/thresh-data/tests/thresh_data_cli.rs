//! End-to-end integration tests for the `thresh-data` CLI binary.
//!
//! These spawn the compiled binary via `env!("CARGO_BIN_EXE_thresh-data")`
//! (which `cargo test` builds automatically when the test runs) so the
//! `main` / `dispatch` / `cmd_list` / `cmd_run` / `run_manifest` /
//! `print_result` / `check_and_report_regression` / `parse_dir_arg` /
//! `default_scenario_dir` paths are all covered — unit tests alone would
//! leave most of the bin target uncovered by SonarCloud / Codecov.

use std::path::PathBuf;
use std::process::Command;

fn bin_path() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_thresh-data"))
}

/// Minimal valid scenario TOML that the synthetic runner can actually run.
const SYNTH_CV_CLEAN_TOML: &str = r#"
name = "test-synth"
description = "Integration test scenario"
source = "Synthetic"

[parameters]
duration_s = 10.0
dt = 1.0
measurement_noise_sigma = 50.0
gate_threshold = 500.0

[baselines]
mota = 0.3
"#;

const ADSB_TOML: &str = r#"
name = "test-adsb"
description = "ADS-B scenario (should error out)"
source = { AdsB = { region = "JFK" } }

[parameters]
duration_s = 10.0
dt = 1.0
measurement_noise_sigma = 50.0
gate_threshold = 500.0
"#;

/// Orbital scenario that references NO cached TLE file and real NORAD IDs
/// that no offline test environment can fetch. Used by `run_orbital_errors_out`
/// to verify that the CLI exits non-zero when the orbital runner cannot
/// source any TLEs. This is the expected failure mode with both binary
/// builds — without `orbital` it errors with "feature required"; with
/// `orbital` it errors with the HTTP fetch failure surfaced by
/// `CelestrakClient` in a sandboxed environment.
const ORBITAL_TOML_NO_TLE: &str = r#"
name = "test-orbital"
description = "Orbital scenario (should error out)"
[source.Orbital]
norad_ids = [25544, 48274]
station_lat_deg = 38.8339
station_lon_deg = -104.8214
station_alt_m = 1885.0

[parameters]
duration_s = 10.0
dt = 1.0
measurement_noise_sigma = 50.0
gate_threshold = 500.0
"#;

const BAD_BASELINE_TOML: &str = r#"
name = "bad-baseline"
description = "Scenario whose baseline is impossible to meet"
source = "Synthetic"

[parameters]
duration_s = 10.0
dt = 1.0
measurement_noise_sigma = 50.0
gate_threshold = 500.0

[baselines]
mota = 1.5
"#;

fn write_scenario(dir: &std::path::Path, name: &str, contents: &str) -> PathBuf {
    let path = dir.join(name);
    std::fs::write(&path, contents).expect("write scenario toml");
    path
}

// ---------------------------------------------------------------------------
// help / dispatch
// ---------------------------------------------------------------------------

#[test]
fn help_subcommand_prints_usage() {
    let out = Command::new(bin_path())
        .arg("help")
        .output()
        .expect("spawn thresh-data");
    assert!(out.status.success());
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("thresh-data list"));
    assert!(stderr.contains("thresh-data run"));
}

#[test]
fn help_flag_prints_usage() {
    let out = Command::new(bin_path())
        .arg("--help")
        .output()
        .expect("spawn thresh-data");
    assert!(out.status.success());
    assert!(String::from_utf8_lossy(&out.stderr).contains("thresh-data list"));
}

#[test]
fn no_args_prints_usage() {
    let out = Command::new(bin_path())
        .output()
        .expect("spawn thresh-data");
    assert!(out.status.success());
    assert!(String::from_utf8_lossy(&out.stderr).contains("thresh-data list"));
}

#[test]
fn unknown_subcommand_errors() {
    let out = Command::new(bin_path())
        .arg("nope")
        .output()
        .expect("spawn thresh-data");
    assert!(!out.status.success());
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("unknown subcommand") && stderr.contains("nope"));
}

// ---------------------------------------------------------------------------
// list
// ---------------------------------------------------------------------------

#[test]
fn list_walks_directory() {
    let dir = tempfile::tempdir().unwrap();
    write_scenario(dir.path(), "synth.toml", SYNTH_CV_CLEAN_TOML);
    write_scenario(dir.path(), "adsb.toml", ADSB_TOML);
    // A non-.toml file must be skipped silently.
    std::fs::write(dir.path().join("notes.md"), "ignore me").unwrap();

    let out = Command::new(bin_path())
        .args(["list", "--dir"])
        .arg(dir.path())
        .output()
        .expect("spawn thresh-data");
    assert!(
        out.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.contains("test-synth"));
    assert!(stdout.contains("test-adsb"));
    assert!(stdout.contains("Synthetic"));
    assert!(stdout.contains("AdsB(region=JFK)"));
}

#[test]
fn list_missing_dir_errors() {
    let out = Command::new(bin_path())
        .args(["list", "--dir", "/definitely/does/not/exist/xyz"])
        .output()
        .expect("spawn thresh-data");
    assert!(!out.status.success());
    assert!(String::from_utf8_lossy(&out.stderr).contains("does not exist"));
}

#[test]
fn list_empty_dir_succeeds() {
    let dir = tempfile::tempdir().unwrap();
    let out = Command::new(bin_path())
        .args(["list", "--dir"])
        .arg(dir.path())
        .output()
        .expect("spawn thresh-data");
    assert!(out.status.success());
    assert!(String::from_utf8_lossy(&out.stdout).contains("no scenarios found"));
}

#[test]
fn list_skips_unparseable_toml() {
    let dir = tempfile::tempdir().unwrap();
    write_scenario(dir.path(), "ok.toml", SYNTH_CV_CLEAN_TOML);
    write_scenario(dir.path(), "broken.toml", "this is not valid = [");
    let out = Command::new(bin_path())
        .args(["list", "--dir"])
        .arg(dir.path())
        .output()
        .expect("spawn thresh-data");
    assert!(out.status.success());
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    // The good file is listed; the bad file emits a skip message to stderr.
    assert!(stdout.contains("test-synth"));
    assert!(stderr.contains("skip"));
}

#[test]
fn list_default_dir_via_env() {
    let dir = tempfile::tempdir().unwrap();
    write_scenario(dir.path(), "synth.toml", SYNTH_CV_CLEAN_TOML);
    let out = Command::new(bin_path())
        .arg("list")
        .env("THRESH_DATA_SCENARIOS", dir.path())
        .output()
        .expect("spawn thresh-data");
    assert!(out.status.success());
    assert!(String::from_utf8_lossy(&out.stdout).contains("test-synth"));
}

// ---------------------------------------------------------------------------
// run
// ---------------------------------------------------------------------------

#[test]
fn run_synthetic_prints_metrics_and_regression_ok() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_scenario(dir.path(), "synth.toml", SYNTH_CV_CLEAN_TOML);
    let out = Command::new(bin_path())
        .arg("run")
        .arg(&path)
        .output()
        .expect("spawn thresh-data");
    assert!(
        out.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.contains("scenario:"));
    assert!(stdout.contains("MOTA:"));
    assert!(stdout.contains("MOTP:"));
    assert!(stdout.contains("IDF1:"));
    assert!(stdout.contains("HOTA:"));
    assert!(stdout.contains("ID switches:"));
    assert!(stdout.contains("regression: OK"));
}

#[test]
fn run_synthetic_reports_regression_failure() {
    // Baseline MOTA=1.5 is impossible to reach, so the regression check
    // must fail and the CLI must exit non-zero.
    let dir = tempfile::tempdir().unwrap();
    let path = write_scenario(dir.path(), "bad.toml", BAD_BASELINE_TOML);
    let out = Command::new(bin_path())
        .arg("run")
        .arg(&path)
        .output()
        .expect("spawn thresh-data");
    assert!(!out.status.success());
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.contains("regression: FAIL"));
    assert!(stdout.contains("MOTA"));
}

#[test]
fn run_adsb_errors_out() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_scenario(dir.path(), "adsb.toml", ADSB_TOML);
    let out = Command::new(bin_path())
        .arg("run")
        .arg(&path)
        .output()
        .expect("spawn thresh-data");
    assert!(!out.status.success());
    assert!(String::from_utf8_lossy(&out.stderr).contains("ADS-B"));
}

#[test]
fn run_orbital_errors_out_without_tle_or_network() {
    // With no `tle_file` set and no cached TLEs on disk, the orbital
    // runner must exit non-zero regardless of whether the binary was
    // built with `--features orbital`:
    //   - Without the feature: errors with "feature required".
    //   - With the feature: errors when `CelestrakClient` can't reach
    //     the network (sandbox-friendly assumption).
    let dir = tempfile::tempdir().unwrap();
    let path = write_scenario(dir.path(), "orbital.toml", ORBITAL_TOML_NO_TLE);
    let out = Command::new(bin_path())
        .arg("run")
        .arg(&path)
        .output()
        .expect("spawn thresh-data");
    assert!(!out.status.success());
    // Accept any of the expected error phrases so this test is agnostic
    // to which feature flags the binary was built with.
    let stderr = String::from_utf8_lossy(&out.stderr).to_lowercase();
    assert!(
        stderr.contains("orbital")
            || stderr.contains("feature required")
            || stderr.contains("celestrak")
            || stderr.contains("http")
            || stderr.contains("fetch"),
        "unexpected stderr: {stderr}"
    );
}

#[test]
fn run_without_path_errors() {
    let out = Command::new(bin_path())
        .arg("run")
        .output()
        .expect("spawn thresh-data");
    assert!(!out.status.success());
    assert!(String::from_utf8_lossy(&out.stderr).contains(".toml"));
}

#[test]
fn run_nonexistent_file_errors() {
    let out = Command::new(bin_path())
        .arg("run")
        .arg("/definitely/does/not/exist.toml")
        .output()
        .expect("spawn thresh-data");
    assert!(!out.status.success());
}

// ---------------------------------------------------------------------------
// Orbital end-to-end (feature-gated)
// ---------------------------------------------------------------------------

/// When the binary is built with `--features orbital` and the scenario
/// references a cached TLE file, the orbital runner must complete
/// successfully with the full pipeline (SGP4 → ENU → measurement →
/// tracker → metrics) and exit 0. Runs offline because the TLE is
/// checked into the repo under `crates/thresh-data/scenarios/`.
#[cfg(feature = "orbital")]
#[test]
fn run_orbital_iss_cached_tle_end_to_end() {
    // Scenario manifest lives alongside its cached TLE in the repo; we
    // point the CLI at the committed file directly so the test is
    // effectively the same regression the `orbital-benchmark-gate` CI
    // job runs.
    let manifest = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("scenarios")
        .join("orbital-iss.toml");
    assert!(manifest.exists(), "fixture {} missing", manifest.display());
    let out = Command::new(bin_path())
        .arg("run")
        .arg(&manifest)
        .output()
        .expect("spawn thresh-data");
    assert!(
        out.status.success(),
        "orbital-iss scenario exited non-zero. stdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.contains("scenario:    orbital-iss"));
    assert!(stdout.contains("MOTA:"));
    assert!(stdout.contains("regression: OK"));
}

#[cfg(feature = "orbital")]
#[test]
fn run_orbital_starlink_train_cached_tle_end_to_end() {
    let manifest = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("scenarios")
        .join("orbital-starlink-train.toml");
    assert!(manifest.exists(), "fixture missing");
    let out = Command::new(bin_path())
        .arg("run")
        .arg(&manifest)
        .output()
        .expect("spawn thresh-data");
    assert!(
        out.status.success(),
        "starlink-train scenario exited non-zero. stdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.contains("scenario:    orbital-starlink-train"));
    assert!(stdout.contains("regression: OK"));
}
