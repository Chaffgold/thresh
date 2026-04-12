//! `thresh-data` — benchmark scenario runner CLI.
//!
//! Subcommands:
//!
//! ```text
//! thresh-data list
//! thresh-data run <scenario.toml>
//! ```
//!
//! `list` walks the default scenario directory (`crates/thresh-data/scenarios/`
//! relative to the workspace root, or the directory set via
//! `THRESH_DATA_SCENARIOS`) and prints the manifest name and description for
//! each `*.toml` file it finds.
//!
//! `run` loads a manifest by path, dispatches to the benchmark runner for its
//! source type, and prints the resulting MOT metrics. Only `Synthetic`
//! sources run end-to-end today; `AdsB` and `Orbital` sources print a
//! "feature required" stub and exit with a non-zero status so CI can
//! distinguish "broken pipeline" from "feature not enabled".

use std::path::PathBuf;
use std::process::ExitCode;

use thresh_data::benchmark::{
    BenchmarkResult, ScenarioManifest, ScenarioSource, load_scenario, run_synthetic_benchmark,
};

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match dispatch(&args) {
        Ok(()) => ExitCode::SUCCESS,
        Err(err) => {
            eprintln!("thresh-data: {err}");
            ExitCode::from(1)
        }
    }
}

fn dispatch(args: &[String]) -> Result<(), String> {
    let sub = args.first().map(String::as_str).unwrap_or("");
    match sub {
        "list" => cmd_list(&args[1..]),
        "run" => cmd_run(&args[1..]),
        "-h" | "--help" | "help" | "" => {
            print_usage();
            Ok(())
        }
        other => Err(format!(
            "unknown subcommand `{other}` — try `thresh-data help`"
        )),
    }
}

fn print_usage() {
    eprintln!(
        "Usage:\n\
         \x20 thresh-data list [--dir <path>]        List available scenario manifests\n\
         \x20 thresh-data run <scenario.toml>        Run a scenario and print metrics\n\
         \x20 thresh-data help                       Print this message\n\
         \n\
         Scenario directory:\n\
         \x20 Defaults to ./crates/thresh-data/scenarios when run from the workspace root.\n\
         \x20 Override via `THRESH_DATA_SCENARIOS` or `--dir <path>` on the `list` subcommand."
    );
}

// ---------------------------------------------------------------------------
// list
// ---------------------------------------------------------------------------

fn cmd_list(args: &[String]) -> Result<(), String> {
    let dir = parse_dir_arg(args)?.unwrap_or_else(default_scenario_dir);
    if !dir.exists() {
        return Err(format!(
            "scenario directory {} does not exist (set THRESH_DATA_SCENARIOS or pass --dir)",
            dir.display()
        ));
    }

    let mut manifests: Vec<(PathBuf, ScenarioManifest)> = Vec::new();
    let entries =
        std::fs::read_dir(&dir).map_err(|e| format!("failed to read {}: {e}", dir.display()))?;
    for entry in entries {
        let entry = entry.map_err(|e| format!("dir entry error: {e}"))?;
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) != Some("toml") {
            continue;
        }
        match load_scenario(&path) {
            Ok(m) => manifests.push((path, m)),
            Err(e) => eprintln!("  skip {}: {e}", path.display()),
        }
    }

    manifests.sort_by(|a, b| a.1.name.cmp(&b.1.name));

    if manifests.is_empty() {
        println!("(no scenarios found in {})", dir.display());
        return Ok(());
    }

    println!("Available scenarios in {}:", dir.display());
    for (path, m) in &manifests {
        println!("  {} — {}", m.name, m.description);
        println!("      source: {}", describe_source(&m.source));
        println!("      file:   {}", path.display());
    }
    Ok(())
}

fn parse_dir_arg(args: &[String]) -> Result<Option<PathBuf>, String> {
    let mut it = args.iter();
    while let Some(a) = it.next() {
        if a == "--dir" {
            let v = it
                .next()
                .ok_or_else(|| "--dir requires a path".to_string())?;
            return Ok(Some(PathBuf::from(v)));
        }
    }
    Ok(None)
}

fn default_scenario_dir() -> PathBuf {
    if let Ok(env) = std::env::var("THRESH_DATA_SCENARIOS") {
        PathBuf::from(env)
    } else {
        PathBuf::from("crates/thresh-data/scenarios")
    }
}

fn describe_source(src: &ScenarioSource) -> String {
    match src {
        ScenarioSource::Synthetic => "Synthetic".into(),
        ScenarioSource::AdsB { region } => format!("AdsB(region={region})"),
        ScenarioSource::Orbital { norad_ids, .. } => {
            format!("Orbital({} satellites)", norad_ids.len())
        }
    }
}

// ---------------------------------------------------------------------------
// run
// ---------------------------------------------------------------------------

fn cmd_run(args: &[String]) -> Result<(), String> {
    let path = args
        .first()
        .map(PathBuf::from)
        .ok_or_else(|| "run requires a scenario .toml path".to_string())?;
    let manifest = load_scenario(&path)?;
    // Resolve the manifest's parent so `tle_file` entries with relative
    // paths are looked up next to the `.toml` file, matching the
    // convention documented on `ScenarioSource::Orbital`.
    let manifest_dir = path
        .parent()
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("."));
    run_manifest(&manifest, &manifest_dir)
}

fn run_manifest(manifest: &ScenarioManifest, manifest_dir: &std::path::Path) -> Result<(), String> {
    match &manifest.source {
        ScenarioSource::Synthetic => {
            let result = run_synthetic_benchmark(manifest);
            print_result(&result);
            check_and_report_regression(manifest, &result)
        }
        ScenarioSource::AdsB { region } => Err(format!(
            "ADS-B scenarios (region={region}) require fetching live data; \
             run `thresh-data fetch` first (not yet implemented)"
        )),
        ScenarioSource::Orbital { norad_ids, .. } => {
            run_orbital_dispatch(manifest, manifest_dir, norad_ids)
        }
    }
}

/// Dispatch an orbital scenario to `run_orbital_benchmark` when the
/// `orbital` feature is enabled; otherwise surface a clear "feature
/// required" error so the failure mode is actionable.
#[cfg(feature = "orbital")]
fn run_orbital_dispatch(
    manifest: &ScenarioManifest,
    manifest_dir: &std::path::Path,
    _norad_ids: &[u32],
) -> Result<(), String> {
    let result = thresh_data::benchmark::run_orbital_benchmark(manifest, manifest_dir)?;
    print_result(&result);
    check_and_report_regression(manifest, &result)
}

#[cfg(not(feature = "orbital"))]
fn run_orbital_dispatch(
    _manifest: &ScenarioManifest,
    _manifest_dir: &std::path::Path,
    norad_ids: &[u32],
) -> Result<(), String> {
    Err(format!(
        "Orbital scenarios ({} satellites) require the `orbital` feature. \
         Rebuild with `cargo build -p thresh-data --features orbital --bin thresh-data`.",
        norad_ids.len()
    ))
}

fn print_result(result: &BenchmarkResult) {
    println!("scenario:    {}", result.scenario);
    println!("MOTA:        {:.4}", result.mota);
    println!("MOTP:        {:.4}", result.motp);
    println!("IDF1:        {:.4}", result.idf1);
    println!("HOTA:        {:.4}", result.hota);
    println!("ID switches: {}", result.id_switches);
    println!("duration:    {} ms", result.duration_ms);
}

fn check_and_report_regression(
    manifest: &ScenarioManifest,
    result: &BenchmarkResult,
) -> Result<(), String> {
    let Some(baselines) = &manifest.baselines else {
        return Ok(());
    };
    let failures = thresh_data::benchmark::check_regression(result, baselines);
    if failures.is_empty() {
        println!("regression: OK");
        Ok(())
    } else {
        println!("regression: FAIL");
        for f in &failures {
            println!("  {f}");
        }
        Err(format!("{} regression check(s) failed", failures.len()))
    }
}

// ---------------------------------------------------------------------------
// Tests (argument dispatch + list walking only — run_manifest path is
// exercised by the existing benchmark tests in `benchmark.rs`)
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn describe_source_strings() {
        assert_eq!(describe_source(&ScenarioSource::Synthetic), "Synthetic");
        assert_eq!(
            describe_source(&ScenarioSource::AdsB {
                region: "JFK".into()
            }),
            "AdsB(region=JFK)"
        );
        assert_eq!(
            describe_source(&ScenarioSource::Orbital {
                norad_ids: vec![25544, 48274],
                tle_file: None,
                station_lat_deg: 0.0,
                station_lon_deg: 0.0,
                station_alt_m: 0.0,
                time_step_s: None,
            }),
            "Orbital(2 satellites)"
        );
    }

    #[test]
    fn parse_dir_arg_extracts_path() {
        let args = vec!["--dir".into(), "/tmp/scenarios".into()];
        let parsed = parse_dir_arg(&args).unwrap();
        assert_eq!(parsed, Some(PathBuf::from("/tmp/scenarios")));
    }

    #[test]
    fn parse_dir_arg_missing_value_errors() {
        let args = vec!["--dir".into()];
        assert!(parse_dir_arg(&args).is_err());
    }

    #[test]
    fn parse_dir_arg_absent_returns_none() {
        let args: Vec<String> = vec![];
        assert_eq!(parse_dir_arg(&args).unwrap(), None);
    }
}
