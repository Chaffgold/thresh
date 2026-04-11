//! RCS computation bridge to PyPOFacets.
//!
//! Pure-Rust configuration and result types (`RcsSweepConfig`, `RcsSample`,
//! `RcsSweepResult`) are always available so they can be constructed,
//! serialised, and unit-tested without a Python install. The actual PyO3
//! bridge to the Python `pofacets` package is gated behind the `rcs-compute`
//! Cargo feature.

use serde::{Deserialize, Serialize};

#[cfg(feature = "rcs-compute")]
use pyo3::prelude::*;
#[cfg(feature = "rcs-compute")]
use pyo3::types::PyAnyMethods;

// ── Configuration / result types (non-gated) ────────────────────────────────

/// Sweep configuration for RCS calculation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RcsSweepConfig {
    /// Frequency in GHz.
    pub frequency_ghz: f64,
    /// Azimuth start (degrees).
    pub az_start_deg: f64,
    /// Azimuth end (degrees).
    pub az_end_deg: f64,
    /// Azimuth step (degrees).
    pub az_step_deg: f64,
    /// Elevation angles for hemisphere sweeps (empty = single elevation only).
    pub el_angles_deg: Vec<f64>,
    /// Polarization: `"VV"`, `"HH"`, `"VH"`, `"HV"`.
    pub polarization: String,
}

impl Default for RcsSweepConfig {
    fn default() -> Self {
        // 10 GHz X-band, full 0-360° azimuth sweep at 5° steps,
        // single elevation at the horizon, co-polarized (VV).
        Self {
            frequency_ghz: 10.0,
            az_start_deg: 0.0,
            az_end_deg: 360.0,
            az_step_deg: 5.0,
            el_angles_deg: vec![0.0],
            polarization: "VV".to_string(),
        }
    }
}

/// Single RCS sample result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RcsSample {
    pub azimuth_deg: f64,
    pub elevation_deg: f64,
    pub rcs_dbsm: f64,
}

/// Full RCS sweep result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RcsSweepResult {
    pub frequency_ghz: f64,
    pub polarization: String,
    pub samples: Vec<RcsSample>,
}

impl RcsSweepResult {
    /// Serialise the sweep result as pretty-printed JSON.
    pub fn to_json(&self) -> serde_json::Result<String> {
        serde_json::to_string_pretty(self)
    }

    /// Write the sweep result as pretty-printed JSON to `path`.
    pub fn write_json(&self, path: &std::path::Path) -> std::io::Result<()> {
        let json = self
            .to_json()
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        std::fs::write(path, json)
    }
}

// ── Sweep → lookup table conversion (non-gated) ─────────────────────────────

/// Convert a sweep result into an [`RcsLookupTable`] compatible with
/// `swerling.rs`.
///
/// Samples are grouped by unique elevation and azimuth. The output table is
/// sorted in ascending order along both axes. Missing grid cells default to
/// the sweep's observed minimum so that bilinear lookup remains well-defined.
///
/// # Panics
///
/// Panics if `result.samples` is empty, since an empty grid cannot satisfy
/// the `RcsLookupTable` lookup contract (which requires at least one
/// azimuth and one elevation entry).
///
/// [`RcsLookupTable`]: crate::swerling::RcsLookupTable
pub fn sweep_to_lookup_table(result: &RcsSweepResult) -> crate::swerling::RcsLookupTable {
    assert!(
        !result.samples.is_empty(),
        "sweep_to_lookup_table requires at least one sample to build a valid RcsLookupTable"
    );

    // Collect unique azimuth / elevation values using an integer key for
    // stable de-duplication of floating-point grid points. BTreeMap gives
    // sorted-order keys and O(log n) insert/lookup per sample (O(n log n)
    // overall, vs the previous O(n²) Vec::contains/position scan).
    use std::collections::BTreeMap;

    fn key(x: f64) -> i64 {
        (x * 1_000.0).round() as i64
    }

    let mut az_idx: BTreeMap<i64, usize> = BTreeMap::new();
    let mut el_idx: BTreeMap<i64, usize> = BTreeMap::new();
    for s in &result.samples {
        let ak = key(s.azimuth_deg);
        let ek = key(s.elevation_deg);
        let next_a = az_idx.len();
        az_idx.entry(ak).or_insert(next_a);
        let next_e = el_idx.len();
        el_idx.entry(ek).or_insert(next_e);
    }

    // BTreeMap gives sorted key iteration; rebuild the index maps so indices
    // match the final sorted order.
    let azimuth_deg: Vec<f64> = az_idx.keys().map(|k| (*k as f64) / 1_000.0).collect();
    let elevation_deg: Vec<f64> = el_idx.keys().map(|k| (*k as f64) / 1_000.0).collect();
    let az_idx: BTreeMap<i64, usize> = az_idx.keys().enumerate().map(|(i, k)| (*k, i)).collect();
    let el_idx: BTreeMap<i64, usize> = el_idx.keys().enumerate().map(|(i, k)| (*k, i)).collect();

    let fill = result
        .samples
        .iter()
        .map(|s| s.rcs_dbsm)
        .fold(f64::INFINITY, f64::min);
    let fill = if fill.is_finite() { fill } else { 0.0 };

    let mut rcs_dbsm: Vec<Vec<f64>> = vec![vec![fill; elevation_deg.len()]; azimuth_deg.len()];
    for s in &result.samples {
        let ai = az_idx[&key(s.azimuth_deg)];
        let ei = el_idx[&key(s.elevation_deg)];
        rcs_dbsm[ai][ei] = s.rcs_dbsm;
    }

    crate::swerling::RcsLookupTable {
        azimuth_deg,
        elevation_deg,
        rcs_dbsm,
    }
}

// ── CLI argument parsing (non-gated, pure logic) ────────────────────────────

pub mod cli {
    //! CLI argument parsing for the `thresh-rcs-compute` binary.
    //!
    //! Kept in the library (not the `bin/` file) so the parser can be unit
    //! tested without pulling in the Python/pyo3 runtime that the binary
    //! itself links against.

    use super::RcsSweepConfig;
    use std::path::PathBuf;

    /// Parsed form of the `thresh-rcs-compute` command line.
    #[derive(Debug)]
    pub enum ParsedArgs {
        /// `--help` / `-h` was requested.
        Help,
        /// A validated run request.
        Run {
            stl: String,
            output: PathBuf,
            config: RcsSweepConfig,
        },
    }

    /// Parse CLI arguments for `thresh-rcs-compute`.
    ///
    /// Accepts a slice of arguments **without** the program name, mirroring
    /// `std::env::args().skip(1).collect()`.
    pub fn parse_args(args: &[String]) -> Result<ParsedArgs, String> {
        if args.iter().any(|a| a == "-h" || a == "--help") {
            return Ok(ParsedArgs::Help);
        }

        let mut stl: Option<String> = None;
        let mut output: Option<PathBuf> = None;
        let mut freq_ghz: Option<f64> = None;
        let mut step_deg: Option<f64> = None;
        let mut az_start_deg: Option<f64> = None;
        let mut az_end_deg: Option<f64> = None;
        let mut el_angles_deg: Vec<f64> = Vec::new();
        let mut polarization: Option<String> = None;

        let mut i = 0;
        while i < args.len() {
            let flag = args[i].as_str();
            let value_at = |i: usize| -> Result<&String, String> {
                args.get(i + 1)
                    .ok_or_else(|| format!("flag `{}` requires a value", args[i]))
            };
            match flag {
                "--stl" => {
                    stl = Some(value_at(i)?.clone());
                    i += 2;
                }
                "--output" => {
                    output = Some(PathBuf::from(value_at(i)?));
                    i += 2;
                }
                "--freq" => {
                    freq_ghz = Some(parse_f64(flag, value_at(i)?)?);
                    i += 2;
                }
                "--step" => {
                    step_deg = Some(parse_f64(flag, value_at(i)?)?);
                    i += 2;
                }
                "--az-start" => {
                    az_start_deg = Some(parse_f64(flag, value_at(i)?)?);
                    i += 2;
                }
                "--az-end" => {
                    az_end_deg = Some(parse_f64(flag, value_at(i)?)?);
                    i += 2;
                }
                "--el" => {
                    el_angles_deg.push(parse_f64(flag, value_at(i)?)?);
                    i += 2;
                }
                "--polarization" => {
                    let pol = value_at(i)?.clone();
                    if !matches!(pol.as_str(), "VV" | "HH" | "VH" | "HV") {
                        return Err(format!(
                            "--polarization must be one of VV, HH, VH, HV (got `{pol}`)"
                        ));
                    }
                    polarization = Some(pol);
                    i += 2;
                }
                other => return Err(format!("unknown flag `{other}`")),
            }
        }

        let stl = stl.ok_or_else(|| "missing required flag `--stl`".to_string())?;
        let output = output.ok_or_else(|| "missing required flag `--output`".to_string())?;
        let frequency_ghz = freq_ghz.ok_or_else(|| "missing required flag `--freq`".to_string())?;
        let az_step_deg = step_deg.ok_or_else(|| "missing required flag `--step`".to_string())?;

        if frequency_ghz <= 0.0 {
            return Err(format!("--freq must be > 0 (got {frequency_ghz})"));
        }
        if az_step_deg <= 0.0 {
            return Err(format!("--step must be > 0 (got {az_step_deg})"));
        }

        let defaults = RcsSweepConfig::default();
        let config = RcsSweepConfig {
            frequency_ghz,
            az_start_deg: az_start_deg.unwrap_or(defaults.az_start_deg),
            az_end_deg: az_end_deg.unwrap_or(defaults.az_end_deg),
            az_step_deg,
            el_angles_deg: if el_angles_deg.is_empty() {
                defaults.el_angles_deg
            } else {
                el_angles_deg
            },
            polarization: polarization.unwrap_or(defaults.polarization),
        };

        if config.az_end_deg <= config.az_start_deg {
            return Err(format!(
                "--az-end ({}) must be greater than --az-start ({})",
                config.az_end_deg, config.az_start_deg
            ));
        }

        Ok(ParsedArgs::Run {
            stl,
            output,
            config,
        })
    }

    fn parse_f64(flag: &str, value: &str) -> Result<f64, String> {
        let parsed: f64 = value
            .parse::<f64>()
            .map_err(|e| format!("flag `{flag}` expected a number, got `{value}`: {e}"))?;
        if !parsed.is_finite() {
            // `parse::<f64>()` happily accepts `NaN`, `inf`, `-inf`; reject
            // them here so they don't slip past the `> 0.0` guard (NaN
            // comparisons are always false) and reach the PyPOFacets bridge.
            return Err(format!(
                "flag `{flag}` expected a finite number, got `{value}`"
            ));
        }
        Ok(parsed)
    }

    /// Return the expected number of RCS samples for a given sweep config.
    /// Used by the CLI to print a summary after a successful run. Counts the
    /// azimuth samples as `floor(span / step) + 1`, matching a sweep that
    /// iterates `az = start, start+step, …` while `az <= end`.
    pub fn config_sample_count(config: &RcsSweepConfig) -> usize {
        let span = config.az_end_deg - config.az_start_deg;
        let n_az = (span / config.az_step_deg).floor() as usize + 1;
        let n_el = config.el_angles_deg.len().max(1);
        n_az * n_el
    }

    /// Human-readable usage string printed on `--help` or error.
    pub fn usage_text() -> &'static str {
        "Usage: thresh-rcs-compute --stl <file> --freq <GHz> --step <deg> --output <file> \\\n\
         \x20                         [--az-start <deg>] [--az-end <deg>] [--el <deg>]... \\\n\
         \x20                         [--polarization VV|HH|VH|HV]\n\
         \n\
         Compute a monostatic RCS sweep for an STL target via the PyPOFacets bridge\n\
         and write the result as JSON compatible with `RcsLookupTable`.\n\
         \n\
         Required:\n\
         \x20 --stl <file>          STL geometry file to load\n\
         \x20 --freq <GHz>          Operating frequency (GHz)\n\
         \x20 --step <deg>          Azimuth step size (degrees)\n\
         \x20 --output <file>       Output JSON path\n\
         \n\
         Optional:\n\
         \x20 --az-start <deg>      Azimuth sweep start (default 0)\n\
         \x20 --az-end <deg>        Azimuth sweep end (default 360)\n\
         \x20 --el <deg>            Elevation angle; repeat for hemisphere sweep (default 0)\n\
         \x20 --polarization <pol>  VV | HH | VH | HV (default VV)\n\
         \x20 -h, --help            Print this help and exit"
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        fn args(s: &[&str]) -> Vec<String> {
            s.iter().map(|a| a.to_string()).collect()
        }

        #[test]
        fn help_flag_returns_help() {
            assert!(matches!(
                parse_args(&args(&["--help"])).unwrap(),
                ParsedArgs::Help
            ));
            assert!(matches!(
                parse_args(&args(&["-h"])).unwrap(),
                ParsedArgs::Help
            ));
        }

        #[test]
        fn minimal_required_args_parse() {
            let parsed = parse_args(&args(&[
                "--stl",
                "sphere.stl",
                "--freq",
                "10.0",
                "--step",
                "5.0",
                "--output",
                "out.json",
            ]))
            .unwrap();
            let ParsedArgs::Run {
                stl,
                output,
                config,
            } = parsed
            else {
                panic!("expected Run");
            };
            assert_eq!(stl, "sphere.stl");
            assert_eq!(output, PathBuf::from("out.json"));
            assert_eq!(config.frequency_ghz, 10.0);
            assert_eq!(config.az_step_deg, 5.0);
            assert_eq!(config.az_start_deg, 0.0);
            assert_eq!(config.az_end_deg, 360.0);
            assert_eq!(config.polarization, "VV");
            assert_eq!(config.el_angles_deg, vec![0.0]);
        }

        #[test]
        fn multiple_el_flags_build_hemisphere() {
            let parsed = parse_args(&args(&[
                "--stl", "x.stl", "--freq", "10", "--step", "10", "--output", "o.json", "--el",
                "0", "--el", "15", "--el", "30",
            ]))
            .unwrap();
            let ParsedArgs::Run { config, .. } = parsed else {
                panic!()
            };
            assert_eq!(config.el_angles_deg, vec![0.0, 15.0, 30.0]);
        }

        #[test]
        fn missing_required_flag_errors() {
            let err =
                parse_args(&args(&["--stl", "x.stl", "--freq", "10", "--step", "5"])).unwrap_err();
            assert!(err.contains("--output"), "got: {err}");
        }

        #[test]
        fn invalid_polarization_errors() {
            let err = parse_args(&args(&[
                "--stl",
                "x.stl",
                "--freq",
                "10",
                "--step",
                "5",
                "--output",
                "o.json",
                "--polarization",
                "XY",
            ]))
            .unwrap_err();
            assert!(err.contains("VV"), "got: {err}");
        }

        #[test]
        fn non_positive_freq_errors() {
            let err = parse_args(&args(&[
                "--stl", "x.stl", "--freq", "-1", "--step", "5", "--output", "o.json",
            ]))
            .unwrap_err();
            assert!(err.contains("--freq must be > 0"), "got: {err}");
        }

        #[test]
        fn az_end_must_exceed_az_start() {
            let err = parse_args(&args(&[
                "--stl",
                "x.stl",
                "--freq",
                "10",
                "--step",
                "5",
                "--output",
                "o.json",
                "--az-start",
                "180",
                "--az-end",
                "90",
            ]))
            .unwrap_err();
            assert!(err.contains("--az-end"), "got: {err}");
        }

        #[test]
        fn unknown_flag_errors() {
            let err = parse_args(&args(&["--nope", "value"])).unwrap_err();
            assert!(err.contains("--nope"), "got: {err}");
        }

        #[test]
        fn flag_without_value_errors() {
            let err = parse_args(&args(&["--stl"])).unwrap_err();
            assert!(err.contains("requires a value"), "got: {err}");
        }

        #[test]
        fn nan_numeric_rejected() {
            // `parse::<f64>()` accepts "NaN"; the parser must reject it
            // before the `> 0.0` guard (which is always false for NaN).
            let err = parse_args(&args(&[
                "--stl", "x.stl", "--freq", "NaN", "--step", "5", "--output", "o.json",
            ]))
            .unwrap_err();
            assert!(err.contains("finite"), "got: {err}");
        }

        #[test]
        fn infinity_numeric_rejected() {
            let err = parse_args(&args(&[
                "--stl", "x.stl", "--freq", "inf", "--step", "5", "--output", "o.json",
            ]))
            .unwrap_err();
            assert!(err.contains("finite"), "got: {err}");
        }

        #[test]
        fn negative_infinity_rejected_on_step() {
            let err = parse_args(&args(&[
                "--stl", "x.stl", "--freq", "10", "--step", "-inf", "--output", "o.json",
            ]))
            .unwrap_err();
            assert!(err.contains("finite"), "got: {err}");
        }

        #[test]
        fn sample_count_azimuth_only() {
            let c = RcsSweepConfig {
                frequency_ghz: 10.0,
                az_start_deg: 0.0,
                az_end_deg: 360.0,
                az_step_deg: 5.0,
                el_angles_deg: vec![0.0],
                polarization: "VV".into(),
            };
            // 0..=360 step 5 → 73 azimuths × 1 elevation
            assert_eq!(config_sample_count(&c), 73);
        }

        #[test]
        fn sample_count_hemisphere() {
            let c = RcsSweepConfig {
                frequency_ghz: 10.0,
                az_start_deg: 0.0,
                az_end_deg: 360.0,
                az_step_deg: 10.0,
                el_angles_deg: vec![0.0, 15.0, 30.0],
                polarization: "VV".into(),
            };
            // 37 azimuths × 3 elevations = 111
            assert_eq!(config_sample_count(&c), 111);
        }

        #[test]
        fn sample_count_non_divisible_span() {
            // Regression: start=0, end=350, step=100 iterates
            // [0, 100, 200, 300] — 4 samples, not 5. The old
            // `ceil(span/step) + 1` formula would have returned 5.
            let c = RcsSweepConfig {
                frequency_ghz: 10.0,
                az_start_deg: 0.0,
                az_end_deg: 350.0,
                az_step_deg: 100.0,
                el_angles_deg: vec![0.0],
                polarization: "VV".into(),
            };
            assert_eq!(config_sample_count(&c), 4);
        }
    }
}

// ── PyO3 bridge (feature gated) ─────────────────────────────────────────────

/// Bridge to PyPOFacets for computing RCS from STL geometry.
#[cfg(feature = "rcs-compute")]
pub struct RcsComputeBridge {
    /// Path to the STL file.
    pub stl_path: String,
}

#[cfg(feature = "rcs-compute")]
impl RcsComputeBridge {
    /// Create a new bridge pointing at an STL file on disk.
    pub fn new(stl_path: impl Into<String>) -> Self {
        Self {
            stl_path: stl_path.into(),
        }
    }

    /// Load the STL file via PyPOFacets and return the facet count.
    pub fn load_geometry(&self) -> PyResult<usize> {
        Python::with_gil(|py| {
            let pofacets = py.import("pofacets")?;
            let model = pofacets.call_method1("load_stl", (self.stl_path.clone(),))?;
            let count: usize = model.getattr("num_facets")?.extract()?;
            Ok(count)
        })
    }

    /// Monostatic RCS sweep at a single elevation cut.
    ///
    /// Calls `pofacets.monostatic_sweep(stl_path, freq_ghz, az_start, az_end,
    /// step, elevation, polarization)` which is expected to return a sequence
    /// of `(azimuth_deg, rcs_dbsm)` pairs.
    pub fn sweep_azimuth(
        &self,
        freq_ghz: f64,
        az_range_deg: (f64, f64),
        step_deg: f64,
        elevation_deg: f64,
        polarization: &str,
    ) -> PyResult<RcsSweepResult> {
        Python::with_gil(|py| {
            let pofacets = py.import("pofacets")?;
            let raw = pofacets.call_method1(
                "monostatic_sweep",
                (
                    self.stl_path.clone(),
                    freq_ghz,
                    az_range_deg.0,
                    az_range_deg.1,
                    step_deg,
                    elevation_deg,
                    polarization.to_string(),
                ),
            )?;
            let pairs: Vec<(f64, f64)> = raw.extract()?;
            let samples = pairs
                .into_iter()
                .map(|(az, rcs_dbsm)| RcsSample {
                    azimuth_deg: az,
                    elevation_deg,
                    rcs_dbsm,
                })
                .collect();
            Ok(RcsSweepResult {
                frequency_ghz: freq_ghz,
                polarization: polarization.to_string(),
                samples,
            })
        })
    }

    /// Full hemisphere sweep across azimuth × elevation as defined by
    /// `config`. If `config.el_angles_deg` is empty a single cut at
    /// elevation 0° is produced.
    pub fn sweep_hemisphere(&self, config: &RcsSweepConfig) -> PyResult<RcsSweepResult> {
        let elevations: Vec<f64> = if config.el_angles_deg.is_empty() {
            vec![0.0]
        } else {
            config.el_angles_deg.clone()
        };

        let mut samples = Vec::new();
        for el in elevations {
            let cut = self.sweep_azimuth(
                config.frequency_ghz,
                (config.az_start_deg, config.az_end_deg),
                config.az_step_deg,
                el,
                &config.polarization,
            )?;
            samples.extend(cut.samples);
        }

        Ok(RcsSweepResult {
            frequency_ghz: config.frequency_ghz,
            polarization: config.polarization.clone(),
            samples,
        })
    }
}

/// Convenience entry point wrapping geometry validation + sweep + JSON export.
///
/// Calls `load_geometry()` first to fail fast on an invalid or missing STL
/// path before running the full hemisphere sweep.
#[cfg(feature = "rcs-compute")]
pub fn compute_and_save_rcs(
    stl_path: &str,
    config: &RcsSweepConfig,
    output_path: &std::path::Path,
) -> PyResult<()> {
    let bridge = RcsComputeBridge::new(stl_path);
    let _ = bridge.load_geometry()?;
    let result = bridge.sweep_hemisphere(config)?;
    result.write_json(output_path).map_err(|e| {
        pyo3::exceptions::PyIOError::new_err(format!("failed to write {output_path:?}: {e}"))
    })?;
    Ok(())
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config_reasonable() {
        let c = RcsSweepConfig::default();
        assert!(c.frequency_ghz > 0.0);
        assert!(c.az_end_deg > c.az_start_deg);
        assert!(c.az_step_deg > 0.0);
        assert_eq!(c.polarization, "VV");
    }

    #[test]
    fn sweep_result_json_roundtrip() {
        let result = RcsSweepResult {
            frequency_ghz: 10.0,
            polarization: "VV".into(),
            samples: vec![
                RcsSample {
                    azimuth_deg: 0.0,
                    elevation_deg: 0.0,
                    rcs_dbsm: -5.0,
                },
                RcsSample {
                    azimuth_deg: 45.0,
                    elevation_deg: 0.0,
                    rcs_dbsm: 3.0,
                },
            ],
        };
        let json = serde_json::to_string(&result).unwrap();
        let parsed: RcsSweepResult = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.samples.len(), 2);
        assert_eq!(parsed.frequency_ghz, 10.0);
    }

    #[test]
    fn sweep_result_pretty_json_and_file_write() {
        let result = RcsSweepResult {
            frequency_ghz: 9.5,
            polarization: "HH".into(),
            samples: vec![RcsSample {
                azimuth_deg: 10.0,
                elevation_deg: 0.0,
                rcs_dbsm: 1.25,
            }],
        };
        let json = result.to_json().unwrap();
        assert!(json.contains("\"polarization\""));

        let dir = std::env::temp_dir();
        let unique = format!(
            "thresh_rcs_compute_write_test_{}_{}.json",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        );
        let path = dir.join(unique);
        result.write_json(&path).unwrap();
        let round: RcsSweepResult = serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
        assert_eq!(round.samples.len(), 1);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    #[should_panic(expected = "requires at least one sample")]
    fn sweep_to_lookup_table_panics_on_empty_samples() {
        let empty = RcsSweepResult {
            frequency_ghz: 10.0,
            polarization: "VV".into(),
            samples: vec![],
        };
        let _ = sweep_to_lookup_table(&empty);
    }

    #[test]
    fn sweep_to_lookup_table_single_elevation() {
        let result = RcsSweepResult {
            frequency_ghz: 10.0,
            polarization: "VV".into(),
            samples: vec![
                RcsSample {
                    azimuth_deg: 0.0,
                    elevation_deg: 0.0,
                    rcs_dbsm: 1.0,
                },
                RcsSample {
                    azimuth_deg: 90.0,
                    elevation_deg: 0.0,
                    rcs_dbsm: 2.0,
                },
                RcsSample {
                    azimuth_deg: 180.0,
                    elevation_deg: 0.0,
                    rcs_dbsm: 3.0,
                },
            ],
        };
        let table = sweep_to_lookup_table(&result);
        assert_eq!(table.azimuth_deg.len(), 3);
        assert_eq!(table.elevation_deg.len(), 1);
        assert_eq!(table.rcs_dbsm[0][0], 1.0);
        assert_eq!(table.rcs_dbsm[1][0], 2.0);
        assert_eq!(table.rcs_dbsm[2][0], 3.0);
    }

    #[test]
    fn sweep_to_lookup_table_hemisphere_grid() {
        let result = RcsSweepResult {
            frequency_ghz: 10.0,
            polarization: "VV".into(),
            samples: vec![
                RcsSample {
                    azimuth_deg: 0.0,
                    elevation_deg: 0.0,
                    rcs_dbsm: 0.0,
                },
                RcsSample {
                    azimuth_deg: 0.0,
                    elevation_deg: 30.0,
                    rcs_dbsm: 1.0,
                },
                RcsSample {
                    azimuth_deg: 90.0,
                    elevation_deg: 0.0,
                    rcs_dbsm: 2.0,
                },
                RcsSample {
                    azimuth_deg: 90.0,
                    elevation_deg: 30.0,
                    rcs_dbsm: 3.0,
                },
            ],
        };
        let table = sweep_to_lookup_table(&result);
        assert_eq!(table.azimuth_deg, vec![0.0, 90.0]);
        assert_eq!(table.elevation_deg, vec![0.0, 30.0]);
        assert_eq!(table.rcs_dbsm[0][0], 0.0);
        assert_eq!(table.rcs_dbsm[0][1], 1.0);
        assert_eq!(table.rcs_dbsm[1][0], 2.0);
        assert_eq!(table.rcs_dbsm[1][1], 3.0);

        // Round-trip through bilinear lookup at an interior point.
        let mid = table.lookup(45.0, 15.0);
        assert!(mid.is_finite());
    }

    /// Sphere test (analytical): RCS of a sphere radius r is pi*r^2 in the
    /// optical/PO regime. This test runs against a real sphere STL but is
    /// gated since it needs PyPOFacets.
    #[cfg(feature = "rcs-compute")]
    #[test]
    #[ignore]
    fn sphere_rcs_matches_analytical() {
        // Requires a sphere.stl file and the `pofacets` Python package.
        // pi * r^2 for r = 1 m is ~3.14 m^2 = 4.97 dBsm.
        let bridge = RcsComputeBridge::new("/tmp/sphere.stl");
        let config = RcsSweepConfig::default();
        let result = bridge.sweep_hemisphere(&config).expect("sweep");
        let mean_dbsm: f64 =
            result.samples.iter().map(|s| s.rcs_dbsm).sum::<f64>() / result.samples.len() as f64;
        let expected_dbsm = 10.0 * (std::f64::consts::PI).log10();
        assert!((mean_dbsm - expected_dbsm).abs() < 2.0);
    }
}
