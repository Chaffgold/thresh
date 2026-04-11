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
/// [`RcsLookupTable`]: crate::swerling::RcsLookupTable
pub fn sweep_to_lookup_table(result: &RcsSweepResult) -> crate::swerling::RcsLookupTable {
    // Collect unique azimuth / elevation values using an integer key for
    // stable de-duplication of floating-point grid points.
    fn key(x: f64) -> i64 {
        (x * 1_000.0).round() as i64
    }

    let mut az_keys: Vec<i64> = Vec::new();
    let mut el_keys: Vec<i64> = Vec::new();
    for s in &result.samples {
        let ak = key(s.azimuth_deg);
        let ek = key(s.elevation_deg);
        if !az_keys.contains(&ak) {
            az_keys.push(ak);
        }
        if !el_keys.contains(&ek) {
            el_keys.push(ek);
        }
    }
    az_keys.sort_unstable();
    el_keys.sort_unstable();

    let azimuth_deg: Vec<f64> = az_keys.iter().map(|k| (*k as f64) / 1_000.0).collect();
    let elevation_deg: Vec<f64> = el_keys.iter().map(|k| (*k as f64) / 1_000.0).collect();

    let fill = result
        .samples
        .iter()
        .map(|s| s.rcs_dbsm)
        .fold(f64::INFINITY, f64::min);
    let fill = if fill.is_finite() { fill } else { 0.0 };

    let mut rcs_dbsm: Vec<Vec<f64>> = vec![vec![fill; elevation_deg.len()]; azimuth_deg.len()];
    for s in &result.samples {
        let ai = az_keys
            .iter()
            .position(|k| *k == key(s.azimuth_deg))
            .unwrap();
        let ei = el_keys
            .iter()
            .position(|k| *k == key(s.elevation_deg))
            .unwrap();
        rcs_dbsm[ai][ei] = s.rcs_dbsm;
    }

    crate::swerling::RcsLookupTable {
        azimuth_deg,
        elevation_deg,
        rcs_dbsm,
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

/// Convenience entry point wrapping load + sweep + JSON export.
#[cfg(feature = "rcs-compute")]
pub fn compute_and_save_rcs(
    stl_path: &str,
    config: &RcsSweepConfig,
    output_path: &std::path::Path,
) -> PyResult<()> {
    let bridge = RcsComputeBridge::new(stl_path);
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
        let path = dir.join("thresh_rcs_compute_write_test.json");
        result.write_json(&path).unwrap();
        let round: RcsSweepResult = serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
        assert_eq!(round.samples.len(), 1);
        let _ = std::fs::remove_file(&path);
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
