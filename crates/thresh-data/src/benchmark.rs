//! Benchmark scenario manifest, runner, and regression checking.

use std::collections::HashMap;
use std::path::Path;
use std::time::Instant;

use nalgebra::DVector;
use serde::{Deserialize, Serialize};

use thresh_eval::hota::compute_hota_at_threshold;
use thresh_eval::matching::FrameData;
use thresh_eval::metrics::{compute_idf1, compute_mot_metrics};
use thresh_synth::measurement_gen::RadarConfig;
use thresh_synth::scenario::{GroundTruth, run_scenario};
use thresh_synth::trajectory::{Segment, SegmentType, Trajectory};
use thresh_tracker::tracker::MultiObjectTracker;

// ---------------------------------------------------------------------------
// Manifest types
// ---------------------------------------------------------------------------

/// Top-level benchmark scenario description.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScenarioManifest {
    pub name: String,
    pub description: String,
    pub source: ScenarioSource,
    pub parameters: ScenarioParameters,
    pub baselines: Option<Baselines>,
}

/// Where the data comes from.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ScenarioSource {
    Synthetic,
    AdsB { region: String },
    Orbital { norad_ids: Vec<u32> },
}

/// Common scenario parameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScenarioParameters {
    pub duration_s: f64,
    pub dt: f64,
    pub measurement_noise_sigma: f64,
    pub gate_threshold: f64,
}

/// Expected metric baselines for regression gating.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Baselines {
    pub mota: Option<f64>,
    pub hota: Option<f64>,
    pub idf1: Option<f64>,
}

// ---------------------------------------------------------------------------
// Loading
// ---------------------------------------------------------------------------

/// Load a scenario manifest from a TOML file.
pub fn load_scenario(path: &Path) -> Result<ScenarioManifest, String> {
    let content = std::fs::read_to_string(path)
        .map_err(|e| format!("failed to read {}: {e}", path.display()))?;
    toml::from_str(&content).map_err(|e| format!("failed to parse {}: {e}", path.display()))
}

// ---------------------------------------------------------------------------
// Benchmark result
// ---------------------------------------------------------------------------

/// Results produced by running a benchmark scenario.
#[derive(Debug, Clone)]
pub struct BenchmarkResult {
    pub scenario: String,
    pub mota: f64,
    pub motp: f64,
    pub idf1: f64,
    pub hota: f64,
    pub id_switches: usize,
    pub duration_ms: u64,
}

// ---------------------------------------------------------------------------
// Runner
// ---------------------------------------------------------------------------

/// Run a synthetic benchmark scenario end-to-end.
///
/// 1. Generate trajectories from manifest parameters.
/// 2. Generate radar measurements with the given noise / detection settings.
/// 3. Run the tracker.
/// 4. Evaluate MOT metrics.
pub fn run_synthetic_benchmark(manifest: &ScenarioManifest) -> BenchmarkResult {
    let start = Instant::now();
    let params = &manifest.parameters;

    // --- Build trajectories (deterministic, spread out) ---
    let trajectories = build_trajectories(params);

    let scenario = thresh_synth::scenario::Scenario {
        name: manifest.name.clone(),
        trajectories,
    };

    // --- Radar config from manifest parameters ---
    let radar_config = RadarConfig {
        range_sigma: params.measurement_noise_sigma,
        azimuth_sigma: params.measurement_noise_sigma / 10_000.0,
        elevation_sigma: params.measurement_noise_sigma / 10_000.0,
        p_detection: 1.0, // default; low-pd scenarios override via name convention
        clutter_rate: 0.0,
        ..Default::default()
    };

    let (gt_entries, measurements) = run_scenario(&scenario, &radar_config);

    // --- Run tracker ---
    let mut tracker =
        MultiObjectTracker::new_cv_position(params.measurement_noise_sigma, params.gate_threshold);

    // Group measurements and ground truth by time step
    let mut meas_by_time: HashMap<i64, Vec<DVector<f64>>> = HashMap::new();
    for tm in &measurements {
        let key = (tm.time / params.dt).round() as i64;
        let pos = measurement_to_cartesian(&tm.measurement);
        meas_by_time.entry(key).or_default().push(pos);
    }

    let mut gt_by_time: HashMap<i64, Vec<GroundTruth>> = HashMap::new();
    for g in &gt_entries {
        let key = (g.time / params.dt).round() as i64;
        gt_by_time.entry(key).or_default().push(g.clone());
    }

    let max_step = meas_by_time
        .keys()
        .chain(gt_by_time.keys())
        .copied()
        .max()
        .unwrap_or(0);

    let mut frame_data_vec: Vec<FrameData> = Vec::new();

    for step in 0..=max_step {
        let dets: Vec<DVector<f64>> = meas_by_time.remove(&step).unwrap_or_default();

        tracker.step(&dets, params.dt);

        // Build FrameData for this step
        let gt_positions: Vec<(u64, [f64; 3])> = gt_by_time
            .get(&step)
            .map(|gs| {
                gs.iter()
                    .map(|g| (u64::from(g.target_id), g.position))
                    .collect()
            })
            .unwrap_or_default();

        let track_positions: Vec<(u64, [f64; 3])> = tracker
            .tracks
            .iter()
            .filter(|t| t.lifecycle == thresh_core::track::TrackState::Confirmed)
            .map(|t| {
                let pos = [t.state[0], t.state[2], t.state[4]];
                (t.id.0, pos)
            })
            .collect();

        frame_data_vec.push(FrameData {
            gt: gt_positions,
            tracks: track_positions,
        });
    }

    // --- Compute metrics ---
    let dist_threshold = params.measurement_noise_sigma * 5.0;
    let (mota, motp, id_switches) = compute_mot_metrics(&frame_data_vec, dist_threshold);
    let idf1 = compute_idf1(&frame_data_vec, dist_threshold);
    let (hota, _, _) = compute_hota_at_threshold(&frame_data_vec, dist_threshold);

    let duration_ms = start.elapsed().as_millis() as u64;

    BenchmarkResult {
        scenario: manifest.name.clone(),
        mota,
        motp,
        idf1,
        hota,
        id_switches,
        duration_ms,
    }
}

// ---------------------------------------------------------------------------
// Regression checking
// ---------------------------------------------------------------------------

/// Check a benchmark result against baselines.
/// Returns a list of failure messages (empty = pass).
pub fn check_regression(result: &BenchmarkResult, baselines: &Baselines) -> Vec<String> {
    let mut failures = Vec::new();
    if let Some(baseline_mota) = baselines.mota
        && result.mota < baseline_mota
    {
        failures.push(format!(
            "MOTA {:.2} below baseline {:.2}",
            result.mota, baseline_mota
        ));
    }
    if let Some(baseline_hota) = baselines.hota
        && result.hota < baseline_hota
    {
        failures.push(format!(
            "HOTA {:.2} below baseline {:.2}",
            result.hota, baseline_hota
        ));
    }
    if let Some(baseline_idf1) = baselines.idf1
        && result.idf1 < baseline_idf1
    {
        failures.push(format!(
            "IDF1 {:.2} below baseline {:.2}",
            result.idf1, baseline_idf1
        ));
    }
    failures
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Convert a measurement to Cartesian [x, y, z] for the tracker.
fn measurement_to_cartesian(m: &thresh_core::measurement::Measurement) -> DVector<f64> {
    match m {
        thresh_core::measurement::Measurement::Radar {
            range,
            azimuth,
            elevation,
            ..
        } => {
            let x = range * elevation.cos() * azimuth.cos();
            let y = range * elevation.cos() * azimuth.sin();
            let z = range * elevation.sin();
            DVector::from_column_slice(&[x, y, z])
        }
        thresh_core::measurement::Measurement::AdsB { lat, lon, alt, .. } => {
            DVector::from_column_slice(&[*lat, *lon, *alt])
        }
        thresh_core::measurement::Measurement::EoIr { .. } => {
            // Bearing-only: not directly usable for position-based tracker
            DVector::from_column_slice(&[0.0, 0.0, 0.0])
        }
    }
}

/// Build a set of CV trajectories spread in space for benchmark.
fn build_trajectories(params: &ScenarioParameters) -> Vec<Trajectory> {
    let n_targets = 5;
    (0..n_targets)
        .map(|i| {
            let spacing = 5000.0;
            Trajectory {
                target_id: i,
                initial_position: [
                    10_000.0 + i as f64 * spacing,
                    5_000.0 + i as f64 * spacing * 0.5,
                    3_000.0,
                ],
                initial_velocity: [200.0 + i as f64 * 20.0, 50.0 - i as f64 * 10.0, 0.0],
                segments: vec![Segment {
                    segment_type: SegmentType::Cv,
                    duration: params.duration_s,
                }],
                dt: params.dt,
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cv_clean_manifest() -> ScenarioManifest {
        ScenarioManifest {
            name: "synth-cv-clean".into(),
            description: "5 CV targets, low noise, perfect detection".into(),
            source: ScenarioSource::Synthetic,
            parameters: ScenarioParameters {
                duration_s: 30.0,
                dt: 1.0,
                measurement_noise_sigma: 50.0,
                gate_threshold: 500.0,
            },
            baselines: Some(Baselines {
                mota: Some(0.5),
                hota: None,
                idf1: None,
            }),
        }
    }

    #[test]
    fn load_and_parse_scenario_toml() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.toml");
        let manifest = cv_clean_manifest();
        let toml_str = toml::to_string_pretty(&manifest).unwrap();
        std::fs::write(&path, &toml_str).unwrap();

        let loaded = load_scenario(&path).unwrap();
        assert_eq!(loaded.name, "synth-cv-clean");
        assert_eq!(loaded.parameters.duration_s, 30.0);
    }

    #[test]
    fn run_synth_cv_clean_mota() {
        let manifest = cv_clean_manifest();
        let result = run_synthetic_benchmark(&manifest);
        // With confirmed-only tracks and well-separated CV targets, MOTA
        // should be solidly positive. The first few frames have FN while
        // tracks are still tentative, but after confirmation tracking is
        // reliable.
        assert!(
            result.mota > 0.5,
            "MOTA should be reasonable for clean CV scenario, got {}",
            result.mota
        );
    }

    #[test]
    fn regression_check_catches_failure() {
        let result = BenchmarkResult {
            scenario: "test".into(),
            mota: 0.72,
            motp: 5.0,
            idf1: 0.60,
            hota: 0.50,
            id_switches: 3,
            duration_ms: 100,
        };
        let baselines = Baselines {
            mota: Some(0.80),
            hota: Some(0.70),
            idf1: Some(0.75),
        };
        let failures = check_regression(&result, &baselines);
        assert_eq!(failures.len(), 3);
        assert!(failures[0].contains("MOTA"));
        assert!(failures[1].contains("HOTA"));
        assert!(failures[2].contains("IDF1"));
    }

    #[test]
    fn regression_check_passes_above_baseline() {
        let result = BenchmarkResult {
            scenario: "test".into(),
            mota: 0.95,
            motp: 3.0,
            idf1: 0.90,
            hota: 0.85,
            id_switches: 0,
            duration_ms: 50,
        };
        let baselines = Baselines {
            mota: Some(0.80),
            hota: Some(0.70),
            idf1: Some(0.75),
        };
        let failures = check_regression(&result, &baselines);
        assert!(
            failures.is_empty(),
            "Expected no failures, got: {failures:?}"
        );
    }
}
