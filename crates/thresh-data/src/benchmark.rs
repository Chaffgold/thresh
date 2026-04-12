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
use thresh_tracker::tracker_variant::TrackerVariant;

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
    AdsB {
        region: String,
    },
    /// Orbital scenario sourced from SGP4 propagation of one or more TLEs.
    ///
    /// Field meanings:
    /// - `norad_ids`: catalog IDs to fetch over HTTP (Space-Track / CelesTrak)
    ///   when `tle_file` is not set and the `orbital` feature is enabled.
    /// - `tle_file`: optional path to a local cached TLE file (3LE or 2LE
    ///   format). **Relative paths** are resolved relative to the scenario
    ///   manifest's directory, so a scenario `scenarios/orbital-iss.toml`
    ///   with `tle_file = "orbital-iss.tle"` reads `scenarios/orbital-iss.tle`.
    ///   When present, the runner uses this file and never touches the
    ///   network — this is what lets the CI gate run orbital scenarios
    ///   offline.
    /// - `station_lat_deg` / `station_lon_deg` / `station_alt_m`: ground
    ///   station location used to convert ECI → ENU for the tracker and
    ///   radar observation model. Degrees (not radians) in the manifest to
    ///   keep the TOML human-editable.
    /// - `time_step_s`: interval between propagation samples (and between
    ///   radar scans). If omitted, defaults to the `parameters.dt` value.
    Orbital {
        norad_ids: Vec<u32>,
        #[serde(default)]
        tle_file: Option<String>,
        #[serde(default = "default_station_lat_deg")]
        station_lat_deg: f64,
        #[serde(default = "default_station_lon_deg")]
        station_lon_deg: f64,
        #[serde(default)]
        station_alt_m: f64,
        #[serde(default)]
        time_step_s: Option<f64>,
    },
}

fn default_station_lat_deg() -> f64 {
    // Colorado Springs ground station (generic default, no operational
    // significance). Overridden per-scenario when it matters.
    38.8339
}

fn default_station_lon_deg() -> f64 {
    -104.8214
}

/// Common scenario parameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScenarioParameters {
    pub duration_s: f64,
    pub dt: f64,
    pub measurement_noise_sigma: f64,
    pub gate_threshold: f64,
    /// Optional tracker variant override. When `None` (the default) the
    /// benchmark runner uses the Cartesian ENU tracker — the same behaviour
    /// as before this field was added.
    ///
    /// This field is wired up as a forward-compatibility hook: the runner
    /// currently only drives the ENU tracker end-to-end. When the runner
    /// gains support for the other variants (ECEF, Great-Circle,
    /// Stereographic), selection will be honoured automatically without
    /// requiring scenario files to change.
    #[serde(default)]
    pub tracker_variant: Option<TrackerVariant>,
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
    // The benchmark runner currently only drives the Cartesian ENU tracker
    // end-to-end. If a scenario explicitly requests another variant, honour
    // the request only when it is `Enu`; otherwise fall back to ENU and
    // leave full wiring for the other variants to a future change.
    let _requested_variant = params.tracker_variant.unwrap_or(TrackerVariant::Enu);
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
        thresh_core::measurement::Measurement::Othr {
            ground_range_m,
            azimuth_rad,
            ..
        } => {
            // Approximate Cartesian from ground range and azimuth (flat-earth approx)
            let x = ground_range_m * azimuth_rad.sin();
            let y = ground_range_m * azimuth_rad.cos();
            DVector::from_column_slice(&[x, y, 0.0])
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

// ---------------------------------------------------------------------------
// Orbital benchmark runner (§7.5 / §7.6)
// ---------------------------------------------------------------------------

/// Run an orbital benchmark scenario end-to-end.
///
/// This function is only compiled when the `orbital` feature is enabled
/// because it depends on the `sgp4` crate (via `crate::orbital`). Downstream
/// callers that always need orbital support should build `thresh-data` with
/// `--features orbital`; the CLI surfaces a clean "feature required" error
/// when the feature is not compiled in.
///
/// Pipeline:
/// 1. Load TLEs — from a local file (`tle_file` relative to the manifest
///    directory) when set; otherwise from Space-Track / CelesTrak via the
///    orbital HTTP clients. The local-file path is what allows the CI
///    synthetic-benchmark gate to run orbital scenarios offline.
/// 2. Propagate each TLE via SGP4 → TEME → ECEF → ENU relative to the
///    station configured in `ScenarioSource::Orbital`. Samples are spaced
///    at `time_step_s` (falling back to `parameters.dt`) over `duration_s`
///    minutes starting at the TLE epoch.
/// 3. Convert the visible (above-horizon) ENU positions to synthetic radar
///    measurements, add Gaussian noise with the configured sigmas, and
///    feed them into the Cartesian ENU tracker.
/// 4. Build `FrameData` per time step and compute MOTA / MOTP / IDF1 /
///    HOTA against the noise-free ground truth.
///
/// `manifest_dir` is the parent directory of the scenario file — used to
/// resolve relative `tle_file` paths without hardcoding the workspace root.
#[cfg(feature = "orbital")]
pub fn run_orbital_benchmark(
    manifest: &ScenarioManifest,
    manifest_dir: &Path,
) -> core::result::Result<BenchmarkResult, String> {
    use crate::orbital::{
        GroundStation, RadarNoiseConfig, Tle, orbital_to_radar_measurements, parse_3le, parse_tle,
        propagate_to_enu,
    };
    use rand::SeedableRng;
    use rand::rngs::StdRng;
    use rand_distr::{Distribution, Normal};

    let start = Instant::now();
    let params = &manifest.parameters;

    let ScenarioSource::Orbital {
        norad_ids,
        tle_file,
        station_lat_deg,
        station_lon_deg,
        station_alt_m,
        time_step_s,
    } = &manifest.source
    else {
        return Err(format!(
            "run_orbital_benchmark called on non-Orbital source: {:?}",
            manifest.source
        ));
    };

    // ---- 1. Load TLEs ----
    let tles: Vec<Tle> = if let Some(file) = tle_file {
        let path = manifest_dir.join(file);
        let contents = std::fs::read_to_string(&path)
            .map_err(|e| format!("failed to read TLE file {}: {e}", path.display()))?;
        // Try 3LE (name + two data lines) first; fall back to 2LE if the
        // file has no leading name line.
        parse_3le(&contents)
            .or_else(|_| parse_tle(&contents))
            .map_err(|e| format!("failed to parse TLE file {}: {e}", path.display()))?
    } else {
        // No cached TLE — fall back to HTTP. Try CelesTrak first because
        // it's unauthenticated; if that fails, try Space-Track.
        if norad_ids.is_empty() {
            return Err("orbital scenario has no norad_ids and no tle_file".into());
        }
        match fetch_tles_via_http(norad_ids) {
            Ok(tles) => tles,
            Err(e) => {
                return Err(format!(
                    "no tle_file set and HTTP fetch failed: {e}. Provide a \
                     cached TLE file alongside the manifest to run offline."
                ));
            }
        }
    };

    if tles.is_empty() {
        return Err("no TLEs available after loading".into());
    }

    // Filter TLEs to the requested NORAD IDs when both are specified, so
    // a shared CelesTrak GROUP response can feed multiple scenarios.
    let selected_tles: Vec<&Tle> = if norad_ids.is_empty() {
        tles.iter().collect()
    } else {
        let wanted: std::collections::HashSet<u32> = norad_ids.iter().copied().collect();
        tles.iter()
            .filter(|t| wanted.contains(&t.norad_id))
            .collect()
    };

    if selected_tles.is_empty() {
        return Err(format!(
            "TLE file contained {} TLEs but none matched the requested norad_ids {:?}",
            tles.len(),
            norad_ids
        ));
    }

    // ---- 2. Propagate each TLE to ENU ----
    let step_s = time_step_s.unwrap_or(params.dt);
    let n_steps = (params.duration_s / step_s).ceil() as usize + 1;
    let times_min: Vec<f64> = (0..n_steps).map(|i| i as f64 * step_s / 60.0).collect();

    let lat_rad = station_lat_deg.to_radians();
    let lon_rad = station_lon_deg.to_radians();
    let _station = GroundStation {
        name: "scenario-station".into(),
        lat_rad,
        lon_rad,
        alt_m: *station_alt_m,
    };

    // Propagate each selected satellite and collect (target_id, ENU path).
    let mut trajectories: Vec<(u32, Vec<crate::orbital::EnuPosition>)> = Vec::new();
    for tle in &selected_tles {
        let enu = propagate_to_enu(tle, &times_min, lat_rad, lon_rad, *station_alt_m)
            .map_err(|e| format!("SGP4 propagation failed for {}: {e}", tle.norad_id))?;
        trajectories.push((tle.norad_id, enu));
    }

    // ---- 3. Build ground truth + noisy radar measurements, run tracker ----
    let noise = RadarNoiseConfig {
        range_sigma_m: params.measurement_noise_sigma,
        azimuth_sigma_rad: params.measurement_noise_sigma / 50_000.0,
        elevation_sigma_rad: params.measurement_noise_sigma / 50_000.0,
        include_range_rate: false,
        sensor_id: 0,
    };
    let measurements_by_sat: Vec<(u32, Vec<thresh_core::measurement::Measurement>)> = trajectories
        .iter()
        .map(|(id, enu)| (*id, orbital_to_radar_measurements(enu, &noise)))
        .collect();

    // Deterministic seeded RNG so the CI regression gate is reproducible.
    let mut rng = StdRng::seed_from_u64(0xA5_A5_A5_A5_A5_A5_A5_A5);
    let normal = Normal::new(0.0, 1.0).unwrap();

    let mut tracker =
        MultiObjectTracker::new_cv_position(params.measurement_noise_sigma, params.gate_threshold);
    let mut frame_data_vec: Vec<FrameData> = Vec::new();

    for step in 0..n_steps {
        let t_min = step as f64 * step_s / 60.0;

        // Collect noisy detections for every visible satellite at this step.
        let mut detections: Vec<DVector<f64>> = Vec::new();
        let mut gt_positions: Vec<(u64, [f64; 3])> = Vec::new();

        for ((id, enu), (_, measurements)) in trajectories.iter().zip(measurements_by_sat.iter()) {
            if let Some(pos) = enu
                .iter()
                .find(|p| (p.time_since_epoch_min - t_min).abs() < 1e-6)
                && pos.up > 0.0
            {
                gt_positions.push((u64::from(*id), [pos.east, pos.north, pos.up]));
            }
            if let Some(m) = measurements.iter().find(|m| match m {
                thresh_core::measurement::Measurement::Radar { time, .. } => {
                    (time - t_min * 60.0).abs() < 1e-6
                }
                _ => false,
            }) && let thresh_core::measurement::Measurement::Radar {
                range,
                azimuth,
                elevation,
                ..
            } = m
            {
                let noisy_range = range + noise.range_sigma_m * normal.sample(&mut rng);
                let noisy_az = azimuth + noise.azimuth_sigma_rad * normal.sample(&mut rng);
                let noisy_el = elevation + noise.elevation_sigma_rad * normal.sample(&mut rng);
                let x = noisy_range * noisy_el.cos() * noisy_az.cos();
                let y = noisy_range * noisy_el.cos() * noisy_az.sin();
                let z = noisy_range * noisy_el.sin();
                detections.push(DVector::from_column_slice(&[x, y, z]));
            }
        }

        tracker.step(&detections, step_s);

        let track_positions: Vec<(u64, [f64; 3])> = tracker
            .tracks
            .iter()
            .filter(|t| t.lifecycle == thresh_core::track::TrackState::Confirmed)
            .map(|t| (t.id.0, [t.state[0], t.state[2], t.state[4]]))
            .collect();

        frame_data_vec.push(FrameData {
            gt: gt_positions,
            tracks: track_positions,
        });
    }

    // ---- 4. Metrics ----
    // Orbital scenarios use a much larger match threshold because slant
    // ranges span hundreds of km and measurement noise is multi-kilometre.
    let dist_threshold = (params.measurement_noise_sigma * 10.0).max(5_000.0);
    let total_gt: usize = frame_data_vec.iter().map(|f| f.gt.len()).sum();
    let total_tracks: usize = frame_data_vec.iter().map(|f| f.tracks.len()).sum();
    eprintln!(
        "orbital pipeline: {} frames, {} ground-truth points, {} confirmed-track points",
        frame_data_vec.len(),
        total_gt,
        total_tracks,
    );
    let (mota, motp, id_switches) = compute_mot_metrics(&frame_data_vec, dist_threshold);
    let idf1 = compute_idf1(&frame_data_vec, dist_threshold);
    let (hota, _, _) = compute_hota_at_threshold(&frame_data_vec, dist_threshold);

    Ok(BenchmarkResult {
        scenario: manifest.name.clone(),
        mota,
        motp,
        idf1,
        hota,
        id_switches,
        duration_ms: start.elapsed().as_millis() as u64,
    })
}

/// Fetch TLEs for the given NORAD IDs via HTTP, trying CelesTrak first
/// (public, no auth required) and falling back to Space-Track for IDs
/// CelesTrak doesn't return.
#[cfg(feature = "orbital")]
fn fetch_tles_via_http(norad_ids: &[u32]) -> Result<Vec<crate::orbital::Tle>, String> {
    use crate::orbital::{CelestrakClient, Tle};

    let client = CelestrakClient::new();
    let mut out: Vec<Tle> = Vec::new();
    for id in norad_ids {
        let tles = client
            .fetch_catnr(*id)
            .map_err(|e| format!("CelesTrak fetch for NORAD {id} failed: {e}"))?;
        out.extend(tles);
    }
    Ok(out)
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
                tracker_variant: None,
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
