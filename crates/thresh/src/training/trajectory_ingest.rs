//! Canonical acquisition-trajectory ingestion for the real-data training runs.
//!
//! Reads the acquisition layer's canonical trajectory Parquet (the schema in
//! `python/acquisition/schema.py`, written by `acquisition/opensky_cli.py` and
//! `storage.write_partition`) and converts each aircraft's observations into
//! synth-ready [`Waypoint`] tracks:
//!
//! 1. **read** — decode the columns we need (`icao24`, `timestamp_us`, `lat`,
//!    `lon`, altitudes, ground speed / track / vertical rate, `category`) and
//!    group rows by `icao24`, sorted by time.
//! 2. **split** — break a track at observation gaps larger than
//!    [`IngestConfig::gap_split_s`] (mirrors the Python stitching default) and
//!    drop segments that are too short to train on.
//! 3. **frame** — convert WGS84 to a per-segment ENU frame whose origin is the
//!    segment's mean position at ground level, so every target stays well
//!    inside the synth radar's `max_range_m` regardless of where on Earth it
//!    was recorded. Velocity comes from the reported ground speed + true track
//!    when present and finite differences otherwise.
//! 4. **resample** — interpolate the sparse (~10 s anonymous OpenSky cadence)
//!    waypoints onto the synth sample grid via [`Waypoint::interpolate`], as
//!    [`generate_imm_training_samples`](super::generate_imm_training_samples)
//!    requires waypoints pre-sampled at the radar config's `sample_rate_hz`.
//!
//! Gated behind the `training-export` feature with the other `arrow`/`parquet`
//! consumers so default and CI builds stay lean.

use std::collections::BTreeMap;
use std::fs::File;
use std::path::Path;

use arrow::array::{Array, Float64Array, Int64Array, StringArray};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

use thresh_core::geodetic::wgs84_to_enu;
use thresh_synth::trajectory::Waypoint;

/// One decoded observation row (pre-ENU).
#[derive(Debug, Clone)]
struct Observation {
    time_s: f64,
    lat_deg: f64,
    lon_deg: f64,
    alt_m: f64,
    vel_ground_mps: Option<f64>,
    track_deg: Option<f64>,
    vrate_mps: Option<f64>,
}

/// One ingested, synth-ready track segment.
#[derive(Debug, Clone)]
pub struct IngestedTrack {
    /// Source aircraft (lowercase hex Mode-S address).
    pub icao24: String,
    /// Segment index within this aircraft's observations (gap splits).
    pub segment: u32,
    /// thresh class index in `[0, 5)` derived from the ADS-B emitter category.
    pub class_id: u32,
    /// ENU waypoints resampled to the synth grid, time rebased to 0.
    pub waypoints: Vec<Waypoint>,
}

/// Ingestion thresholds.
#[derive(Debug, Clone)]
pub struct IngestConfig {
    /// Split a track at observation gaps larger than this (seconds). Matches
    /// the Python acquisition layer's stitching default.
    pub gap_split_s: f64,
    /// Drop segments with fewer raw observations than this.
    pub min_points: usize,
    /// Drop segments spanning less wall-clock time than this (seconds).
    pub min_duration_s: f64,
    /// Grid rate the waypoints are resampled to. MUST equal the
    /// `TrajectoryRadarConfig::sample_rate_hz` used downstream.
    pub sample_rate_hz: f64,
}

impl Default for IngestConfig {
    fn default() -> Self {
        Self {
            gap_split_s: 60.0,
            min_points: 10,
            min_duration_s: 60.0,
            sample_rate_hz: 1.0,
        }
    }
}

/// Shared CLI surface for the two dataset-generation binaries:
/// `gen-* [OUT] [--trajectories CANONICAL.parquet] [--sample-rate-hz F]`.
#[derive(Debug, Clone)]
pub struct GenArgs {
    /// Output Parquet path.
    pub out: std::path::PathBuf,
    /// Canonical acquisition trajectories (real-data mode when present).
    pub trajectories: Option<std::path::PathBuf>,
    /// Synth grid rate for the real-data mode.
    pub sample_rate_hz: f64,
}

impl GenArgs {
    /// Parse `std::env::args()` with per-binary defaults.
    pub fn parse(default_out: &str, default_sample_rate_hz: f64) -> Self {
        let mut parsed = Self {
            out: std::path::PathBuf::from(default_out),
            trajectories: None,
            sample_rate_hz: default_sample_rate_hz,
        };
        let mut args = std::env::args().skip(1);
        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--trajectories" => {
                    parsed.trajectories = args.next().map(std::path::PathBuf::from);
                }
                "--sample-rate-hz" => {
                    parsed.sample_rate_hz = args
                        .next()
                        .and_then(|v| v.parse().ok())
                        .expect("--sample-rate-hz needs a positive number");
                }
                other => parsed.out = std::path::PathBuf::from(other),
            }
        }
        parsed
    }
}

/// Run `generate` once per ingested track (trajectory_id = index), skipping
/// degenerate tracks with a warning instead of aborting the whole dataset —
/// one unluckily-shaped segment must not kill an hours-long generation run.
pub fn generate_per_track<T, E: std::fmt::Display>(
    tracks: &[IngestedTrack],
    mut generate: impl FnMut(u32, &IngestedTrack) -> Result<Vec<T>, E>,
) -> Vec<T> {
    let mut samples = Vec::new();
    let mut skipped = 0usize;
    for (index, track) in tracks.iter().enumerate() {
        match generate(index as u32, track) {
            Ok(track_samples) => samples.extend(track_samples),
            Err(err) => {
                skipped += 1;
                eprintln!("skipping {}#{}: {err}", track.icao24, track.segment);
            }
        }
    }
    if skipped > 0 {
        eprintln!("skipped {skipped}/{} track segments", tracks.len());
    }
    samples
}

/// Map an ADS-B emitter category to the thresh class index.
///
/// Mirrors `python/acquisition/schema.py::_CATEGORY_MAP` +
/// `python/training/classes.py` (index-aligned to `CLASS_BOX_DIMS`):
/// light fixed-wing 0, heavy fixed-wing 1, rotorcraft 2,
/// glider/balloon/UAV 3, other 4.
pub fn class_id_for_adsb_category(category: Option<&str>) -> u32 {
    match category.map(str::to_ascii_uppercase).as_deref() {
        Some("A1" | "A2") => 0,
        Some("A3" | "A4" | "A5") => 1,
        Some("A7") => 2,
        Some("B1" | "B2" | "B6") => 3,
        _ => 4,
    }
}

/// Read a canonical trajectory Parquet and return synth-ready track segments.
pub fn ingest_trajectories(
    path: &Path,
    config: &IngestConfig,
) -> Result<Vec<IngestedTrack>, Box<dyn std::error::Error>> {
    let grouped = read_grouped_observations(path)?;
    let mut tracks = Vec::new();
    for (icao24, (category, mut observations)) in grouped {
        observations.sort_by(|a, b| a.time_s.total_cmp(&b.time_s));
        observations.dedup_by(|a, b| (a.time_s - b.time_s).abs() < 1e-9);
        let class_id = class_id_for_adsb_category(category.as_deref());
        for (segment, chunk) in split_on_gaps(&observations, config.gap_split_s)
            .into_iter()
            .enumerate()
        {
            if let Some(track) = build_track(&icao24, segment as u32, class_id, chunk, config) {
                tracks.push(track);
            }
        }
    }
    Ok(tracks)
}

/// Phase 1: decode the Parquet into per-aircraft observation lists.
///
/// Rows without a usable altitude (geometric preferred, barometric fallback)
/// are skipped — a target with unknown altitude cannot be placed in the ENU
/// scene. The first non-null `category` per aircraft wins.
#[allow(clippy::type_complexity)]
fn read_grouped_observations(
    path: &Path,
) -> Result<BTreeMap<String, (Option<String>, Vec<Observation>)>, Box<dyn std::error::Error>> {
    let file = File::open(path)?;
    let reader = ParquetRecordBatchReaderBuilder::try_new(file)?.build()?;

    let mut grouped: BTreeMap<String, (Option<String>, Vec<Observation>)> = BTreeMap::new();
    for batch in reader {
        let batch = batch?;
        let icao24 = column_as::<StringArray>(&batch, "icao24")?;
        let timestamp_us = column_as::<Int64Array>(&batch, "timestamp_us")?;
        let lat = column_as::<Float64Array>(&batch, "lat")?;
        let lon = column_as::<Float64Array>(&batch, "lon")?;
        let alt_geom = column_as::<Float64Array>(&batch, "alt_geom_m")?;
        let alt_baro = column_as::<Float64Array>(&batch, "alt_baro_m")?;
        let vel = column_as::<Float64Array>(&batch, "vel_ground_mps")?;
        let track = column_as::<Float64Array>(&batch, "track_deg")?;
        let vrate = column_as::<Float64Array>(&batch, "vrate_mps")?;
        let category = column_as::<StringArray>(&batch, "category")?;

        for row in 0..batch.num_rows() {
            let alt_m = optional(alt_geom, row).or_else(|| optional(alt_baro, row));
            let Some(alt_m) = alt_m else { continue };
            let entry = grouped.entry(icao24.value(row).to_string()).or_default();
            if entry.0.is_none() && category.is_valid(row) {
                entry.0 = Some(category.value(row).to_string());
            }
            entry.1.push(Observation {
                time_s: timestamp_us.value(row) as f64 / 1e6,
                lat_deg: lat.value(row),
                lon_deg: lon.value(row),
                alt_m,
                vel_ground_mps: optional(vel, row),
                track_deg: optional(track, row),
                vrate_mps: optional(vrate, row),
            });
        }
    }
    Ok(grouped)
}

fn column_as<'a, T: 'static>(
    batch: &'a arrow::record_batch::RecordBatch,
    name: &str,
) -> Result<&'a T, Box<dyn std::error::Error>> {
    batch
        .column_by_name(name)
        .ok_or_else(|| format!("canonical trajectory parquet is missing column '{name}'"))?
        .as_any()
        .downcast_ref::<T>()
        .ok_or_else(|| format!("column '{name}' has an unexpected Arrow type").into())
}

fn optional(array: &Float64Array, row: usize) -> Option<f64> {
    array.is_valid(row).then(|| array.value(row))
}

/// Phase 2: split at observation gaps larger than `gap_s`.
fn split_on_gaps(observations: &[Observation], gap_s: f64) -> Vec<Vec<Observation>> {
    let mut segments = Vec::new();
    let mut current: Vec<Observation> = Vec::new();
    for obs in observations {
        if let Some(last) = current.last()
            && obs.time_s - last.time_s > gap_s
        {
            segments.push(std::mem::take(&mut current));
        }
        current.push(obs.clone());
    }
    if !current.is_empty() {
        segments.push(current);
    }
    segments
}

/// Phases 3+4: ENU-frame a segment and resample it onto the synth grid.
///
/// Returns `None` when the segment fails the length/duration thresholds.
fn build_track(
    icao24: &str,
    segment: u32,
    class_id: u32,
    observations: Vec<Observation>,
    config: &IngestConfig,
) -> Option<IngestedTrack> {
    let duration = match (observations.first(), observations.last()) {
        (Some(first), Some(last)) => last.time_s - first.time_s,
        _ => return None,
    };
    if observations.len() < config.min_points || duration < config.min_duration_s {
        return None;
    }
    let sparse = to_enu_waypoints(&observations);
    let waypoints = resample_to_grid(&sparse, config.sample_rate_hz);
    (waypoints.len() >= 2).then(|| IngestedTrack {
        icao24: icao24.to_string(),
        segment,
        class_id,
        waypoints,
    })
}

/// Phase 3: WGS84 → per-segment ENU waypoints, time rebased to zero.
///
/// The ENU origin is the segment's mean lat/lon at ground level, so ranges to
/// a sensor at the origin stay bounded by the segment's own extent. Velocity
/// prefers the reported ground speed + true track (E = v·sin, N = v·cos, U =
/// vertical rate) and falls back to position finite differences.
fn to_enu_waypoints(observations: &[Observation]) -> Vec<Waypoint> {
    let n = observations.len() as f64;
    let ref_lat = observations.iter().map(|o| o.lat_deg).sum::<f64>() / n;
    let ref_lon = observations.iter().map(|o| o.lon_deg).sum::<f64>() / n;
    let t0 = observations[0].time_s;

    let positions: Vec<[f64; 3]> = observations
        .iter()
        .map(|o| {
            let enu = wgs84_to_enu(
                o.lat_deg.to_radians(),
                o.lon_deg.to_radians(),
                o.alt_m,
                ref_lat.to_radians(),
                ref_lon.to_radians(),
                0.0,
            );
            [enu.x, enu.y, enu.z]
        })
        .collect();

    observations
        .iter()
        .enumerate()
        .map(|(i, o)| Waypoint {
            time: o.time_s - t0,
            position: positions[i],
            velocity: observation_velocity(o, i, &positions, observations),
        })
        .collect()
}

/// Reported ground-speed/track velocity, else finite difference to the
/// neighbouring observation.
fn observation_velocity(
    obs: &Observation,
    index: usize,
    positions: &[[f64; 3]],
    observations: &[Observation],
) -> [f64; 3] {
    if let (Some(speed), Some(track)) = (obs.vel_ground_mps, obs.track_deg) {
        let track_rad = track.to_radians();
        return [
            speed * track_rad.sin(),
            speed * track_rad.cos(),
            obs.vrate_mps.unwrap_or(0.0),
        ];
    }
    let (i, j) = if index + 1 < positions.len() {
        (index, index + 1)
    } else {
        (index - 1, index)
    };
    let dt = observations[j].time_s - observations[i].time_s;
    if dt.abs() < 1e-9 {
        return [0.0, 0.0, 0.0];
    }
    [
        (positions[j][0] - positions[i][0]) / dt,
        (positions[j][1] - positions[i][1]) / dt,
        (positions[j][2] - positions[i][2]) / dt,
    ]
}

/// Phase 4: interpolate sparse waypoints onto the `sample_rate_hz` grid.
fn resample_to_grid(sparse: &[Waypoint], sample_rate_hz: f64) -> Vec<Waypoint> {
    if sparse.len() < 2 || sample_rate_hz <= 0.0 {
        return Vec::new();
    }
    let dt = 1.0 / sample_rate_hz;
    let t_end = sparse[sparse.len() - 1].time;
    let steps = (t_end / dt).floor() as usize;

    let mut grid = Vec::with_capacity(steps + 1);
    let mut segment = 0usize;
    for step in 0..=steps {
        let t = step as f64 * dt;
        while segment + 2 < sparse.len() && sparse[segment + 1].time < t {
            segment += 1;
        }
        grid.push(Waypoint::interpolate(
            &sparse[segment],
            &sparse[segment + 1],
            t,
        ));
    }
    grid
}

#[cfg(test)]
mod tests {
    use super::*;

    fn obs(time_s: f64, lat: f64, lon: f64, alt: f64) -> Observation {
        Observation {
            time_s,
            lat_deg: lat,
            lon_deg: lon,
            alt_m: alt,
            vel_ground_mps: None,
            track_deg: None,
            vrate_mps: None,
        }
    }

    #[test]
    fn category_mapping_mirrors_python_taxonomy() {
        assert_eq!(class_id_for_adsb_category(Some("A1")), 0);
        assert_eq!(class_id_for_adsb_category(Some("a4")), 1);
        assert_eq!(class_id_for_adsb_category(Some("A7")), 2);
        assert_eq!(class_id_for_adsb_category(Some("B6")), 3);
        assert_eq!(class_id_for_adsb_category(Some("C1")), 4);
        assert_eq!(class_id_for_adsb_category(None), 4);
    }

    #[test]
    fn gap_split_breaks_at_threshold() {
        let observations = vec![
            obs(0.0, 50.0, 8.0, 1000.0),
            obs(10.0, 50.0, 8.0, 1000.0),
            obs(120.0, 50.0, 8.0, 1000.0),
        ];
        let segments = split_on_gaps(&observations, 60.0);
        assert_eq!(segments.len(), 2);
        assert_eq!(segments[0].len(), 2);
        assert_eq!(segments[1].len(), 1);
    }

    #[test]
    fn enu_frame_centres_the_segment() {
        // Two points straddling the reference latitude: ENU north components
        // should be symmetric around zero and the east components ~0.
        let observations = vec![obs(0.0, 49.99, 8.0, 1000.0), obs(10.0, 50.01, 8.0, 1000.0)];
        let wps = to_enu_waypoints(&observations);
        assert!((wps[0].position[0]).abs() < 1.0, "east ~0");
        assert!(
            (wps[0].position[1] + wps[1].position[1]).abs() < 1.0,
            "north symmetric: {} vs {}",
            wps[0].position[1],
            wps[1].position[1]
        );
        // ~0.02° of latitude ≈ 2.2 km separation.
        let span = wps[1].position[1] - wps[0].position[1];
        assert!((span - 2_224.0).abs() < 30.0, "span {span}");
        // Finite-difference velocity ≈ span / 10 s, northbound.
        assert!((wps[0].velocity[1] - span / 10.0).abs() < 1.0);
    }

    #[test]
    fn reported_velocity_beats_finite_difference() {
        let mut o = obs(0.0, 50.0, 8.0, 1000.0);
        o.vel_ground_mps = Some(100.0);
        o.track_deg = Some(90.0); // due east
        o.vrate_mps = Some(-2.0);
        let v = observation_velocity(
            &o,
            0,
            &[[0.0; 3], [0.0; 3]],
            &[o.clone(), obs(10.0, 50.0, 8.0, 1000.0)],
        );
        assert!((v[0] - 100.0).abs() < 1e-9);
        assert!(v[1].abs() < 1e-9);
        assert!((v[2] + 2.0).abs() < 1e-9);
    }

    #[test]
    fn resample_hits_the_grid_and_clamps() {
        let sparse = vec![
            Waypoint {
                time: 0.0,
                position: [0.0; 3],
                velocity: [1.0, 0.0, 0.0],
            },
            Waypoint {
                time: 10.0,
                position: [100.0, 0.0, 0.0],
                velocity: [1.0, 0.0, 0.0],
            },
            Waypoint {
                time: 20.0,
                position: [300.0, 0.0, 0.0],
                velocity: [1.0, 0.0, 0.0],
            },
        ];
        let grid = resample_to_grid(&sparse, 1.0);
        assert_eq!(grid.len(), 21);
        assert!(
            (grid[5].position[0] - 50.0).abs() < 1e-9,
            "mid-segment lerp"
        );
        assert!(
            (grid[15].position[0] - 200.0).abs() < 1e-9,
            "second segment"
        );
        assert!((grid[20].position[0] - 300.0).abs() < 1e-9, "endpoint");
        let dt = grid[1].time - grid[0].time;
        assert!((dt - 1.0).abs() < 1e-9);
    }

    #[test]
    fn build_track_enforces_thresholds() {
        let short: Vec<Observation> = (0..5)
            .map(|i| obs(i as f64 * 10.0, 50.0, 8.0, 1000.0))
            .collect();
        let config = IngestConfig::default();
        assert!(
            build_track("abc123", 0, 4, short, &config).is_none(),
            "too few points"
        );

        let long: Vec<Observation> = (0..12)
            .map(|i| obs(i as f64 * 10.0, 50.0 + i as f64 * 0.001, 8.0, 1000.0))
            .collect();
        let track = build_track("abc123", 0, 4, long, &config).expect("valid segment");
        assert_eq!(track.waypoints.len(), 111, "110 s at 1 Hz + endpoint");
        assert!((track.waypoints[0].time).abs() < 1e-9, "time rebased to 0");
    }
}
