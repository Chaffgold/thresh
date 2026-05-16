//! Trajectory-driven radar synthesis for the flight-data training pipeline.
//!
//! Per the design of OpenSpec change `flight-data-training-pipeline`:
//!
//! - **Decision 3 (Track A):** training pairs `(point_cloud, gt_boxes_3D)` are
//!   produced by running `thresh-synth` with a *real ADS-B trajectory* as the
//!   truth target list. This module is that producer.
//! - **Decision 4 (system-level ADS-B):** ADS-B records are ground truth, never
//!   sensor measurements. This module is the *only* place where measurement-level
//!   data — point clouds for Track A, radar plots for Track B — is generated.
//!
//! Two public APIs:
//!
//! - [`from_trajectory`] emits paired snapshots whose tensor shapes match the
//!   ONNX detector contract (`(1000, 4)` point cloud + `(100, 7)` boxes padded
//!   with a validity mask).
//! - [`measurements_from_trajectory`] emits per-tick lists of
//!   [`thresh_core::measurement::Measurement`] values — the input the classical
//!   tracker consumes during Phase 5's filter-state pipeline.
//!
//! ADS-B is typically reported at ~1 Hz; radar sample rates are 10 Hz+. The
//! synth interpolates between ADS-B waypoints linearly (see
//! [`crate::trajectory::Waypoint::interpolate`]) at the configured
//! `sample_rate_hz`. No physics-level interpolation (RCS oscillation,
//! aspect-angle changes) is attempted — that level of fidelity is out of scope.
//!
//! No PyO3 / Python binding is provided in this module yet; a separate
//! follow-up will mirror the `jsbsim.rs` / `radar_scene.rs` pattern. Until
//! then, downstream Python code converts canonical Parquet trajectories into
//! `Waypoint` slices via a thin helper.

use rand::Rng;
use rand_distr::{Distribution, Normal, Uniform};
use thresh_core::measurement::Measurement;

use crate::measurement_gen::{RadarConfig, detection_probability};
use crate::trajectory::Waypoint;

/// Fixed point-cloud tensor shape (matches the ONNX detector contract).
pub const POINT_CLOUD_SIZE: usize = 1000;
/// Channels per point: `[x, y, z, intensity]`.
pub const POINT_DIM: usize = 4;
/// Maximum ground-truth boxes per snapshot.
pub const MAX_GT_BOXES: usize = 100;
/// Box channels: `[x, y, z, length, width, height, yaw]`.
pub const BOX_DIM: usize = 7;

/// Sensor position in the local ENU frame (metres).
#[derive(Debug, Clone, Copy)]
pub struct SensorPose {
    pub position_enu_m: [f64; 3],
}

impl Default for SensorPose {
    fn default() -> Self {
        Self {
            position_enu_m: [0.0, 0.0, 0.0],
        }
    }
}

/// Per-class box dimensions `(length, width, height)` in metres.
#[derive(Debug, Clone, Copy)]
pub struct BoxDimensions {
    pub length: f32,
    pub width: f32,
    pub height: f32,
}

/// Default box dimensions per thresh detection class.
///
/// Indexing matches the canonical class enum:
/// 0=`light-fixed-wing`, 1=`heavy-fixed-wing`, 2=`rotorcraft`,
/// 3=`glider-or-balloon-or-uav`, 4=`other`.
pub const CLASS_BOX_DIMS: [BoxDimensions; 5] = [
    BoxDimensions {
        length: 10.0,
        width: 11.0,
        height: 3.0,
    }, // light-fixed-wing (Cessna-class)
    BoxDimensions {
        length: 40.0,
        width: 40.0,
        height: 12.0,
    }, // heavy-fixed-wing (737-class)
    BoxDimensions {
        length: 15.0,
        width: 14.0,
        height: 4.0,
    }, // rotorcraft (Bell 212-class)
    BoxDimensions {
        length: 18.0,
        width: 18.0,
        height: 1.5,
    }, // glider / balloon / UAV (varied)
    BoxDimensions {
        length: 10.0,
        width: 10.0,
        height: 5.0,
    }, // other (generic)
];

/// Default per-class RCS in m^2 (used by the detection-probability model).
pub const CLASS_RCS_M2: [f64; 5] = [
    1.0,   // light-fixed-wing
    100.0, // heavy-fixed-wing
    10.0,  // rotorcraft
    0.5,   // glider / balloon / UAV
    1.0,   // other
];

/// Configuration for trajectory-driven radar synthesis.
#[derive(Debug, Clone)]
pub struct TrajectoryRadarConfig {
    /// Sensor pose in ENU frame.
    pub sensor: SensorPose,
    /// Synth sample rate (Hz). Trajectory waypoints are interpolated to this.
    pub sample_rate_hz: f64,
    /// Maximum detection range (m). Targets beyond this are missed deterministically.
    pub max_range_m: f64,
    /// Per-detection point cloud "blob" size: number of returns emitted per
    /// detected target before clutter padding.
    pub returns_per_target: usize,
    /// Standard deviation (m) of additive Gaussian noise on each return's
    /// position. Independent across x, y, z.
    pub return_position_noise_m: f64,
    /// Maximum intensity (0..=1) for true-target returns.
    pub target_intensity_max: f32,
    /// Maximum intensity (0..=1) for clutter returns.
    pub clutter_intensity_max: f32,
    /// Per-axis half-extent (m) of the clutter cube centred on the sensor.
    /// Clutter is drawn uniformly within this volume.
    pub clutter_half_extent_m: f64,
    /// Underlying radar configuration (passed through to
    /// [`detection_probability`] from [`crate::measurement_gen`] for per-tick
    /// detection sampling).
    pub radar: RadarConfig,
    /// Per-class RCS (m^2). See [`CLASS_RCS_M2`] for sensible defaults.
    pub class_rcs_m2: [f64; 5],
}

impl Default for TrajectoryRadarConfig {
    fn default() -> Self {
        Self {
            sensor: SensorPose::default(),
            sample_rate_hz: 10.0,
            max_range_m: 200_000.0,
            returns_per_target: 16,
            return_position_noise_m: 5.0,
            target_intensity_max: 1.0,
            clutter_intensity_max: 0.2,
            clutter_half_extent_m: 100_000.0,
            radar: RadarConfig::default(),
            class_rcs_m2: CLASS_RCS_M2,
        }
    }
}

/// A target's full trajectory plus its class.
#[derive(Debug, Clone)]
pub struct TargetTrack {
    /// Sorted waypoints (any rate; commonly 1 Hz from ADS-B).
    pub waypoints: Vec<Waypoint>,
    /// Class index in `[0, 5)` — see [`CLASS_BOX_DIMS`].
    pub class_id: u32,
    /// Optional per-target box-dimension override. When `None`, looked up
    /// from [`CLASS_BOX_DIMS`].
    pub size_override: Option<BoxDimensions>,
}

/// One synthesis snapshot — a `(point_cloud, gt_boxes)` pair plus class index
/// and validity mask, all in shapes matching the ONNX detector contract.
///
/// All tensors are flattened row-major.
#[derive(Debug, Clone)]
pub struct RadarSnapshot {
    /// Snapshot time (seconds since trajectory epoch).
    pub time_s: f64,
    /// Flattened `[POINT_CLOUD_SIZE * POINT_DIM]` = 4000 floats.
    pub point_cloud: Vec<f32>,
    /// Flattened `[MAX_GT_BOXES * BOX_DIM]` = 700 floats.
    pub gt_boxes: Vec<f32>,
    /// Validity mask of length `MAX_GT_BOXES`.
    pub gt_valid: Vec<bool>,
    /// Per-box class index (length `MAX_GT_BOXES`).
    pub gt_classes: Vec<i64>,
}

impl RadarSnapshot {
    fn empty(time_s: f64) -> Self {
        Self {
            time_s,
            point_cloud: vec![0.0; POINT_CLOUD_SIZE * POINT_DIM],
            gt_boxes: vec![0.0; MAX_GT_BOXES * BOX_DIM],
            gt_valid: vec![false; MAX_GT_BOXES],
            gt_classes: vec![0; MAX_GT_BOXES],
        }
    }
}

/// Errors returned by trajectory-driven synthesis.
#[derive(Debug)]
pub enum TrajectoryRadarError {
    EmptyTrajectory,
    UnsortedWaypoints(usize),
    InvalidClass(u32),
    InvalidSampleRate(f64),
    InvalidNoiseParam(&'static str),
}

impl std::fmt::Display for TrajectoryRadarError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmptyTrajectory => write!(f, "trajectory contains no waypoints"),
            Self::UnsortedWaypoints(i) => {
                write!(f, "waypoints are not sorted by time (index {i} regresses)")
            }
            Self::InvalidClass(c) => write!(f, "class id {c} is out of range [0, 5)"),
            Self::InvalidSampleRate(rate) => {
                write!(f, "sample_rate_hz must be positive, got {rate}")
            }
            Self::InvalidNoiseParam(name) => {
                write!(f, "config field {name} must be positive and finite")
            }
        }
    }
}

impl std::error::Error for TrajectoryRadarError {}

/// Pre-built per-tick sampling distributions, hoisted out of the hot path so
/// failures (zero or negative noise parameters) surface as a recoverable
/// [`TrajectoryRadarError`] rather than a panic from `Distribution::new`.
struct SnapshotDistributions {
    return_noise: Normal<f64>,
    target_intensity: Uniform<f32>,
    clutter_xy: Uniform<f32>,
    clutter_z: Uniform<f32>,
    clutter_intensity: Uniform<f32>,
}

fn build_distributions(
    config: &TrajectoryRadarConfig,
) -> Result<SnapshotDistributions, TrajectoryRadarError> {
    let return_noise = Normal::new(0.0, config.return_position_noise_m)
        .map_err(|_| TrajectoryRadarError::InvalidNoiseParam("return_position_noise_m"))?;
    let target_intensity = Uniform::new(
        config.target_intensity_max * 0.5,
        config.target_intensity_max,
    )
    .map_err(|_| TrajectoryRadarError::InvalidNoiseParam("target_intensity_max"))?;
    let half_ext = config.clutter_half_extent_m as f32;
    let clutter_xy = Uniform::new(-half_ext, half_ext)
        .map_err(|_| TrajectoryRadarError::InvalidNoiseParam("clutter_half_extent_m"))?;
    let clutter_z = Uniform::new(0.0_f32, half_ext * 0.2)
        .map_err(|_| TrajectoryRadarError::InvalidNoiseParam("clutter_half_extent_m"))?;
    let clutter_intensity = Uniform::new(0.0_f32, config.clutter_intensity_max)
        .map_err(|_| TrajectoryRadarError::InvalidNoiseParam("clutter_intensity_max"))?;
    Ok(SnapshotDistributions {
        return_noise,
        target_intensity,
        clutter_xy,
        clutter_z,
        clutter_intensity,
    })
}

/// Synthesise paired `(point_cloud, gt_boxes)` snapshots at the configured
/// sample rate over the union of all targets' trajectory time windows.
///
/// Returns one [`RadarSnapshot`] per snapshot tick. Targets out of detection
/// range at a given tick are dropped from that tick's snapshot but may
/// reappear later as they enter range.
pub fn from_trajectory<R: Rng>(
    targets: &[TargetTrack],
    config: &TrajectoryRadarConfig,
    rng: &mut R,
) -> Result<Vec<RadarSnapshot>, TrajectoryRadarError> {
    validate_inputs(targets, config)?;
    let dt = sample_period(config.sample_rate_hz)?;
    let distributions = build_distributions(config)?;
    let (t_start, t_end) = trajectory_time_bounds(targets);
    let mut snapshots = Vec::new();
    for i in 0..tick_count(t_start, t_end, dt) {
        let t = t_start + i as f64 * dt;
        snapshots.push(make_snapshot(targets, config, &distributions, t, rng));
    }
    Ok(snapshots)
}

/// Synthesise measurement-level radar returns per tick. One inner `Vec` per
/// snapshot tick; entries are missed-detection-filtered (a missed target
/// contributes zero measurements that tick).
///
/// Wraps [`crate::measurement_gen::generate_radar_with_rcs`] tick-by-tick;
/// adds per-class RCS lookup based on the target's `class_id`.
pub fn measurements_from_trajectory<R: Rng>(
    targets: &[TargetTrack],
    config: &TrajectoryRadarConfig,
    rng: &mut R,
) -> Result<Vec<Vec<Measurement>>, TrajectoryRadarError> {
    validate_inputs(targets, config)?;
    let dt = sample_period(config.sample_rate_hz)?;
    let (t_start, t_end) = trajectory_time_bounds(targets);
    let mut per_tick = Vec::new();
    for i in 0..tick_count(t_start, t_end, dt) {
        let t = t_start + i as f64 * dt;
        let mut tick = Vec::new();
        for target in targets {
            if let Some(wp) = interpolate_at(&target.waypoints, t) {
                let translated = translate_to_sensor_frame(&wp, &config.sensor);
                let rcs = class_rcs(config, target.class_id);
                if let Some(m) = crate::measurement_gen::generate_radar_with_rcs(
                    &translated,
                    &config.radar,
                    Some(rcs),
                    rng,
                ) {
                    tick.push(m);
                }
            }
        }
        per_tick.push(tick);
    }
    Ok(per_tick)
}

// ---------------------------------------------------------------------------
// Internals
// ---------------------------------------------------------------------------

/// Compute the per-tick sample period in seconds, surfacing zero/negative
/// `sample_rate_hz` as a recoverable error rather than relying on
/// validate_inputs to prevent the division.
fn sample_period(sample_rate_hz: f64) -> Result<f64, TrajectoryRadarError> {
    if sample_rate_hz <= 0.0 || !sample_rate_hz.is_finite() {
        return Err(TrajectoryRadarError::InvalidSampleRate(sample_rate_hz));
    }
    Ok(1.0 / sample_rate_hz)
}

/// Number of snapshot ticks over `[t_start, t_end]` at step `dt`, matching
/// the inclusive `t <= t_end` sweep with a 1 ns epsilon. Callers drive the
/// sweep off an integer index (`t_start + i * dt`) rather than an
/// accumulating float counter: this avoids drift and the float-loop-counter
/// reliability rule (rust:S2193). `dt > 0` is guaranteed by `sample_period`.
fn tick_count(t_start: f64, t_end: f64, dt: f64) -> usize {
    ((t_end + 1e-9 - t_start) / dt).floor() as usize + 1
}

/// Look up a class's RCS, defaulting to `1.0` for any out-of-range class id.
/// `validate_inputs` rejects out-of-range class ids; this is a defence-in-depth
/// fallback that means callers cannot panic on the lookup.
fn class_rcs(config: &TrajectoryRadarConfig, class_id: u32) -> f64 {
    config
        .class_rcs_m2
        .get(class_id as usize)
        .copied()
        .unwrap_or(1.0)
}

/// Look up a class's default box dimensions; same defence-in-depth fallback
/// as [`class_rcs`].
fn class_box_dims(class_id: u32) -> BoxDimensions {
    CLASS_BOX_DIMS
        .get(class_id as usize)
        .copied()
        .unwrap_or(BoxDimensions {
            length: 10.0,
            width: 10.0,
            height: 5.0,
        })
}

fn validate_inputs(
    targets: &[TargetTrack],
    config: &TrajectoryRadarConfig,
) -> Result<(), TrajectoryRadarError> {
    if targets.is_empty() {
        return Err(TrajectoryRadarError::EmptyTrajectory);
    }
    if config.sample_rate_hz <= 0.0 {
        return Err(TrajectoryRadarError::InvalidSampleRate(
            config.sample_rate_hz,
        ));
    }
    for target in targets {
        if target.waypoints.is_empty() {
            return Err(TrajectoryRadarError::EmptyTrajectory);
        }
        if target.class_id as usize >= CLASS_BOX_DIMS.len() {
            return Err(TrajectoryRadarError::InvalidClass(target.class_id));
        }
        let regression = target
            .waypoints
            .windows(2)
            .position(|pair| matches!(pair, [a, b] if b.time < a.time));
        if let Some(i) = regression {
            return Err(TrajectoryRadarError::UnsortedWaypoints(i + 1));
        }
    }
    Ok(())
}

fn trajectory_time_bounds(targets: &[TargetTrack]) -> (f64, f64) {
    let mut t_start = f64::INFINITY;
    let mut t_end = f64::NEG_INFINITY;
    for target in targets {
        // `validate_inputs` guarantees non-empty waypoints, but we still
        // pattern-match rather than unwrap so SonarCloud doesn't flag a
        // potential panic.
        let Some(first) = target.waypoints.first().map(|wp| wp.time) else {
            continue;
        };
        let Some(last) = target.waypoints.last().map(|wp| wp.time) else {
            continue;
        };
        if first < t_start {
            t_start = first;
        }
        if last > t_end {
            t_end = last;
        }
    }
    (t_start, t_end)
}

fn interpolate_at(waypoints: &[Waypoint], t: f64) -> Option<Waypoint> {
    let first = waypoints.first()?;
    let last = waypoints.last()?;
    if t < first.time || t > last.time {
        return None;
    }
    // Binary search for the segment containing `t`. `mid` uses the
    // overflow-safe form `lo + (hi - lo) / 2` rather than `(lo + hi) / 2`
    // (CWE-190 / the classic broken-binary-search bug), and every index
    // goes through `.get()` so the loop cannot panic.
    let mut lo = 0usize;
    let mut hi = waypoints.len();
    while lo + 1 < hi {
        let mid = lo + (hi - lo) / 2;
        match waypoints.get(mid) {
            Some(wp) if wp.time <= t => lo = mid,
            Some(_) => hi = mid,
            None => break,
        }
    }
    let lower = waypoints.get(lo)?;
    match waypoints.get(lo + 1) {
        Some(upper) => Some(Waypoint::interpolate(lower, upper, t)),
        None => Some(lower.clone()),
    }
}

fn translate_to_sensor_frame(wp: &Waypoint, sensor: &SensorPose) -> Waypoint {
    Waypoint {
        time: wp.time,
        position: [
            wp.position[0] - sensor.position_enu_m[0],
            wp.position[1] - sensor.position_enu_m[1],
            wp.position[2] - sensor.position_enu_m[2],
        ],
        velocity: wp.velocity,
    }
}

fn range_to_sensor(wp: &Waypoint, sensor: &SensorPose) -> f64 {
    let dx = wp.position[0] - sensor.position_enu_m[0];
    let dy = wp.position[1] - sensor.position_enu_m[1];
    let dz = wp.position[2] - sensor.position_enu_m[2];
    (dx * dx + dy * dy + dz * dz).sqrt()
}

fn yaw_from_velocity(vel: &[f64; 3]) -> f64 {
    vel[1].atan2(vel[0])
}

fn write_box(snapshot: &mut RadarSnapshot, slot: usize, wp: &Waypoint, target: &TargetTrack) {
    let dims = target
        .size_override
        .unwrap_or_else(|| class_box_dims(target.class_id));
    let base = slot * BOX_DIM;
    snapshot.gt_boxes[base] = wp.position[0] as f32;
    snapshot.gt_boxes[base + 1] = wp.position[1] as f32;
    snapshot.gt_boxes[base + 2] = wp.position[2] as f32;
    snapshot.gt_boxes[base + 3] = dims.length;
    snapshot.gt_boxes[base + 4] = dims.width;
    snapshot.gt_boxes[base + 5] = dims.height;
    snapshot.gt_boxes[base + 6] = yaw_from_velocity(&wp.velocity) as f32;
    snapshot.gt_valid[slot] = true;
    snapshot.gt_classes[slot] = target.class_id as i64;
}

fn write_target_returns<R: Rng>(
    snapshot: &mut RadarSnapshot,
    point_cursor: &mut usize,
    wp: &Waypoint,
    config: &TrajectoryRadarConfig,
    distributions: &SnapshotDistributions,
    rng: &mut R,
) {
    for _ in 0..config.returns_per_target {
        if *point_cursor >= POINT_CLOUD_SIZE {
            break;
        }
        let base = *point_cursor * POINT_DIM;
        snapshot.point_cloud[base] =
            (wp.position[0] + distributions.return_noise.sample(rng)) as f32;
        snapshot.point_cloud[base + 1] =
            (wp.position[1] + distributions.return_noise.sample(rng)) as f32;
        snapshot.point_cloud[base + 2] =
            (wp.position[2] + distributions.return_noise.sample(rng)) as f32;
        snapshot.point_cloud[base + 3] = distributions.target_intensity.sample(rng);
        *point_cursor += 1;
    }
}

fn fill_clutter<R: Rng>(
    snapshot: &mut RadarSnapshot,
    point_cursor: &mut usize,
    distributions: &SnapshotDistributions,
    rng: &mut R,
) {
    while *point_cursor < POINT_CLOUD_SIZE {
        let base = *point_cursor * POINT_DIM;
        snapshot.point_cloud[base] = distributions.clutter_xy.sample(rng);
        snapshot.point_cloud[base + 1] = distributions.clutter_xy.sample(rng);
        snapshot.point_cloud[base + 2] = distributions.clutter_z.sample(rng);
        snapshot.point_cloud[base + 3] = distributions.clutter_intensity.sample(rng);
        *point_cursor += 1;
    }
}

fn make_snapshot<R: Rng>(
    targets: &[TargetTrack],
    config: &TrajectoryRadarConfig,
    distributions: &SnapshotDistributions,
    t: f64,
    rng: &mut R,
) -> RadarSnapshot {
    let mut snapshot = RadarSnapshot::empty(t);
    let mut box_slot = 0usize;
    let mut point_cursor = 0usize;
    for target in targets {
        let Some(wp) = interpolate_at(&target.waypoints, t) else {
            continue;
        };
        let range = range_to_sensor(&wp, &config.sensor);
        if range > config.max_range_m || range < 1.0 {
            continue;
        }
        let rcs = class_rcs(config, target.class_id);
        let pd = detection_probability(range, Some(rcs), &config.radar);
        if rng.random::<f64>() > pd {
            continue;
        }
        if box_slot < MAX_GT_BOXES {
            write_box(&mut snapshot, box_slot, &wp, target);
            box_slot += 1;
        }
        write_target_returns(
            &mut snapshot,
            &mut point_cursor,
            &wp,
            config,
            distributions,
            rng,
        );
    }
    fill_clutter(&mut snapshot, &mut point_cursor, distributions, rng);
    snapshot
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    fn cv_target(x0: f64, y0: f64, z0: f64, vx: f64, vy: f64, vz: f64, t_max: f64) -> TargetTrack {
        let mut waypoints = Vec::new();
        let mut t = 0.0;
        while t <= t_max + 1e-9 {
            waypoints.push(Waypoint {
                time: t,
                position: [x0 + vx * t, y0 + vy * t, z0 + vz * t],
                velocity: [vx, vy, vz],
            });
            t += 1.0; // 1 Hz, like ADS-B
        }
        TargetTrack {
            waypoints,
            class_id: 1, // heavy-fixed-wing
            size_override: None,
        }
    }

    fn detection_friendly_config() -> TrajectoryRadarConfig {
        TrajectoryRadarConfig {
            radar: RadarConfig {
                p_detection: 1.0,
                ..RadarConfig::default()
            },
            sample_rate_hz: 10.0,
            ..TrajectoryRadarConfig::default()
        }
    }

    #[test]
    fn waypoint_interpolation_is_linear() {
        let a = Waypoint {
            time: 0.0,
            position: [0.0, 0.0, 0.0],
            velocity: [100.0, 0.0, 0.0],
        };
        let b = Waypoint {
            time: 10.0,
            position: [1000.0, 0.0, 0.0],
            velocity: [100.0, 0.0, 0.0],
        };
        let mid = Waypoint::interpolate(&a, &b, 5.0);
        assert!((mid.position[0] - 500.0).abs() < 1e-6);
        assert!((mid.time - 5.0).abs() < 1e-6);
    }

    #[test]
    fn waypoint_interpolation_clamps_outside_range() {
        let a = Waypoint {
            time: 0.0,
            position: [0.0; 3],
            velocity: [0.0; 3],
        };
        let b = Waypoint {
            time: 1.0,
            position: [10.0, 0.0, 0.0],
            velocity: [10.0, 0.0, 0.0],
        };
        // Before window — clamps to a.
        let before = Waypoint::interpolate(&a, &b, -5.0);
        assert!((before.position[0] - 0.0).abs() < 1e-9);
        // After window — clamps to b.
        let after = Waypoint::interpolate(&a, &b, 5.0);
        assert!((after.position[0] - 10.0).abs() < 1e-9);
    }

    #[test]
    fn from_trajectory_emits_at_sample_rate() {
        let target = cv_target(1000.0, 0.0, 3000.0, 100.0, 0.0, 0.0, 10.0);
        let config = detection_friendly_config();
        let mut rng = StdRng::seed_from_u64(42);
        let snapshots = from_trajectory(&[target], &config, &mut rng).unwrap();
        // 0..=10 s at 10 Hz inclusive of both endpoints = ~101 snapshots.
        assert!(
            (snapshots.len() as i64 - 101).abs() <= 1,
            "expected ~101 snapshots at 10 Hz over 10 s, got {}",
            snapshots.len()
        );
    }

    #[test]
    fn point_cloud_has_target_cluster_near_ground_truth() {
        let target = cv_target(1000.0, 0.0, 3000.0, 100.0, 0.0, 0.0, 5.0);
        let config = detection_friendly_config();
        let mut rng = StdRng::seed_from_u64(7);
        let snapshots = from_trajectory(std::slice::from_ref(&target), &config, &mut rng).unwrap();
        let snap = &snapshots[0];

        // Box matches the trajectory's t=0 position.
        assert!(snap.gt_valid[0]);
        assert!((snap.gt_boxes[0] - 1000.0).abs() < 1e-3);
        assert!((snap.gt_boxes[1] - 0.0).abs() < 1e-3);
        assert!((snap.gt_boxes[2] - 3000.0).abs() < 1e-3);
        assert_eq!(snap.gt_classes[0], 1);

        // At least one point cloud return should be within ~30 m of the box.
        let bx = snap.gt_boxes[0];
        let by = snap.gt_boxes[1];
        let bz = snap.gt_boxes[2];
        let mut close = 0usize;
        for i in 0..POINT_CLOUD_SIZE {
            let dx = snap.point_cloud[i * POINT_DIM] - bx;
            let dy = snap.point_cloud[i * POINT_DIM + 1] - by;
            let dz = snap.point_cloud[i * POINT_DIM + 2] - bz;
            let r = (dx * dx + dy * dy + dz * dz).sqrt();
            if r < 30.0 {
                close += 1;
            }
        }
        assert!(
            close >= config.returns_per_target / 2,
            "expected at least {} close returns, got {}",
            config.returns_per_target / 2,
            close
        );
    }

    #[test]
    fn point_cloud_tensors_have_fixed_shape() {
        let target = cv_target(1000.0, 0.0, 3000.0, 100.0, 0.0, 0.0, 2.0);
        let config = detection_friendly_config();
        let mut rng = StdRng::seed_from_u64(123);
        let snapshots = from_trajectory(&[target], &config, &mut rng).unwrap();
        for snap in &snapshots {
            assert_eq!(snap.point_cloud.len(), POINT_CLOUD_SIZE * POINT_DIM);
            assert_eq!(snap.gt_boxes.len(), MAX_GT_BOXES * BOX_DIM);
            assert_eq!(snap.gt_valid.len(), MAX_GT_BOXES);
            assert_eq!(snap.gt_classes.len(), MAX_GT_BOXES);
        }
    }

    #[test]
    fn out_of_range_target_is_dropped() {
        let mut config = detection_friendly_config();
        config.max_range_m = 500.0;
        let target = cv_target(10_000.0, 0.0, 3000.0, 0.0, 0.0, 0.0, 1.0);
        let mut rng = StdRng::seed_from_u64(1);
        let snapshots = from_trajectory(&[target], &config, &mut rng).unwrap();
        for snap in &snapshots {
            assert!(!snap.gt_valid[0], "out-of-range target should not be boxed");
        }
    }

    #[test]
    fn measurements_from_trajectory_emits_per_tick() {
        let target = cv_target(1000.0, 0.0, 3000.0, 100.0, 0.0, 0.0, 5.0);
        let config = detection_friendly_config();
        let mut rng = StdRng::seed_from_u64(99);
        let per_tick = measurements_from_trajectory(&[target], &config, &mut rng).unwrap();
        assert!(per_tick.len() > 40); // ~51 ticks over 5 s at 10 Hz
        let mut total = 0usize;
        for tick in &per_tick {
            total += tick.len();
        }
        // p_detection=1.0 → every tick should yield one measurement.
        assert!(
            total > per_tick.len() / 2,
            "expected many radar measurements, got {} over {} ticks",
            total,
            per_tick.len()
        );
    }

    #[test]
    fn validates_empty_targets() {
        let config = detection_friendly_config();
        let mut rng = StdRng::seed_from_u64(0);
        assert!(matches!(
            from_trajectory::<StdRng>(&[], &config, &mut rng),
            Err(TrajectoryRadarError::EmptyTrajectory)
        ));
    }

    #[test]
    fn validates_invalid_class() {
        let mut target = cv_target(1000.0, 0.0, 3000.0, 100.0, 0.0, 0.0, 1.0);
        target.class_id = 99;
        let config = detection_friendly_config();
        let mut rng = StdRng::seed_from_u64(0);
        assert!(matches!(
            from_trajectory(&[target], &config, &mut rng),
            Err(TrajectoryRadarError::InvalidClass(99))
        ));
    }

    #[test]
    fn validates_zero_sample_rate() {
        let target = cv_target(1000.0, 0.0, 3000.0, 100.0, 0.0, 0.0, 1.0);
        let mut config = detection_friendly_config();
        config.sample_rate_hz = 0.0;
        let mut rng = StdRng::seed_from_u64(0);
        assert!(matches!(
            from_trajectory(&[target], &config, &mut rng),
            Err(TrajectoryRadarError::InvalidSampleRate(_))
        ));
    }
}
