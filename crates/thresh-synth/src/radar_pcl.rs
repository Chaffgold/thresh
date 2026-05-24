//! Synthesize radar point-cloud training examples from trajectories.
//!
//! Produces examples that match a transformer-based point-cloud detector
//! contract (see the `flight-data-training-pipeline` OpenSpec change,
//! Decisions 8–10):
//!
//! - point-cloud input of shape `(1, NUM_POINTS, POINT_DIM)` — exactly
//!   [`NUM_POINTS`] points, each `[x, y, z, intensity]`;
//! - ground-truth boxes of shape `(1, MAX_BOXES, BOX_DIM)` — up to
//!   [`MAX_BOXES`] boxes `[x, y, z, L, W, H, yaw]`, padded with a validity
//!   mask;
//! - a per-box class index in `[0, 5)`.
//!
//! The working frame is a local ENU frame centred at the sensor: target ENU
//! positions have the sensor position subtracted, and `yaw` is the target
//! heading `atan2(vy, vx)`.
//!
//! Two APIs are provided:
//!
//! - [`from_trajectory`] — "Track A": dense point-cloud snapshots with GT
//!   boxes, suitable for training a point-cloud detector.
//! - [`measurements_from_trajectory`] — "Track B": a stream of
//!   [`Measurement::Radar`] detections (range / azimuth / elevation plus a
//!   Doppler range-rate), reusing the crate's existing radar measurement
//!   generator.
//!
//! The radar equation, SNR, and detection-probability logic are reused from
//! [`crate::measurement_gen`] rather than reimplemented here.

use rand::Rng;
use rand_distr::{Distribution, Normal};
use thresh_core::measurement::Measurement;

use crate::measurement_gen::{
    RadarConfig, RadarEquationConfig, albersheim_pd, compute_snr, generate_radar_with_rcs,
};
use crate::trajectory::Waypoint;

/// Number of points in every point-cloud snapshot (detector input length).
pub const NUM_POINTS: usize = 1000;

/// Maximum number of ground-truth boxes per snapshot (detector output length).
pub const MAX_BOXES: usize = 100;

/// Per-point feature dimension: `[x, y, z, intensity]`.
pub const POINT_DIM: usize = 4;

/// Per-box feature dimension: `[x, y, z, L, W, H, yaw]`.
pub const BOX_DIM: usize = 7;

/// Coarse target detection classes (maps to a `classes` tensor index).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DetectionClass {
    /// Light fixed-wing aircraft (small general aviation).
    LightFixedWing,
    /// Heavy fixed-wing aircraft (airliner / transport).
    HeavyFixedWing,
    /// Rotorcraft (helicopter).
    Rotorcraft,
    /// Glider, balloon, or small UAV.
    GliderOrBalloonOrUav,
    /// Anything not covered by the other classes.
    Other,
}

impl DetectionClass {
    /// Class index in `0..=4`, matching the variant declaration order.
    pub fn index(self) -> u8 {
        match self {
            DetectionClass::LightFixedWing => 0,
            DetectionClass::HeavyFixedWing => 1,
            DetectionClass::Rotorcraft => 2,
            DetectionClass::GliderOrBalloonOrUav => 3,
            DetectionClass::Other => 4,
        }
    }
}

/// Physical extent, radar cross-section, and class for a single target.
#[derive(Debug, Clone)]
pub struct TargetProfile {
    /// Radar cross-section (square metres).
    pub rcs_m2: f64,
    /// Bounding-box dimensions `[length, width, height]` (metres).
    pub dims_lwh: [f64; 3],
    /// Detection class for this target.
    pub class: DetectionClass,
}

impl TargetProfile {
    /// Preset for a light fixed-wing aircraft (~1 m² RCS).
    pub fn light_aircraft() -> Self {
        Self {
            rcs_m2: 1.0,
            dims_lwh: [10.0, 10.0, 3.0],
            class: DetectionClass::LightFixedWing,
        }
    }

    /// Preset for an airliner-class heavy fixed-wing aircraft (~100 m² RCS).
    pub fn airliner() -> Self {
        Self {
            rcs_m2: 100.0,
            dims_lwh: [60.0, 60.0, 18.0],
            class: DetectionClass::HeavyFixedWing,
        }
    }
}

/// Configuration for point-cloud snapshot synthesis.
#[derive(Debug, Clone)]
pub struct RadarPclConfig {
    /// Sensor position in the world ENU frame (metres). Subtracted from
    /// target positions to obtain the sensor-centred ENU frame.
    pub sensor_position_enu: [f64; 3],
    /// Snapshot sampling period (seconds).
    pub sample_period_s: f64,
    /// Radar equation parameters for SNR / detection probability.
    pub radar_equation: RadarEquationConfig,
    /// Number of point returns emitted per detected target.
    pub returns_per_target: usize,
    /// Per-axis Gaussian std of point returns around the target (metres).
    pub point_position_sigma_m: f64,
    /// Number of low-intensity clutter points added per snapshot.
    pub clutter_points: usize,
    /// Maximum detection range (metres); targets beyond this are absent.
    pub max_range_m: f64,
}

impl Default for RadarPclConfig {
    fn default() -> Self {
        Self::x_band_default()
    }
}

impl RadarPclConfig {
    /// Default X-band surveillance configuration (sensor at the origin).
    pub fn x_band_default() -> Self {
        Self {
            sensor_position_enu: [0.0, 0.0, 0.0],
            sample_period_s: 1.0,
            radar_equation: RadarEquationConfig::x_band_surveillance(),
            returns_per_target: 20,
            point_position_sigma_m: 5.0,
            clutter_points: 50,
            max_range_m: 200_000.0,
        }
    }
}

/// One point-cloud snapshot matching the detector tensor contract.
///
/// `points` always has exactly [`NUM_POINTS`] entries; `boxes`,
/// `box_validity`, and `box_classes` each have exactly [`MAX_BOXES`] entries
/// (padded with zeros / `false` / `0` where invalid).
#[derive(Debug, Clone)]
pub struct PointCloudSnapshot {
    /// Snapshot time (seconds).
    pub time: f64,
    /// Point cloud, each point `[x, y, z, intensity]` (sensor-centred ENU).
    pub points: Vec<[f32; POINT_DIM]>,
    /// Ground-truth boxes, each `[x, y, z, L, W, H, yaw]`.
    pub boxes: Vec<[f32; BOX_DIM]>,
    /// Per-box validity mask (`true` for real boxes).
    pub box_validity: Vec<bool>,
    /// Per-box class index (`0..=4`).
    pub box_classes: Vec<u8>,
}

/// Linearly interpolate `(position, velocity)` at time `t` from waypoints.
///
/// Returns `None` if `t` is outside the waypoint time span or there are
/// fewer than two waypoints. Waypoints are assumed time-ordered.
fn interpolate_state(waypoints: &[Waypoint], t: f64) -> Option<([f64; 3], [f64; 3])> {
    if waypoints.len() < 2 {
        return None;
    }
    let first = waypoints.first()?;
    let last = waypoints.last()?;
    if t < first.time || t > last.time {
        return None;
    }
    let upper = waypoints.iter().position(|w| w.time >= t)?;
    if upper == 0 {
        return Some((first.position, first.velocity));
    }
    let a = &waypoints[upper - 1];
    let b = &waypoints[upper];
    let span = b.time - a.time;
    let frac = if span > 1e-12 {
        (t - a.time) / span
    } else {
        0.0
    };
    let lerp3 = |x: &[f64; 3], y: &[f64; 3]| {
        [
            x[0] + (y[0] - x[0]) * frac,
            x[1] + (y[1] - x[1]) * frac,
            x[2] + (y[2] - x[2]) * frac,
        ]
    };
    Some((
        lerp3(&a.position, &b.position),
        lerp3(&a.velocity, &b.velocity),
    ))
}

/// Minimum end-time across all targets' waypoint lists, or `None` if any
/// target has fewer than two waypoints.
fn min_end_time(targets: &[(Vec<Waypoint>, TargetProfile)]) -> Option<f64> {
    let mut min_t: Option<f64> = None;
    for (wps, _) in targets {
        let end = wps.last()?.time;
        if wps.len() < 2 {
            return None;
        }
        min_t = Some(min_t.map_or(end, |m: f64| m.min(end)));
    }
    min_t
}

/// Sensor-relative position and slant range for a target position.
fn relative_position(target_pos: &[f64; 3], sensor: &[f64; 3]) -> ([f64; 3], f64) {
    let rel = [
        target_pos[0] - sensor[0],
        target_pos[1] - sensor[1],
        target_pos[2] - sensor[2],
    ];
    let range = (rel[0] * rel[0] + rel[1] * rel[1] + rel[2] * rel[2]).sqrt();
    (rel, range)
}

/// Detection-probability `(snr_db, pd)` for a target at `range` with `rcs`.
fn detection_pd(range: f64, rcs: f64, req: &RadarEquationConfig) -> (f64, f64) {
    let snr_linear = compute_snr(range, rcs, req);
    let snr_db = 10.0 * snr_linear.max(1e-30).log10();
    (snr_db, albersheim_pd(snr_db, req.pfa))
}

/// Append the GT box for a detected target to the box buffers.
fn push_box(
    boxes: &mut Vec<[f32; BOX_DIM]>,
    classes: &mut Vec<u8>,
    rel: &[f64; 3],
    vel: &[f64; 3],
    profile: &TargetProfile,
) {
    let yaw = vel[1].atan2(vel[0]);
    boxes.push([
        rel[0] as f32,
        rel[1] as f32,
        rel[2] as f32,
        profile.dims_lwh[0] as f32,
        profile.dims_lwh[1] as f32,
        profile.dims_lwh[2] as f32,
        yaw as f32,
    ]);
    classes.push(profile.class.index());
}

/// Append `returns_per_target` Gaussian-clustered point returns for a target.
fn push_target_returns<R: Rng>(
    points: &mut Vec<[f32; POINT_DIM]>,
    rel: &[f64; 3],
    snr_db: f64,
    config: &RadarPclConfig,
    rng: &mut R,
) {
    let offset = Normal::new(0.0, config.point_position_sigma_m).unwrap();
    let intensity = (snr_db / 40.0).clamp(0.0, 1.0) as f32;
    for _ in 0..config.returns_per_target {
        let dx: f64 = offset.sample(rng);
        let dy: f64 = offset.sample(rng);
        let dz: f64 = offset.sample(rng);
        points.push([
            (rel[0] + dx) as f32,
            (rel[1] + dy) as f32,
            (rel[2] + dz) as f32,
            intensity,
        ]);
    }
}

/// Process one target for a snapshot: detect, emit box and returns.
///
/// Returns early without emitting anything if the target is out of range,
/// undetected this scan, or the box budget [`MAX_BOXES`] is exhausted.
fn process_target<R: Rng>(
    points: &mut Vec<[f32; POINT_DIM]>,
    boxes: &mut Vec<[f32; BOX_DIM]>,
    classes: &mut Vec<u8>,
    state: (&[f64; 3], &[f64; 3]),
    profile: &TargetProfile,
    config: &RadarPclConfig,
    rng: &mut R,
) {
    if boxes.len() >= MAX_BOXES {
        return;
    }
    let (pos, vel) = state;
    let (rel, range) = relative_position(pos, &config.sensor_position_enu);
    if range > config.max_range_m || range < 1e-6 {
        return;
    }
    let (snr_db, pd) = detection_pd(range, profile.rcs_m2, &config.radar_equation);
    if rng.random::<f64>() > pd {
        return;
    }
    push_box(boxes, classes, &rel, vel, profile);
    push_target_returns(points, &rel, snr_db, config, rng);
}

/// Append low-intensity clutter points at uniform-random positions.
///
/// Clutter is scattered uniformly within a cube of half-width
/// `max_range_m / 2` per axis (a plausible coverage volume), with a small
/// uniform intensity in `[0, 0.1]`.
fn fill_clutter<R: Rng>(points: &mut Vec<[f32; POINT_DIM]>, config: &RadarPclConfig, rng: &mut R) {
    let half = config.max_range_m / 2.0;
    for _ in 0..config.clutter_points {
        let x = rng.random_range(-half..half);
        let y = rng.random_range(-half..half);
        let z = rng.random_range(-half..half);
        let intensity = rng.random_range(0.0..0.1_f32);
        points.push([x as f32, y as f32, z as f32, intensity]);
    }
}

/// Truncate or zero-pad `points` to exactly [`NUM_POINTS`] entries.
fn pad_points(points: &mut Vec<[f32; POINT_DIM]>) {
    points.truncate(NUM_POINTS);
    points.resize(NUM_POINTS, [0.0; POINT_DIM]);
}

/// Pad box buffers to exactly [`MAX_BOXES`] entries and build the mask.
fn pad_boxes(boxes: &mut Vec<[f32; BOX_DIM]>, classes: &mut Vec<u8>) -> Vec<bool> {
    let valid_count = boxes.len().min(MAX_BOXES);
    boxes.truncate(MAX_BOXES);
    classes.truncate(MAX_BOXES);
    boxes.resize(MAX_BOXES, [0.0; BOX_DIM]);
    classes.resize(MAX_BOXES, 0);
    let mut validity = vec![false; MAX_BOXES];
    for v in validity.iter_mut().take(valid_count) {
        *v = true;
    }
    validity
}

/// Build a single snapshot at time `t` for all targets.
fn build_snapshot<R: Rng>(
    targets: &[(Vec<Waypoint>, TargetProfile)],
    t: f64,
    config: &RadarPclConfig,
    rng: &mut R,
) -> PointCloudSnapshot {
    let mut points: Vec<[f32; POINT_DIM]> = Vec::new();
    let mut boxes: Vec<[f32; BOX_DIM]> = Vec::new();
    let mut classes: Vec<u8> = Vec::new();

    for (wps, profile) in targets {
        if let Some((pos, vel)) = interpolate_state(wps, t) {
            process_target(
                &mut points,
                &mut boxes,
                &mut classes,
                (&pos, &vel),
                profile,
                config,
                rng,
            );
        }
    }

    fill_clutter(&mut points, config, rng);
    pad_points(&mut points);
    let box_validity = pad_boxes(&mut boxes, &mut classes);

    PointCloudSnapshot {
        time: t,
        points,
        boxes,
        box_validity,
        box_classes: classes,
    }
}

/// Synthesize point-cloud snapshots (Track A) from one or more target
/// trajectories.
///
/// Snapshots are emitted at `t = 0, sample_period_s, 2*sample_period_s, …`
/// up to and including the minimum end-time across all targets' waypoints.
/// Each snapshot satisfies the detector tensor contract (see
/// [`PointCloudSnapshot`]). Targets that are out of range or missed (per the
/// radar-equation detection probability) contribute no box or returns for
/// that snapshot.
pub fn from_trajectory<R: Rng>(
    targets: &[(Vec<Waypoint>, TargetProfile)],
    config: &RadarPclConfig,
    rng: &mut R,
) -> Vec<PointCloudSnapshot> {
    let Some(end_time) = min_end_time(targets) else {
        return Vec::new();
    };
    let dt = config.sample_period_s.max(1e-9);
    let n_steps = (end_time / dt).floor() as usize;

    let mut snapshots = Vec::with_capacity(n_steps + 1);
    for k in 0..=n_steps {
        let t = (k as f64) * dt;
        snapshots.push(build_snapshot(targets, t, config, rng));
    }
    snapshots
}

/// Radial range-rate (Doppler) of a target, sensor assumed at the origin.
fn range_rate(pos: &[f64; 3], vel: &[f64; 3]) -> f64 {
    let range = (pos[0] * pos[0] + pos[1] * pos[1] + pos[2] * pos[2]).sqrt();
    if range < 1e-9 {
        return 0.0;
    }
    (vel[0] * pos[0] + vel[1] * pos[1] + vel[2] * pos[2]) / range
}

/// Synthesize a radar measurement stream (Track B) from a trajectory.
///
/// For each waypoint, a noisy [`Measurement::Radar`] is generated via
/// [`generate_radar_with_rcs`] (which assumes the sensor is at the ENU
/// origin). Missed or out-of-range waypoints contribute nothing. Each
/// returned measurement augments the base detection with a Doppler
/// `range_rate`: the noise-free radial velocity plus Gaussian noise of std
/// `doppler_sigma`.
pub fn measurements_from_trajectory<R: Rng>(
    waypoints: &[Waypoint],
    rcs_m2: f64,
    config: &RadarConfig,
    doppler_sigma: f64,
    rng: &mut R,
) -> Vec<Measurement> {
    let doppler_noise = Normal::new(0.0, doppler_sigma.max(0.0)).unwrap();
    let mut out = Vec::new();
    for wp in waypoints {
        let Some(Measurement::Radar {
            range,
            azimuth,
            elevation,
            time,
            sensor_id,
            ..
        }) = generate_radar_with_rcs(wp, config, Some(rcs_m2), rng)
        else {
            continue;
        };
        let doppler = range_rate(&wp.position, &wp.velocity) + doppler_noise.sample(rng);
        out.push(Measurement::Radar {
            range,
            azimuth,
            elevation,
            range_rate: Some(doppler),
            time,
            sensor_id,
        });
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::trajectory::{Segment, SegmentType, Trajectory};
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    fn cv_waypoints(
        initial_position: [f64; 3],
        velocity: [f64; 3],
        duration: f64,
    ) -> Vec<Waypoint> {
        Trajectory {
            target_id: 0,
            initial_position,
            initial_velocity: velocity,
            segments: vec![Segment {
                segment_type: SegmentType::Cv,
                duration,
            }],
            dt: 1.0,
        }
        .generate()
    }

    #[test]
    fn snapshot_shapes_match_onnx_contract() {
        let wps = cv_waypoints([5_000.0, 0.0, 1_000.0], [200.0, 0.0, 0.0], 10.0);
        let targets = vec![(wps, TargetProfile::light_aircraft())];
        let config = RadarPclConfig::x_band_default();
        let mut rng = StdRng::seed_from_u64(7);

        let snaps = from_trajectory(&targets, &config, &mut rng);
        assert!(!snaps.is_empty());
        for s in &snaps {
            assert_eq!(s.points.len(), NUM_POINTS);
            assert_eq!(s.boxes.len(), MAX_BOXES);
            assert_eq!(s.box_validity.len(), MAX_BOXES);
            assert_eq!(s.box_classes.len(), MAX_BOXES);
        }
    }

    #[test]
    fn cv_target_produces_cluster_near_gt_box() {
        // Close-range target → high Pd. Sensor offset from origin to exercise
        // the sensor-relative transform.
        let wps = cv_waypoints([3_000.0, 500.0, 800.0], [150.0, 0.0, 0.0], 8.0);
        let targets = vec![(wps, TargetProfile::light_aircraft())];
        let config = RadarPclConfig {
            sensor_position_enu: [100.0, -50.0, 0.0],
            clutter_points: 30,
            ..RadarPclConfig::x_band_default()
        };
        let mut rng = StdRng::seed_from_u64(42);

        let snaps = from_trajectory(&targets, &config, &mut rng);

        let mut checked = false;
        for s in &snaps {
            let Some(box_idx) = s.box_validity.iter().position(|&v| v) else {
                continue;
            };
            let bx = s.boxes[box_idx];
            // The valid box center should match the target's sensor-relative
            // position; recompute from the trajectory at this time.
            let (pos, _vel) = interpolate_state(&targets[0].0, s.time).unwrap();
            let (rel, _) = relative_position(&pos, &config.sensor_position_enu);
            assert!((f64::from(bx[0]) - rel[0]).abs() < 1.0);
            assert!((f64::from(bx[1]) - rel[1]).abs() < 1.0);
            assert!((f64::from(bx[2]) - rel[2]).abs() < 1.0);

            // Centroid of the target-return points (intensity > 0.1 separates
            // them from the low-intensity clutter).
            let mut sum = [0.0f64; 3];
            let mut count = 0usize;
            for p in &s.points {
                if p[3] > 0.1 {
                    sum[0] += f64::from(p[0]);
                    sum[1] += f64::from(p[1]);
                    sum[2] += f64::from(p[2]);
                    count += 1;
                }
            }
            assert!(count > 0, "expected target-return points");
            let centroid = [
                sum[0] / count as f64,
                sum[1] / count as f64,
                sum[2] / count as f64,
            ];
            let tol = 3.0 * config.point_position_sigma_m;
            assert!((centroid[0] - f64::from(bx[0])).abs() < tol);
            assert!((centroid[1] - f64::from(bx[1])).abs() < tol);
            assert!((centroid[2] - f64::from(bx[2])).abs() < tol);
            checked = true;
            break;
        }
        assert!(checked, "no snapshot had a valid box");
    }

    #[test]
    fn measurements_have_expected_noise() {
        let wps = cv_waypoints([10_000.0, 5_000.0, 3_000.0], [250.0, 0.0, 0.0], 500.0);
        let config = RadarConfig {
            p_detection: 1.0,
            clutter_rate: 0.0,
            ..Default::default()
        };
        let mut rng = StdRng::seed_from_u64(123);

        let meas = measurements_from_trajectory(&wps, 5.0, &config, 2.0, &mut rng);
        assert!(meas.len() > 100);

        let mut residuals = Vec::new();
        for (wp, m) in wps.iter().zip(meas.iter()) {
            // Track B keeps every detection in order; with p_detection = 1.0
            // and clutter_rate = 0.0, measurement k corresponds to waypoint k.
            let Measurement::Radar {
                range, range_rate, ..
            } = m
            else {
                panic!("expected radar measurement");
            };
            assert!(range_rate.is_some());
            assert!(range_rate.unwrap().is_finite());
            let true_range =
                (wp.position[0].powi(2) + wp.position[1].powi(2) + wp.position[2].powi(2)).sqrt();
            residuals.push(range - true_range);
        }

        let n = residuals.len() as f64;
        let mean = residuals.iter().sum::<f64>() / n;
        let var = residuals.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / n;
        let std = var.sqrt();
        assert!(mean.abs() < 0.3 * config.range_sigma, "mean bias {mean}");
        assert!(
            (std - config.range_sigma).abs() < 0.3 * config.range_sigma,
            "std {std} vs {}",
            config.range_sigma
        );
    }

    #[test]
    fn target_beyond_max_range_has_no_valid_box() {
        let wps = cv_waypoints([500_000.0, 0.0, 10_000.0], [200.0, 0.0, 0.0], 10.0);
        let targets = vec![(wps, TargetProfile::airliner())];
        let config = RadarPclConfig {
            max_range_m: 200_000.0,
            ..RadarPclConfig::x_band_default()
        };
        let mut rng = StdRng::seed_from_u64(9);

        let snaps = from_trajectory(&targets, &config, &mut rng);
        assert!(!snaps.is_empty());
        for s in &snaps {
            assert!(
                s.box_validity.iter().all(|&v| !v),
                "no box expected beyond max range"
            );
        }
    }

    #[test]
    fn detection_class_indices() {
        let variants = [
            DetectionClass::LightFixedWing,
            DetectionClass::HeavyFixedWing,
            DetectionClass::Rotorcraft,
            DetectionClass::GliderOrBalloonOrUav,
            DetectionClass::Other,
        ];
        let indices: Vec<u8> = variants.iter().map(|c| c.index()).collect();
        assert_eq!(indices, vec![0, 1, 2, 3, 4]);
    }
}
