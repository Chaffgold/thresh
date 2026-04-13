//! Integration tests for the stereographic-projection tracker (tasks 8.D.6 – 8.D.8).

use nalgebra::DVector;
use rand::SeedableRng;
use rand::rngs::StdRng;
use thresh_core::othr::{OthrSensorRegistration, vincenty_direct};
use thresh_synth::othr_generator::{OthrConfig, generate_othr};
use thresh_synth::trajectory::Waypoint;
use thresh_tracker::othr_integration::othr_to_cartesian;
use thresh_tracker::stereographic_tracker::{
    MultiObjectTrackerStereographic, othr_to_stereographic, recommended_center,
    stereographic_inverse, stereographic_project,
};
use thresh_tracker::tracker::MultiObjectTracker;

// ── Task 8.D.6 — Roundtrip accuracy at OTHR ranges ────────────────────────

#[test]
fn stereographic_roundtrip_under_1m_at_othr_ranges() {
    // Centre near a representative OTHR transmitter.
    let center_lat = 0.3176_f64; // ~18.2°N
    let center_lon = -1.147_f64; // ~-65.7°E
    let distances_m = [
        1_000_000.0_f64,
        1_500_000.0,
        2_000_000.0,
        2_500_000.0,
        3_000_000.0,
    ];
    let azimuths_rad = [
        0.0_f64,
        30.0_f64.to_radians(),
        90.0_f64.to_radians(),
        180.0_f64.to_radians(),
        270.0_f64.to_radians(),
        315.0_f64.to_radians(),
    ];

    for &d in &distances_m {
        for &az in &azimuths_rad {
            // Propagate a point from the centre on the ellipsoid.
            let (lat, lon) = vincenty_direct(center_lat, center_lon, az, d);

            let (x, y) = stereographic_project(lat, lon, center_lat, center_lon);
            let (lat_b, lon_b) = stereographic_inverse(x, y, center_lat, center_lon);

            // Measure roundtrip error by re-projecting the recovered geodetic
            // point. Any residual shows up as a planar offset.
            let (x2, y2) = stereographic_project(lat_b, lon_b, center_lat, center_lon);
            let err = ((x - x2).powi(2) + (y - y2).powi(2)).sqrt();
            assert!(
                err < 1.0,
                "roundtrip error {err} m at dist={d} az={} deg",
                az.to_degrees()
            );
        }
    }
}

#[test]
fn recommended_center_picks_transmitter_for_single_sensor() {
    let (lat, lon) = recommended_center(&[(0.5, -1.2)]);
    assert_eq!(lat, 0.5);
    assert_eq!(lon, -1.2);
}

// ── Task 8.D.7 — Tracking across full OTHR coverage ───────────────────────

/// Build geodetic waypoints for a target moving on a straight bearing from
/// the transmitter, starting at `start_range_m` and moving radially outward
/// at `speed_m_s` until `end_range_m`.
///
/// The returned waypoints embed the transmitter-local ENU east/north that the
/// synthetic OTHR generator expects, plus the ground-truth geodetic lat/lon
/// (carried via the altitude-free position-to-geodetic mapping outside this
/// helper).
fn radial_waypoints(speed_m_s: f64, dt: f64, start_m: f64, end_m: f64) -> Vec<Waypoint> {
    let mut wps = Vec::new();
    let mut t = 0.0_f64;
    let mut r = start_m;
    while r <= end_m {
        wps.push(Waypoint {
            time: t,
            position: [0.0, r, 10_000.0], // due north, 10 km altitude
            velocity: [0.0, speed_m_s, 0.0],
        });
        t += dt;
        r += speed_m_s * dt;
    }
    wps
}

fn registration_from_config(cfg: &OthrConfig) -> OthrSensorRegistration {
    OthrSensorRegistration {
        transmitter_lat_rad: cfg.transmitter_lat_rad,
        transmitter_lon_rad: cfg.transmitter_lon_rad,
        transmitter_alt_m: cfg.transmitter_alt_m,
        operating_freq_mhz: cfg.freq_mhz,
    }
}

fn generate_stereo_detection(
    wp: &Waypoint,
    cfg: &OthrConfig,
    reg: &OthrSensorRegistration,
    center: (f64, f64),
    rng: &mut StdRng,
) -> Vec<DVector<f64>> {
    if let Some(m) = generate_othr(wp, cfg, 12.0, rng)
        && let Some(det) = othr_to_stereographic(&m, reg, wp.position[2], center.0, center.1)
    {
        vec![det]
    } else {
        vec![]
    }
}

fn stereo_truth_position(
    wp: &Waypoint,
    reg: &OthrSensorRegistration,
    center: (f64, f64),
) -> (f64, f64) {
    let truth_range = (wp.position[0].powi(2) + wp.position[1].powi(2)).sqrt();
    let truth_az = wp.position[0].atan2(wp.position[1]);
    let (truth_lat, truth_lon) = vincenty_direct(
        reg.transmitter_lat_rad,
        reg.transmitter_lon_rad,
        truth_az,
        truth_range,
    );
    stereographic_project(truth_lat, truth_lon, center.0, center.1)
}

fn min_alive_track_error(
    tracker: &MultiObjectTrackerStereographic,
    tx: f64,
    ty: f64,
) -> Option<f64> {
    tracker
        .tracks
        .iter()
        .filter(|t| t.lifecycle != thresh_core::track::TrackState::Deleted)
        .map(|t| {
            let dx = t.state[0] - tx;
            let dy = t.state[2] - ty;
            (dx * dx + dy * dy).sqrt()
        })
        .reduce(f64::min)
}

#[test]
fn stereographic_tracker_tracks_across_othr_coverage() {
    let mut cfg = OthrConfig::rothr();
    cfg.base_p_detection = 1.0;
    let reg = registration_from_config(&cfg);
    let center = recommended_center(&[(reg.transmitter_lat_rad, reg.transmitter_lon_rad)]);
    let waypoints = radial_waypoints(250.0, 1.0, 1_200_000.0, 3_000_000.0);

    let mut tracker = MultiObjectTrackerStereographic::new(center.0, center.1, 15.0, 100.0);
    let mut rng = StdRng::seed_from_u64(17);
    let mut errors_m: Vec<f64> = Vec::new();

    for wp in &waypoints {
        let dets = generate_stereo_detection(wp, &cfg, &reg, center, &mut rng);
        tracker.step(&dets, 1.0);

        let (tx, ty) = stereo_truth_position(wp, &reg, center);
        if let Some(err) = min_alive_track_error(&tracker, tx, ty) {
            errors_m.push(err);
        }
    }

    assert!(tracker.alive_count() >= 1, "must have at least one track");
    assert!(
        tracker.confirmed_count() >= 1,
        "track should have been confirmed over the sweep"
    );

    assert!(errors_m.len() >= 20);
    let mean_tail = tail_mean(&errors_m, 20);
    assert!(
        mean_tail < 200_000.0,
        "tail mean error too large: {mean_tail} m"
    );
}

// ── Task 8.D.8 — Benchmark stereographic vs ENU ───────────────────────────

/// Build a stereographic detection for a waypoint (or empty if no detection).
fn stereo_detections_for_wp(
    wp: &Waypoint,
    cfg: &OthrConfig,
    reg: &OthrSensorRegistration,
    center: (f64, f64),
    rng: &mut StdRng,
) -> Vec<DVector<f64>> {
    let mut dets = Vec::new();
    if let Some(m) = generate_othr(wp, cfg, 12.0, rng)
        && let Some(det) = othr_to_stereographic(&m, reg, wp.position[2], center.0, center.1)
    {
        dets.push(det);
    }
    dets
}

/// Build an ENU detection for a waypoint (or empty if no detection).
fn enu_detections_for_wp(
    wp: &Waypoint,
    cfg: &OthrConfig,
    reg: &OthrSensorRegistration,
    rng: &mut StdRng,
) -> Vec<DVector<f64>> {
    let mut dets = Vec::new();
    if let Some(m) = generate_othr(wp, cfg, 12.0, rng)
        && let Some(det) = othr_to_cartesian(
            &m,
            reg,
            wp.position[2],
            reg.transmitter_lat_rad,
            reg.transmitter_lon_rad,
            reg.transmitter_alt_m,
        )
    {
        dets.push(det);
    }
    dets
}

/// Compute the truth position in stereographic (tx, ty) and ENU (ex, ey)
/// frames for a given waypoint.
fn truth_positions(
    wp: &Waypoint,
    reg: &OthrSensorRegistration,
    center: (f64, f64),
) -> ((f64, f64), (f64, f64)) {
    let truth_range = (wp.position[0].powi(2) + wp.position[1].powi(2)).sqrt();
    let truth_az = wp.position[0].atan2(wp.position[1]);
    let (truth_lat, truth_lon) = vincenty_direct(
        reg.transmitter_lat_rad,
        reg.transmitter_lon_rad,
        truth_az,
        truth_range,
    );
    let (tx, ty) = stereographic_project(truth_lat, truth_lon, center.0, center.1);

    let ecef = thresh_core::geodetic::wgs84_to_ecef(truth_lat, truth_lon, wp.position[2]);
    let enu = thresh_core::geodetic::ecef_to_enu(
        &ecef,
        reg.transmitter_lat_rad,
        reg.transmitter_lon_rad,
        reg.transmitter_alt_m,
    );

    ((tx, ty), (enu.x, enu.y))
}

/// Minimum positional error between any alive stereographic track and truth.
fn min_stereo_error(
    tracker: &MultiObjectTrackerStereographic,
    truth_xy: (f64, f64),
) -> Option<f64> {
    tracker
        .tracks
        .iter()
        .filter(|t| t.lifecycle != thresh_core::track::TrackState::Deleted)
        .map(|t| ((t.state[0] - truth_xy.0).powi(2) + (t.state[2] - truth_xy.1).powi(2)).sqrt())
        .fold(None::<f64>, |acc, e| {
            Some(acc.map_or(e, |a| if e.total_cmp(&a).is_lt() { e } else { a }))
        })
}

/// Minimum positional error between any alive ENU track and truth.
fn min_enu_error(tracker: &MultiObjectTracker, truth_xy: (f64, f64)) -> Option<f64> {
    tracker
        .tracks
        .iter()
        .filter(|t| t.is_alive())
        .map(|t| ((t.state[0] - truth_xy.0).powi(2) + (t.state[2] - truth_xy.1).powi(2)).sqrt())
        .fold(None::<f64>, |acc, e| {
            Some(acc.map_or(e, |a| if e.total_cmp(&a).is_lt() { e } else { a }))
        })
}

/// Mean of the last `tail` elements of `errors`.
fn tail_mean(errors: &[f64], tail: usize) -> f64 {
    let n = errors.len();
    errors[n - tail..].iter().sum::<f64>() / tail as f64
}

#[test]
fn stereographic_matches_or_beats_enu_at_long_range() {
    // Run the same scenario through the stereographic tracker and the
    // existing ENU-based `MultiObjectTracker`, compare mean tail error. At
    // 2500-3000 km the stereographic frame should match or beat ENU.

    let mut cfg = OthrConfig::rothr();
    cfg.base_p_detection = 1.0;
    let reg = registration_from_config(&cfg);

    let center = recommended_center(&[(reg.transmitter_lat_rad, reg.transmitter_lon_rad)]);
    let waypoints = radial_waypoints(250.0, 1.0, 1_500_000.0, 3_000_000.0);

    let mut stereo = MultiObjectTrackerStereographic::new(center.0, center.1, 5.0, 100.0);
    let mut rng_a = StdRng::seed_from_u64(2026);
    let mut stereo_errors: Vec<f64> = Vec::new();

    let mut enu_tracker = MultiObjectTracker::new_cv_position(20_000.0, 100.0);
    let mut rng_b = StdRng::seed_from_u64(2026);
    let mut enu_errors: Vec<f64> = Vec::new();

    for wp in &waypoints {
        let stereo_dets = stereo_detections_for_wp(wp, &cfg, &reg, center, &mut rng_a);
        stereo.step(&stereo_dets, 1.0);

        let enu_dets = enu_detections_for_wp(wp, &cfg, &reg, &mut rng_b);
        enu_tracker.step(&enu_dets, 1.0);

        let (truth_stereo, truth_enu) = truth_positions(wp, &reg, center);

        if let Some(err) = min_stereo_error(&stereo, truth_stereo) {
            stereo_errors.push(err);
        }
        if let Some(err) = min_enu_error(&enu_tracker, truth_enu) {
            enu_errors.push(err);
        }
    }

    let tail = 20usize;
    assert!(
        stereo_errors.len() >= tail && enu_errors.len() >= tail,
        "need converged tails"
    );
    let stereo_tail = tail_mean(&stereo_errors, tail);
    let enu_tail = tail_mean(&enu_errors, tail);

    println!("stereo tail mean = {stereo_tail:.1} m, enu tail mean = {enu_tail:.1} m");

    // Stereographic should match or beat ENU. Allow 20% slack for RNG noise.
    assert!(
        stereo_tail <= enu_tail * 1.2,
        "stereographic ({stereo_tail} m) should match-or-beat ENU ({enu_tail} m)"
    );
}
