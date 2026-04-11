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

#[test]
fn stereographic_tracker_tracks_across_othr_coverage() {
    let mut cfg = OthrConfig::rothr();
    cfg.base_p_detection = 1.0;
    let reg = registration_from_config(&cfg);

    let center = recommended_center(&[(reg.transmitter_lat_rad, reg.transmitter_lon_rad)]);

    // Target starts at 1200 km and moves to ~3000 km — full coverage sweep.
    let waypoints = radial_waypoints(250.0, 1.0, 1_200_000.0, 3_000_000.0);

    let mut tracker = MultiObjectTrackerStereographic::new(center.0, center.1, 15.0, 100.0);
    let mut rng = StdRng::seed_from_u64(17);

    let mut errors_m: Vec<f64> = Vec::new();

    for wp in &waypoints {
        // Generate synthetic OTHR measurement in the transmitter-local ENU.
        let mut dets: Vec<DVector<f64>> = Vec::new();
        if let Some(m) = generate_othr(wp, &cfg, 12.0, &mut rng)
            && let Some(det) = othr_to_stereographic(&m, &reg, wp.position[2], center.0, center.1)
        {
            dets.push(det);
        }
        tracker.step(&dets, 1.0);

        // Evaluate the best track against the projected ground truth.
        // Ground truth in the stereographic plane: propagate the ideal
        // waypoint through Vincenty from the transmitter and project.
        let truth_range = (wp.position[0].powi(2) + wp.position[1].powi(2)).sqrt();
        let truth_az = wp.position[0].atan2(wp.position[1]);
        let (truth_lat, truth_lon) = vincenty_direct(
            reg.transmitter_lat_rad,
            reg.transmitter_lon_rad,
            truth_az,
            truth_range,
        );
        let (tx, ty) = stereographic_project(truth_lat, truth_lon, center.0, center.1);

        if let Some(err) = tracker
            .tracks
            .iter()
            .filter(|t| t.lifecycle != thresh_core::track::TrackState::Deleted)
            .map(|t| {
                let dx = t.state[0] - tx;
                let dy = t.state[2] - ty;
                (dx * dx + dy * dy).sqrt()
            })
            .fold(None::<f64>, |acc, e| {
                Some(acc.map_or(e, |a| if e.total_cmp(&a).is_lt() { e } else { a }))
            })
        {
            errors_m.push(err);
        }
    }

    assert!(tracker.alive_count() >= 1, "must have at least one track");
    assert!(
        tracker.confirmed_count() >= 1,
        "track should have been confirmed over the sweep"
    );

    // Tail mean should be within a few OTHR range sigmas.
    assert!(errors_m.len() >= 20);
    let n = errors_m.len();
    let tail = &errors_m[n - 20..];
    let mean_tail: f64 = tail.iter().sum::<f64>() / tail.len() as f64;
    assert!(
        mean_tail < 200_000.0,
        "tail mean error too large: {mean_tail} m"
    );
}

// ── Task 8.D.8 — Benchmark stereographic vs ENU ───────────────────────────

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

    // ── Stereographic run ─────────────────────────────────────────────
    let mut stereo = MultiObjectTrackerStereographic::new(center.0, center.1, 5.0, 100.0);
    let mut rng_a = StdRng::seed_from_u64(2026);
    let mut stereo_errors: Vec<f64> = Vec::new();

    // ── ENU run (existing tracker from tracker.rs) ────────────────────
    let mut enu_tracker = MultiObjectTracker::new_cv_position(20_000.0, 100.0);
    let mut rng_b = StdRng::seed_from_u64(2026);
    let mut enu_errors: Vec<f64> = Vec::new();

    for wp in &waypoints {
        // Stereographic detection
        let mut stereo_dets: Vec<DVector<f64>> = Vec::new();
        if let Some(m) = generate_othr(wp, &cfg, 12.0, &mut rng_a)
            && let Some(det) = othr_to_stereographic(&m, &reg, wp.position[2], center.0, center.1)
        {
            stereo_dets.push(det);
        }
        stereo.step(&stereo_dets, 1.0);

        let truth_range = (wp.position[0].powi(2) + wp.position[1].powi(2)).sqrt();
        let truth_az = wp.position[0].atan2(wp.position[1]);
        let (truth_lat, truth_lon) = vincenty_direct(
            reg.transmitter_lat_rad,
            reg.transmitter_lon_rad,
            truth_az,
            truth_range,
        );
        let (tx, ty) = stereographic_project(truth_lat, truth_lon, center.0, center.1);

        if let Some(err) = stereo
            .tracks
            .iter()
            .filter(|t| t.lifecycle != thresh_core::track::TrackState::Deleted)
            .map(|t| ((t.state[0] - tx).powi(2) + (t.state[2] - ty).powi(2)).sqrt())
            .fold(None::<f64>, |acc, e| {
                Some(acc.map_or(e, |a| if e.total_cmp(&a).is_lt() { e } else { a }))
            })
        {
            stereo_errors.push(err);
        }

        // ENU detection (cartesian around the transmitter)
        let mut enu_dets: Vec<DVector<f64>> = Vec::new();
        if let Some(m) = generate_othr(wp, &cfg, 12.0, &mut rng_b)
            && let Some(det) = othr_to_cartesian(
                &m,
                &reg,
                wp.position[2],
                reg.transmitter_lat_rad,
                reg.transmitter_lon_rad,
                reg.transmitter_alt_m,
            )
        {
            enu_dets.push(det);
        }
        enu_tracker.step(&enu_dets, 1.0);

        // ENU truth: Vincenty-propagate then convert via ECEF/ENU.
        let ecef = thresh_core::geodetic::wgs84_to_ecef(truth_lat, truth_lon, wp.position[2]);
        let enu = thresh_core::geodetic::ecef_to_enu(
            &ecef,
            reg.transmitter_lat_rad,
            reg.transmitter_lon_rad,
            reg.transmitter_alt_m,
        );
        if let Some(err) = enu_tracker
            .tracks
            .iter()
            .filter(|t| t.is_alive())
            .map(|t| ((t.state[0] - enu.x).powi(2) + (t.state[2] - enu.y).powi(2)).sqrt())
            .fold(None::<f64>, |acc, e| {
                Some(acc.map_or(e, |a| if e.total_cmp(&a).is_lt() { e } else { a }))
            })
        {
            enu_errors.push(err);
        }
    }

    let tail = 20usize;
    let ns = stereo_errors.len();
    let ne = enu_errors.len();
    assert!(ns >= tail && ne >= tail, "need converged tails");
    let stereo_tail: f64 = stereo_errors[ns - tail..].iter().sum::<f64>() / tail as f64;
    let enu_tail: f64 = enu_errors[ne - tail..].iter().sum::<f64>() / tail as f64;

    println!("stereo tail mean = {stereo_tail:.1} m, enu tail mean = {enu_tail:.1} m");

    // Stereographic should match or beat ENU. Allow 20% slack for RNG noise.
    assert!(
        stereo_tail <= enu_tail * 1.2,
        "stereographic ({stereo_tail} m) should match-or-beat ENU ({enu_tail} m)"
    );
}
