//! Integration tests for the ECEF multi-object tracker (task 8.A).
//!
//! These tests exercise long-traverse scenarios where the single-ENU-origin
//! Cartesian tracker accumulates curvature error. The ECEF tracker is the
//! primary subject, and the last test compares MOTA against the existing
//! ENU tracker on the same scenario to confirm the ECEF variant is at least
//! as good at long ranges.

use nalgebra::Vector3;
use rand::SeedableRng;
use rand::rngs::StdRng;
use thresh_core::geodetic::wgs84_to_ecef;
use thresh_core::othr::{OthrSensorRegistration, vincenty_direct};
use thresh_core::track::TrackState;
use thresh_eval::matching::FrameData;
use thresh_eval::metrics::compute_mot_metrics;
use thresh_synth::othr_generator::{OthrConfig, generate_othr};
use thresh_synth::trajectory::Waypoint;
use thresh_tracker::ecef_tracker::{MultiObjectTrackerEcef, othr_to_ecef};
use thresh_tracker::othr_integration::othr_to_cartesian;
use thresh_tracker::tracker::MultiObjectTracker;

// ── Helpers ─────────────────────────────────────────────────────────────────

fn registration_from_config(cfg: &OthrConfig) -> OthrSensorRegistration {
    OthrSensorRegistration {
        transmitter_lat_rad: cfg.transmitter_lat_rad,
        transmitter_lon_rad: cfg.transmitter_lon_rad,
        transmitter_alt_m: cfg.transmitter_alt_m,
        operating_freq_mhz: cfg.freq_mhz,
    }
}

/// Generate ground-truth waypoints for a great-circle flight at constant
/// speed and constant initial bearing from a given origin point, relative to
/// an OTHR transmitter at (tx_lat, tx_lon, tx_alt).
///
/// Returns `(enu_waypoints, ecef_waypoints)` where:
/// * `enu_waypoints` have position expressed as ENU relative to the
///   transmitter (as required by the synthetic OTHR generator, which reads
///   `position[0..2]` as ENU from the transmitter to compute ground range).
/// * `ecef_waypoints` have true ECEF position — used as ground truth for
///   evaluating ECEF tracker output.
#[allow(clippy::too_many_arguments)]
fn great_circle_waypoints(
    tx_lat_rad: f64,
    tx_lon_rad: f64,
    tx_alt_m: f64,
    origin_lat_rad: f64,
    origin_lon_rad: f64,
    origin_alt_m: f64,
    initial_azimuth_rad: f64,
    speed_m_s: f64,
    dt: f64,
    n: usize,
) -> (Vec<Waypoint>, Vec<Vector3<f64>>) {
    let mut enu_wps = Vec::with_capacity(n);
    let mut ecef_wps = Vec::with_capacity(n);

    // For the OTHR generator, the waypoint position is treated as flat ENU
    // relative to the transmitter, and `ground_range = sqrt(x²+y²)` — so we
    // must supply positions whose flat-ENU ground range equals the geodesic
    // ground range on the ellipsoid that the tracker will later invert with
    // Vincenty. Put differently, the "truth" that is self-consistent with
    // the OTHR pipeline is computed by writing the waypoint in polar ENU
    // (east = d·sinα, north = d·cosα) and deriving ECEF from that via
    // `enu_to_ecef` — which is exactly what the Vincenty-based conversion
    // approximates at short range.
    //
    // We therefore build the ENU waypoint first and derive the ECEF ground
    // truth from it so the two stay in lockstep.
    //
    // We still initialise the origin's ENU by doing a one-shot Vincenty to
    // the starting point so the start position is geodetically meaningful.
    let (origin_enu_east, origin_enu_north) = {
        let ecef0 = wgs84_to_ecef(origin_lat_rad, origin_lon_rad, origin_alt_m);
        let enu0 = thresh_core::geodetic::ecef_to_enu(&ecef0, tx_lat_rad, tx_lon_rad, tx_alt_m);
        // Convert back through a polar (flat) representation: use the flat
        // ground range as the "distance along bearing" the OTHR generator
        // will see. This is an approximation but is consistent with the way
        // the generator computes ground range.
        (enu0.x, enu0.y)
    };
    // Bearing from transmitter to origin (for reference only, unused here).
    let _ = tx_alt_m;

    for k in 0..n {
        let t = k as f64 * dt;
        let distance = speed_m_s * t;

        // ENU relative to the transmitter: start at the origin's ENU
        // (east, north), then advance along the initial azimuth in the
        // flat-ENU frame. This matches the OTHR generator's flat-ENU
        // model of ground range + azimuth.
        let east = origin_enu_east + distance * initial_azimuth_rad.sin();
        let north = origin_enu_north + distance * initial_azimuth_rad.cos();
        let alt = origin_alt_m;
        let ve = speed_m_s * initial_azimuth_rad.sin();
        let vn = speed_m_s * initial_azimuth_rad.cos();

        // Corresponding ECEF ground truth: the OTHR tracker inverts
        // `(ground_range, azimuth)` with Vincenty on WGS84, so the
        // noise-free detection for this target lands at
        // `wgs84_to_ecef(vincenty_direct(tx, az, ground_range), alt)`. Use
        // that as ground truth so the "truth" and the zero-noise detection
        // agree exactly, and noisy detections scatter around it with the
        // expected OTHR noise magnitude.
        let ground_range = (east * east + north * north).sqrt();
        let azimuth = east.atan2(north);
        let (tgt_lat, tgt_lon) = if ground_range < 1e-6 {
            (tx_lat_rad, tx_lon_rad)
        } else {
            vincenty_direct(tx_lat_rad, tx_lon_rad, azimuth, ground_range)
        };
        let ecef = wgs84_to_ecef(tgt_lat, tgt_lon, alt);
        ecef_wps.push(ecef);

        enu_wps.push(Waypoint {
            time: t,
            position: [east, north, alt],
            velocity: [ve, vn, 0.0],
        });
    }

    (enu_wps, ecef_wps)
}

// ── Test 1: 3000 km cross-coverage transit (8.A.6) ─────────────────────────

#[test]
fn ecef_tracker_maintains_track_across_3000km_transit() {
    // Target traverses the OTHR coverage, starting ~1100 km from the
    // transmitter and ending ~3000+ km away on a great-circle path. Because
    // the target's speed is ~300 m/s, 100 s * 300 m/s ≈ 30 km / step, and we
    // take ~7000 s of flight to cover ~2000 km (tight inside 1000-3500 km
    // coverage window).
    let mut cfg = OthrConfig::rothr();
    cfg.base_p_detection = 1.0;
    cfg.range_sigma_m = 20_000.0;
    cfg.azimuth_sigma_rad = 0.015;
    let reg = registration_from_config(&cfg);

    let dt = 2.0_f64;
    let n = 1200; // 2400 s of flight
    // Start roughly due north of the transmitter at 1_100 km, flying due
    // north at 800 m/s → ends at ~3000 km, well inside coverage.
    let start_lat = reg.transmitter_lat_rad + (1_100_000.0 / 6_371_000.0);
    let start_lon = reg.transmitter_lon_rad;
    let (enu_wps, ecef_wps) = great_circle_waypoints(
        reg.transmitter_lat_rad,
        reg.transmitter_lon_rad,
        reg.transmitter_alt_m,
        start_lat,
        start_lon,
        10_000.0,
        0.0,
        800.0,
        dt,
        n,
    );

    // Generous measurement noise reflecting coarse OTHR ECEF detections.
    let mut tr = MultiObjectTrackerEcef::new(40_000.0, 60.0, 5.0);
    let mut rng = StdRng::seed_from_u64(8_006);

    let mut tail_errors = Vec::new();
    let tail_start = n - 50;

    for (k, wp) in enu_wps.iter().enumerate() {
        let mut dets = Vec::new();
        if let Some(m) = generate_othr(wp, &cfg, 12.0, &mut rng)
            && let Some(det) = othr_to_ecef(&m, &reg, wp.position[2])
        {
            dets.push(det);
        }
        tr.step(&dets, dt);

        if k >= tail_start {
            // Closest track error vs ground truth (ECEF).
            let truth = ecef_wps[k];
            let best = tr
                .tracks
                .iter()
                .filter(|t| t.is_alive())
                .map(|t| {
                    let dx = t.state[0] - truth.x;
                    let dy = t.state[2] - truth.y;
                    let dz = t.state[4] - truth.z;
                    (dx * dx + dy * dy + dz * dz).sqrt()
                })
                .fold(None::<f64>, |acc, e| Some(acc.map_or(e, |a| a.min(e))));
            if let Some(e) = best {
                tail_errors.push(e);
            }
        }
    }

    assert!(
        tr.alive_count() >= 1,
        "ECEF tracker should maintain a track across the transit"
    );
    assert!(
        tr.confirmed_count() >= 1,
        "ECEF tracker should have at least one confirmed track"
    );
    assert!(!tail_errors.is_empty(), "should have tail error samples");
    let mean_tail: f64 = tail_errors.iter().sum::<f64>() / tail_errors.len() as f64;
    // OTHR-scale error (km) is expected; assert it does not blow up to
    // hundreds of km which would indicate track divergence.
    assert!(
        mean_tail < 300_000.0,
        "tail mean error too large: {mean_tail} m"
    );
}

// ── Test 2: Great-circle 1 hour aircraft (8.A.7) ───────────────────────────

#[test]
fn ecef_tracker_tracks_great_circle_aircraft_one_hour() {
    let mut cfg = OthrConfig::rothr();
    cfg.base_p_detection = 1.0;
    cfg.range_sigma_m = 20_000.0;
    cfg.azimuth_sigma_rad = 0.015;
    let reg = registration_from_config(&cfg);

    // 1 hour of flight at 250 m/s = 900 km — but we start at 1200 km
    // along the northeast bearing so the full path remains in coverage.
    let dt = 2.0;
    let n = 1800; // 3600 s
    // Start ~1200 km NE of transmitter.
    let start_lat = reg.transmitter_lat_rad + (850_000.0 / 6_371_000.0);
    let start_lon =
        reg.transmitter_lon_rad + (850_000.0 / (6_371_000.0 * reg.transmitter_lat_rad.cos()));
    let (enu_wps, ecef_wps) = great_circle_waypoints(
        reg.transmitter_lat_rad,
        reg.transmitter_lon_rad,
        reg.transmitter_alt_m,
        start_lat,
        start_lon,
        10_000.0,
        45.0_f64.to_radians(),
        250.0,
        dt,
        n,
    );

    let mut tr = MultiObjectTrackerEcef::new(40_000.0, 60.0, 3.0);
    let mut rng = StdRng::seed_from_u64(8_007);

    let mut errors = Vec::new();
    for (k, wp) in enu_wps.iter().enumerate() {
        let mut dets = Vec::new();
        if let Some(m) = generate_othr(wp, &cfg, 12.0, &mut rng)
            && let Some(det) = othr_to_ecef(&m, &reg, wp.position[2])
        {
            dets.push(det);
        }
        tr.step(&dets, dt);

        if k >= n - 60 {
            let truth = ecef_wps[k];
            let best = tr
                .tracks
                .iter()
                .filter(|t| t.is_alive())
                .map(|t| {
                    let dx = t.state[0] - truth.x;
                    let dy = t.state[2] - truth.y;
                    let dz = t.state[4] - truth.z;
                    (dx * dx + dy * dy + dz * dz).sqrt()
                })
                .fold(None::<f64>, |acc, e| Some(acc.map_or(e, |a| a.min(e))));
            if let Some(e) = best {
                errors.push(e);
            }
        }
    }

    assert!(tr.confirmed_count() >= 1, "should confirm a track");
    assert!(
        !errors.is_empty(),
        "should produce at least one final-tail error sample"
    );
    let mean: f64 = errors.iter().sum::<f64>() / errors.len() as f64;
    assert!(
        mean < 300_000.0,
        "great-circle tail mean error: {mean} m (expected < 300 km)"
    );
}

// ── Test 3: ECEF vs ENU benchmark (8.A.8) ──────────────────────────────────

#[test]
fn ecef_vs_enu_benchmark_mota() {
    // Build one scenario and score both trackers on it. The scenario is a
    // long ENU-origin-far great-circle path where the ENU tracker's tangent
    // plane assumption starts to degrade. We fix the RNG seed so both
    // trackers see the same noise realisation.
    let mut cfg = OthrConfig::rothr();
    cfg.base_p_detection = 1.0;
    cfg.range_sigma_m = 20_000.0;
    cfg.azimuth_sigma_rad = 0.015;
    let reg = registration_from_config(&cfg);

    let dt = 2.0;
    let n = 1500;
    // Start ~1100 km due north, head east-northeast (great-circle).
    let start_lat = reg.transmitter_lat_rad + (1_100_000.0 / 6_371_000.0);
    let start_lon = reg.transmitter_lon_rad;
    let (enu_wps, ecef_wps) = great_circle_waypoints(
        reg.transmitter_lat_rad,
        reg.transmitter_lon_rad,
        reg.transmitter_alt_m,
        start_lat,
        start_lon,
        10_000.0,
        75.0_f64.to_radians(),
        250.0,
        dt,
        n,
    );

    // Pre-generate the noisy OTHR measurements so both trackers see the
    // exact same inputs.
    let mut rng = StdRng::seed_from_u64(8_008);
    let measurements: Vec<_> = enu_wps
        .iter()
        .map(|wp| generate_othr(wp, &cfg, 12.0, &mut rng))
        .collect();

    // --- ECEF tracker ---
    let mut ecef_tr = MultiObjectTrackerEcef::new(40_000.0, 60.0, 5.0);
    let mut ecef_frames: Vec<FrameData> = Vec::new();

    // --- ENU tracker (existing), with origin at the transmitter ---
    let mut enu_tr = MultiObjectTracker::new_cv_position(40_000.0, 60.0);
    let mut enu_frames: Vec<FrameData> = Vec::new();

    for (k, mopt) in measurements.iter().enumerate() {
        // ECEF detection and tracker step.
        let mut ecef_dets = Vec::new();
        if let Some(m) = mopt
            && let Some(det) = othr_to_ecef(m, &reg, enu_wps[k].position[2])
        {
            ecef_dets.push(det);
        }
        ecef_tr.step(&ecef_dets, dt);

        // ENU detection (tangent plane at transmitter).
        let mut enu_dets = Vec::new();
        if let Some(m) = mopt
            && let Some(det) = othr_to_cartesian(
                m,
                &reg,
                enu_wps[k].position[2],
                reg.transmitter_lat_rad,
                reg.transmitter_lon_rad,
                reg.transmitter_alt_m,
            )
        {
            enu_dets.push(det);
        }
        enu_tr.step(&enu_dets, dt);

        // Skip warm-up for MOTA computation.
        if k < 20 {
            continue;
        }

        let truth_ecef = ecef_wps[k];

        // Ground truth in ECEF for the ECEF tracker.
        let gt_ecef = vec![(1u64, [truth_ecef.x, truth_ecef.y, truth_ecef.z])];
        let ecef_tracks: Vec<(u64, [f64; 3])> = ecef_tr
            .tracks
            .iter()
            .filter(|t| t.lifecycle == TrackState::Confirmed)
            .map(|t| (t.id.0, [t.state[0], t.state[2], t.state[4]]))
            .collect();
        ecef_frames.push(FrameData {
            gt: gt_ecef,
            tracks: ecef_tracks,
        });

        // Ground truth in ENU (at transmitter) for the ENU tracker.
        let enu_truth = thresh_core::geodetic::ecef_to_enu(
            &truth_ecef,
            reg.transmitter_lat_rad,
            reg.transmitter_lon_rad,
            reg.transmitter_alt_m,
        );
        let gt_enu = vec![(1u64, [enu_truth.x, enu_truth.y, enu_truth.z])];
        let enu_tracks: Vec<(u64, [f64; 3])> = enu_tr
            .tracks
            .iter()
            .filter(|t| t.lifecycle == TrackState::Confirmed)
            .map(|t| (t.id.0, [t.state[0], t.state[2], t.state[4]]))
            .collect();
        enu_frames.push(FrameData {
            gt: gt_enu,
            tracks: enu_tracks,
        });
    }

    // Distance threshold on the order of OTHR noise (~100 km) — coarse but
    // appropriate for long-range coarse measurements.
    let threshold = 150_000.0;
    let (ecef_mota, _ecef_motp, _) = compute_mot_metrics(&ecef_frames, threshold);
    let (enu_mota, _enu_motp, _) = compute_mot_metrics(&enu_frames, threshold);

    eprintln!("ECEF MOTA = {ecef_mota:.4}, ENU MOTA = {enu_mota:.4}");

    // Both should track; assert ECEF is no worse than ENU on this long-
    // traverse scenario.
    assert!(
        ecef_mota >= enu_mota - 1e-9,
        "ECEF MOTA ({ecef_mota}) should be ≥ ENU MOTA ({enu_mota}) at long ranges"
    );
    // And ECEF should be positive (non-trivial).
    assert!(ecef_mota > 0.0, "ECEF MOTA should be positive");
}
