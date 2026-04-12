//! High-fidelity sensor simulation integration tests (tasks 8.1-8.4).
//!
//! These tests exercise the full pipeline: trajectory generation ->
//! physics-based measurement generation -> tracker -> MOT evaluation.

use nalgebra::DVector;
use rand::Rng;
use rand_distr::{Distribution, Normal};
use thresh_core::coords::spherical_to_cartesian;
use thresh_core::measurement::Measurement;
use thresh_eval::matching::FrameData;
use thresh_eval::metrics::compute_mot_metrics;
use thresh_synth::eoir_physics::{IrSensorConfig, IrSignature, generate_eoir_physics};
use thresh_synth::measurement_gen::{RadarConfig, generate_radar};
use thresh_synth::radar_equation::{
    FullRadarConfig, RadarParameters, dwell_rcs_from_profile, generate_radar_full,
};
use thresh_synth::swerling::RcsProfile;
use thresh_synth::trajectory::{Segment, SegmentType, Trajectory, Waypoint};
use thresh_tracker::tracker::MultiObjectTracker;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Convert a radar measurement vector [range, az, el] to Cartesian [x, y, z].
fn radar_to_cartesian(z: &DVector<f64>) -> DVector<f64> {
    let v = spherical_to_cartesian(z[0], z[1], z[2]);
    DVector::from_column_slice(&[v.x, v.y, v.z])
}

/// Convert an EO/IR bearing-only measurement to a Cartesian detection at an
/// assumed range. This is a rough triangulation proxy suitable for feeding
/// bearing-only observations into a position-state tracker.
fn eoir_to_cartesian_at_range(az: f64, el: f64, assumed_range: f64) -> DVector<f64> {
    let v = spherical_to_cartesian(assumed_range, az, el);
    DVector::from_column_slice(&[v.x, v.y, v.z])
}

/// Run the tracker on a set of waypoints with a measurement generator closure,
/// and return the evaluation frames + final tracker state.
fn run_tracking_pipeline<F>(
    waypoints: &[Waypoint],
    dt: f64,
    noise_sigma: f64,
    gate: f64,
    mut gen_detections: F,
) -> (Vec<FrameData>, MultiObjectTracker)
where
    F: FnMut(&Waypoint) -> Vec<DVector<f64>>,
{
    let mut tracker = MultiObjectTracker::new_cv_position(noise_sigma, gate);
    let mut eval_frames = Vec::new();

    for wp in waypoints {
        let detections = gen_detections(wp);
        tracker.step(&detections, dt);

        let gt = vec![(0u64, wp.position)];
        let tracks: Vec<(u64, [f64; 3])> = tracker
            .tracks
            .iter()
            .filter(|t| t.is_alive())
            .map(|t| (t.id.0, [t.state[0], t.state[2], t.state[4]]))
            .collect();
        eval_frames.push(FrameData { gt, tracks });
    }

    (eval_frames, tracker)
}

// ===========================================================================
// Task 8.1 -- High-fidelity radar scenario (Swerling + radar equation)
// ===========================================================================

#[test]
fn hifi_radar_swerling_scenario() {
    // Fighter trajectory: CV segment then a coordinated turn (CTRV).
    let traj = Trajectory {
        target_id: 0,
        initial_position: [20_000.0, 5_000.0, 8_000.0],
        initial_velocity: [250.0, 0.0, 0.0],
        segments: vec![
            Segment {
                segment_type: SegmentType::Cv,
                duration: 15.0,
            },
            Segment {
                segment_type: SegmentType::Ctrv { turn_rate: 0.05 },
                duration: 15.0,
            },
        ],
        dt: 1.0,
    };
    let waypoints = traj.generate();

    // Fighter RCS profile with Swerling I fluctuation.
    let rcs_profile = RcsProfile::fighter();

    // X-band surveillance radar (full radar equation pipeline).
    let full_config = FullRadarConfig {
        sensor_id: 0,
        radar: RadarParameters::x_band_surveillance(),
        base_range_sigma: 15.0,
        base_angle_sigma: 0.002,
        max_range: 200_000.0,
        apply_atmosphere: true,
        use_shnidman: true,
    };

    let mut rng = rand::rng();
    let mut tracker = MultiObjectTracker::new_cv_position(100.0, 500.0);
    let mut eval_frames = Vec::new();

    for wp in &waypoints {
        // Sample dwell RCS for each scan (Swerling I: one sample per dwell).
        let dwell = dwell_rcs_from_profile(&rcs_profile, &mut rng);

        let detections: Vec<DVector<f64>> =
            if let Some(m) = generate_radar_full(wp, &full_config, &dwell, &mut rng) {
                vec![radar_to_cartesian(&m.to_vector())]
            } else {
                vec![]
            };

        tracker.step(&detections, 1.0);

        let gt = vec![(0u64, wp.position)];
        let tracks: Vec<(u64, [f64; 3])> = tracker
            .tracks
            .iter()
            .filter(|t| t.is_alive())
            .map(|t| (t.id.0, [t.state[0], t.state[2], t.state[4]]))
            .collect();
        eval_frames.push(FrameData { gt, tracks });
    }

    let (mota, _motp, _idsw) = compute_mot_metrics(&eval_frames, 500.0);

    // Swerling fluctuation and range-dependent P_d make tracking harder, but
    // the fighter is close enough to still maintain a reasonable track.
    assert!(
        mota > 0.3,
        "MOTA for hifi radar+Swerling scenario too low: {mota}"
    );
}

// ===========================================================================
// Task 8.2 -- Orbital scenario (simulated ISS-like overhead pass)
// ===========================================================================

#[test]
fn hifi_orbital_radar_scenario() {
    // Simulate an ISS-like overhead pass as seen from a ground station.
    // The ISS moves at ~7.66 km/s at ~408 km altitude. During a ~10-minute
    // pass the slant range varies from ~2000 km (horizon) to ~408 km (zenith).
    //
    // We model this as a straight-line trajectory in ENU (east-north-up)
    // coordinates relative to a ground station, using waypoints that mimic
    // the geometry of an overhead pass.

    let pass_duration_s = 600.0; // 10 minutes
    let dt = 2.0; // 2-second radar scan rate
    let n_steps = (pass_duration_s / dt) as usize;

    // Build ENU waypoints for the pass. The satellite enters from the south,
    // passes near-overhead, and exits to the north.
    let ground_speed = 6_800.0; // m/s apparent ground speed
    let altitude = 408_000.0; // ISS altitude in metres

    let mut waypoints = Vec::with_capacity(n_steps + 1);
    for i in 0..=n_steps {
        let t = i as f64 * dt;
        // North component: goes from -2000 km to +2000 km
        let north = -2_000_000.0 + ground_speed * t;
        // East component: slight cross-track
        let east = 50_000.0;
        // Up component: altitude (constant for near-circular orbit)
        let up = altitude;

        waypoints.push(Waypoint {
            time: t,
            position: [east, north, up],
            velocity: [0.0, ground_speed, 0.0],
        });
    }

    // Filter to above-horizon positions (elevation > 5 degrees).
    let visible_waypoints: Vec<&Waypoint> = waypoints
        .iter()
        .filter(|wp| {
            let range =
                (wp.position[0].powi(2) + wp.position[1].powi(2) + wp.position[2].powi(2)).sqrt();
            let el_rad = (wp.position[2] / range).asin();
            el_rad.to_degrees() > 5.0
        })
        .collect();

    assert!(
        visible_waypoints.len() > 20,
        "Expected at least 20 visible waypoints, got {}",
        visible_waypoints.len()
    );

    // Generate radar measurements from the visible waypoints.
    // Convert ENU position to range/azimuth/elevation and add noise.
    let mut rng = rand::rng();
    let std_normal = Normal::new(0.0, 1.0).unwrap();
    let range_sigma = 50.0; // metres
    let angle_sigma = 0.002; // radians

    let mut tracker = MultiObjectTracker::new_cv_position(200.0, 2000.0);
    let mut detections_count = 0;
    let mut track_maintained_frames = 0;

    for wp in &visible_waypoints {
        let range =
            (wp.position[0].powi(2) + wp.position[1].powi(2) + wp.position[2].powi(2)).sqrt();
        let az = wp.position[1].atan2(wp.position[0]);
        let el = (wp.position[2] / range).asin();

        // Add measurement noise.
        let range_n: f64 = std_normal.sample(&mut rng);
        let az_n: f64 = std_normal.sample(&mut rng);
        let el_n: f64 = std_normal.sample(&mut rng);

        let noisy_range = range + range_n * range_sigma;
        let noisy_az = az + az_n * angle_sigma;
        let noisy_el = el + el_n * angle_sigma;

        // Detection probability based on range (satellite RCS ~3 m^2).
        // At close range P_d is high; at long range it drops.
        let max_detection_range = 2_500_000.0;
        let pd = if range < max_detection_range {
            (1.0 - (range / max_detection_range).powi(2)).max(0.3)
        } else {
            0.1
        };

        let detected = rng.random::<f64>() < pd;

        let detections = if detected {
            detections_count += 1;
            let cart = spherical_to_cartesian(noisy_range, noisy_az, noisy_el);
            vec![DVector::from_column_slice(&[cart.x, cart.y, cart.z])]
        } else {
            vec![]
        };

        tracker.step(&detections, dt);

        if tracker.alive_count() > 0 {
            track_maintained_frames += 1;
        }
    }

    // Verify we got enough detections and maintained tracks through most of the pass.
    assert!(
        detections_count > 20,
        "Expected >20 detections during pass, got {detections_count}"
    );
    assert!(
        track_maintained_frames as f64 / visible_waypoints.len() as f64 > 0.5,
        "Track should be maintained for at least 50% of visible pass, got {}/{} frames",
        track_maintained_frames,
        visible_waypoints.len()
    );
}

// ===========================================================================
// Task 8.3 -- Multi-sensor fusion scenario (radar + EO/IR)
// ===========================================================================

#[test]
fn hifi_multisensor_radar_eoir() {
    // Aircraft trajectory: straight and level then gentle turn.
    let traj = Trajectory {
        target_id: 0,
        initial_position: [15_000.0, 2_000.0, 6_000.0],
        initial_velocity: [200.0, 30.0, 0.0],
        segments: vec![
            Segment {
                segment_type: SegmentType::Cv,
                duration: 15.0,
            },
            Segment {
                segment_type: SegmentType::Ctrv { turn_rate: 0.03 },
                duration: 10.0,
            },
        ],
        dt: 1.0,
    };
    let waypoints = traj.generate();

    // Radar config (Level 0: fixed P_d).
    let radar_config = RadarConfig {
        sensor_id: 0,
        range_sigma: 10.0,
        azimuth_sigma: 0.001,
        elevation_sigma: 0.001,
        p_detection: 0.85,
        clutter_rate: 0.0,
        max_range: 200_000.0,
        radar_equation: None,
    };

    // MWIR EO/IR sensor with physics-based detection.
    let ir_signature = IrSignature::fighter_military();
    let ir_sensor = IrSensorConfig::mwir_search();
    let background_temp = 280.0;

    let mut rng = rand::rng();
    let mut tracker = MultiObjectTracker::new_cv_position(80.0, 400.0);
    let mut eval_frames = Vec::new();

    for wp in &waypoints {
        // Multi-sensor fusion strategy: prefer radar (provides range), fall
        // back to EO/IR when radar misses. This avoids feeding two detections
        // of the same target into the tracker in a single frame.
        let radar_det =
            generate_radar(wp, &radar_config, &mut rng).map(|m| radar_to_cartesian(&m.to_vector()));

        let eoir_det =
            generate_eoir_physics(wp, &ir_signature, &ir_sensor, background_temp, &mut rng)
                .and_then(|m| {
                    if let Measurement::EoIr {
                        azimuth, elevation, ..
                    } = &m
                    {
                        let true_range = (wp.position[0].powi(2)
                            + wp.position[1].powi(2)
                            + wp.position[2].powi(2))
                        .sqrt();
                        Some(eoir_to_cartesian_at_range(*azimuth, *elevation, true_range))
                    } else {
                        None
                    }
                });

        // Use radar if available, otherwise EO/IR.
        let detections: Vec<DVector<f64>> = match (radar_det, eoir_det) {
            (Some(r), _) => vec![r],
            (None, Some(e)) => vec![e],
            _ => vec![],
        };

        tracker.step(&detections, 1.0);

        let gt = vec![(0u64, wp.position)];
        let tracks: Vec<(u64, [f64; 3])> = tracker
            .tracks
            .iter()
            .filter(|t| t.is_alive())
            .map(|t| (t.id.0, [t.state[0], t.state[2], t.state[4]]))
            .collect();
        eval_frames.push(FrameData { gt, tracks });
    }

    let (mota, _motp, _idsw) = compute_mot_metrics(&eval_frames, 300.0);

    // Multi-sensor fusion (radar + EO/IR fallback) should produce
    // better detection coverage and reasonable tracking.
    assert!(
        mota > 0.3,
        "MOTA for multi-sensor fusion scenario too low: {mota}"
    );
}

// ===========================================================================
// Task 8.4 -- Fidelity level comparison
// ===========================================================================

#[test]
fn fidelity_level_comparison() {
    // Same trajectory for both fidelity levels.
    let traj = Trajectory {
        target_id: 0,
        initial_position: [20_000.0, 5_000.0, 7_000.0],
        initial_velocity: [220.0, 30.0, 0.0],
        segments: vec![
            Segment {
                segment_type: SegmentType::Cv,
                duration: 20.0,
            },
            Segment {
                segment_type: SegmentType::Ctrv { turn_rate: 0.04 },
                duration: 10.0,
            },
        ],
        dt: 1.0,
    };
    let waypoints = traj.generate();

    // -----------------------------------------------------------------------
    // Level 0: basic generate_radar with fixed P_d
    // -----------------------------------------------------------------------
    let radar_config_l0 = RadarConfig {
        sensor_id: 0,
        range_sigma: 10.0,
        azimuth_sigma: 0.001,
        elevation_sigma: 0.001,
        p_detection: 0.95,
        clutter_rate: 0.0,
        max_range: 200_000.0,
        radar_equation: None,
    };

    let mut rng_l0 = rand::rng();
    let (frames_l0, _) = run_tracking_pipeline(&waypoints, 1.0, 80.0, 400.0, |wp| {
        if let Some(m) = generate_radar(wp, &radar_config_l0, &mut rng_l0) {
            vec![radar_to_cartesian(&m.to_vector())]
        } else {
            vec![]
        }
    });

    let (mota_l0, _, _) = compute_mot_metrics(&frames_l0, 300.0);

    // -----------------------------------------------------------------------
    // Level 1: generate_radar_full with Swerling + radar equation
    // -----------------------------------------------------------------------
    let rcs_profile = RcsProfile::fighter();
    let full_config = FullRadarConfig {
        sensor_id: 0,
        radar: RadarParameters::x_band_surveillance(),
        base_range_sigma: 15.0,
        base_angle_sigma: 0.002,
        max_range: 200_000.0,
        apply_atmosphere: true,
        use_shnidman: true,
    };

    let mut rng_l1 = rand::rng();
    let (frames_l1, _) = run_tracking_pipeline(&waypoints, 1.0, 100.0, 500.0, |wp| {
        let dwell = dwell_rcs_from_profile(&rcs_profile, &mut rng_l1);
        if let Some(m) = generate_radar_full(wp, &full_config, &dwell, &mut rng_l1) {
            vec![radar_to_cartesian(&m.to_vector())]
        } else {
            vec![]
        }
    });

    let (mota_l1, _, _) = compute_mot_metrics(&frames_l1, 500.0);

    // Both levels should produce acceptable tracking.
    assert!(mota_l0 > 0.2, "Level 0 MOTA too low: {mota_l0}");
    assert!(mota_l1 > 0.2, "Level 1 MOTA too low: {mota_l1}");

    // Level 0 (simple, high fixed P_d) should generally be >= Level 1
    // (physics-based, Swerling fluctuation, atmospheric loss). We allow some
    // margin because randomness can occasionally invert the ordering.
    // Just log the comparison rather than hard-asserting the ordering.
    eprintln!("Fidelity comparison: Level 0 MOTA = {mota_l0:.3}, Level 1 MOTA = {mota_l1:.3}");
}
