//! Integration tests: end-to-end synth -> tracker -> eval pipeline.

use nalgebra::DVector;
use thresh_eval::matching::FrameData;
use thresh_eval::metrics::{compute_idf1, compute_mot_metrics};
use thresh_synth::measurement_gen::{RadarConfig, generate_radar};
use thresh_synth::trajectory::{Segment, SegmentType, Trajectory};
use thresh_tracker::tracker::MultiObjectTracker;

fn radar_to_cartesian(z: &DVector<f64>) -> DVector<f64> {
    let range = z[0];
    let az = z[1];
    let el = z[2];
    DVector::from_column_slice(&[
        range * el.cos() * az.cos(),
        range * el.cos() * az.sin(),
        range * el.sin(),
    ])
}

#[test]
fn end_to_end_single_target() {
    let traj = Trajectory {
        target_id: 0,
        initial_position: [1000.0, 2000.0, 5000.0],
        initial_velocity: [100.0, 50.0, 0.0],
        segments: vec![Segment {
            segment_type: SegmentType::Cv,
            duration: 20.0,
        }],
        dt: 1.0,
    };
    let waypoints = traj.generate();

    let radar_config = RadarConfig {
        p_detection: 1.0,
        range_sigma: 5.0,
        azimuth_sigma: 0.0005,
        elevation_sigma: 0.0005,
        ..Default::default()
    };
    let mut rng = rand::rng();
    let mut tracker = MultiObjectTracker::new_cv_position(50.0, 100.0);
    let mut eval_frames = Vec::new();

    for wp in &waypoints {
        if let Some(m) = generate_radar(wp, &radar_config, &mut rng) {
            let det = radar_to_cartesian(&m.to_vector());
            tracker.step(std::slice::from_ref(&det), 1.0);
        } else {
            tracker.step(&[], 1.0);
        }

        let gt = vec![(0u64, wp.position)];
        let tracks: Vec<(u64, [f64; 3])> = tracker
            .tracks
            .iter()
            .filter(|t| t.is_alive())
            .map(|t| (t.id.0, [t.state[0], t.state[2], t.state[4]]))
            .collect();
        eval_frames.push(FrameData { gt, tracks });
    }

    let (mota, _motp, idsw) = compute_mot_metrics(&eval_frames, 200.0);
    let idf1 = compute_idf1(&eval_frames, 200.0);

    assert!(mota > 0.5, "MOTA too low: {mota}");
    assert!(idf1 > 0.5, "IDF1 too low: {idf1}");
    assert!(idsw <= 2, "Too many ID switches: {idsw}");
}

#[test]
fn multi_target_tracking() {
    let trajectories: Vec<Trajectory> = (0..5)
        .map(|i| Trajectory {
            target_id: i,
            initial_position: [i as f64 * 5000.0, 0.0, 5000.0],
            initial_velocity: [200.0, 0.0, 0.0],
            segments: vec![Segment {
                segment_type: SegmentType::Cv,
                duration: 10.0,
            }],
            dt: 1.0,
        })
        .collect();

    let radar = RadarConfig {
        p_detection: 1.0,
        range_sigma: 5.0,
        ..Default::default()
    };
    let mut rng = rand::rng();
    let mut tracker = MultiObjectTracker::new_cv_position(50.0, 200.0);

    let all_wps: Vec<Vec<_>> = trajectories.iter().map(|t| t.generate()).collect();
    let n_frames = all_wps[0].len();

    for frame_idx in 0..n_frames {
        let mut detections = Vec::new();
        for target_wps in &all_wps {
            let wp = &target_wps[frame_idx];
            if let Some(m) = generate_radar(wp, &radar, &mut rng) {
                detections.push(radar_to_cartesian(&m.to_vector()));
            }
        }
        tracker.step(&detections, 1.0);
    }

    assert!(
        tracker.alive_count() >= 3,
        "Expected ~5 alive tracks, got {}",
        tracker.alive_count()
    );
}

#[test]
fn benchmark_throughput() {
    use std::time::Instant;

    let mut tracker = MultiObjectTracker::new_cv_position(50.0, 200.0);
    let n_targets = 50;
    let n_frames = 100;

    let start = Instant::now();
    for frame in 0..n_frames {
        let detections: Vec<DVector<f64>> = (0..n_targets)
            .map(|i| {
                DVector::from_column_slice(&[
                    i as f64 * 1000.0 + frame as f64 * 100.0,
                    i as f64 * 500.0,
                    5000.0,
                ])
            })
            .collect();
        tracker.step(&detections, 0.1);
    }
    let elapsed = start.elapsed();
    let hz = n_frames as f64 / elapsed.as_secs_f64();

    assert!(
        hz > 10.0,
        "Tracker too slow: {hz:.1} Hz for {n_targets} targets"
    );
}

#[test]
fn class_specific_mixed_targets() {
    // Aircraft trajectories: CV segments, moderate speed, moderate altitude
    let aircraft: Vec<Trajectory> = vec![
        Trajectory {
            target_id: 0,
            initial_position: [0.0, 0.0, 5000.0],
            initial_velocity: [250.0, 0.0, 0.0],
            segments: vec![Segment {
                segment_type: SegmentType::Cv,
                duration: 15.0,
            }],
            dt: 1.0,
        },
        Trajectory {
            target_id: 1,
            initial_position: [0.0, 20000.0, 8000.0],
            initial_velocity: [200.0, 50.0, 0.0],
            segments: vec![Segment {
                segment_type: SegmentType::Cv,
                duration: 15.0,
            }],
            dt: 1.0,
        },
    ];

    // Ballistic trajectories: high speed, high altitude, no drag
    let ballistic: Vec<Trajectory> = vec![
        Trajectory {
            target_id: 2,
            initial_position: [50000.0, 50000.0, 80000.0],
            initial_velocity: [1500.0, 0.0, 200.0],
            segments: vec![Segment {
                segment_type: SegmentType::Ballistic {
                    drag_coefficient: 0.0,
                },
                duration: 15.0,
            }],
            dt: 1.0,
        },
        Trajectory {
            target_id: 3,
            initial_position: [-40000.0, 60000.0, 100000.0],
            initial_velocity: [1000.0, -500.0, 100.0],
            segments: vec![Segment {
                segment_type: SegmentType::Ballistic {
                    drag_coefficient: 0.0,
                },
                duration: 15.0,
            }],
            dt: 1.0,
        },
    ];

    let all_trajectories: Vec<&Trajectory> = aircraft.iter().chain(ballistic.iter()).collect();

    let radar = RadarConfig {
        p_detection: 1.0,
        range_sigma: 5.0,
        azimuth_sigma: 0.0005,
        elevation_sigma: 0.0005,
        ..Default::default()
    };

    let mut rng = rand::rng();
    // Use a large gate to allow the CV model to associate ballistic targets
    // despite model mismatch (no gravity in CV model).
    let mut tracker = MultiObjectTracker::new_cv_position(100.0, 5000.0);

    let all_wps: Vec<Vec<_>> = all_trajectories.iter().map(|t| t.generate()).collect();
    let n_frames = all_wps.iter().map(|w| w.len()).min().unwrap();
    let mut eval_frames = Vec::new();

    for frame_idx in 0..n_frames {
        let mut detections = Vec::new();
        let mut gt = Vec::new();

        for (traj_idx, target_wps) in all_wps.iter().enumerate() {
            let wp = &target_wps[frame_idx];
            gt.push((traj_idx as u64, wp.position));
            if let Some(m) = generate_radar(wp, &radar, &mut rng) {
                detections.push(radar_to_cartesian(&m.to_vector()));
            }
        }

        tracker.step(&detections, 1.0);

        let tracks: Vec<(u64, [f64; 3])> = tracker
            .tracks
            .iter()
            .filter(|t| t.is_alive())
            .map(|t| (t.id.0, [t.state[0], t.state[2], t.state[4]]))
            .collect();
        eval_frames.push(FrameData { gt, tracks });
    }

    // Large threshold for matching: ballistic targets under CV model
    // accumulate position error from unmodeled gravity.
    let (mota, _motp, _idsw) = compute_mot_metrics(&eval_frames, 2000.0);
    let idf1 = compute_idf1(&eval_frames, 2000.0);

    let alive = tracker.tracks.iter().filter(|t| t.is_alive()).count();

    assert!(mota > 0.4, "MOTA too low for mixed-class scenario: {mota}");
    assert!(idf1 > 0.0, "IDF1 should be positive: {idf1}");
    assert!(alive >= 2, "Expected at least 2 alive tracks, got {alive}");
}
