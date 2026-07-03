//! Generate the Track A detector training-sample Parquet (task 7.2).
//!
//! Built behind the `training-export` feature. Drives a couple of small
//! synthetic scenes through the Phase 4 synth pairing (`from_trajectory`) and
//! writes `(point_cloud, gt_boxes, gt_valid, gt_classes)` snapshots to Parquet.
//!
//! ```sh
//! cargo run -p thresh --features training-export --bin gen-detector-dataset -- \
//!   test-data/training/detector/detector-samples.parquet
//! ```
//!
//! The checked-in sample under `test-data/training/detector/` is produced by
//! this binary with the default arguments (kept small via short scenes + Snappy
//! compression). The full dataset over real acquisition trajectories is wired
//! up alongside the real training run (task 7.5).

use thresh::synth::radar_trajectory::{TargetTrack, TrajectoryRadarConfig};
use thresh::synth::trajectory::{Segment, SegmentType, Trajectory};
use thresh::training::detector::{DetectorSample, generate_detector_samples};
use thresh::training::parquet_export::write_detector_samples_parquet;
use thresh::training::trajectory_ingest::{
    GenArgs, IngestConfig, generate_per_track, ingest_trajectories,
};

const DEFAULT_OUT: &str = "test-data/training/detector/detector-samples.parquet";

fn cv_target(target_id: u32, pos: [f64; 3], vel: [f64; 3], class_id: u32) -> TargetTrack {
    let waypoints = Trajectory {
        target_id,
        initial_position: pos,
        initial_velocity: vel,
        segments: vec![Segment {
            segment_type: SegmentType::Cv,
            duration: 4.0,
        }],
        dt: 0.5,
    }
    .generate();
    TargetTrack {
        waypoints,
        class_id,
        size_override: None,
    }
}

/// Real-data mode: one single-target scene per ingested acquisition track.
fn samples_from_acquisition(
    traj_path: &std::path::Path,
    sample_rate_hz: f64,
) -> Result<Vec<DetectorSample>, Box<dyn std::error::Error>> {
    let ingest_config = IngestConfig {
        sample_rate_hz,
        ..IngestConfig::default()
    };
    let config = TrajectoryRadarConfig {
        sample_rate_hz,
        ..TrajectoryRadarConfig::default()
    };
    let tracks = ingest_trajectories(traj_path, &ingest_config)?;
    println!(
        "ingested {} track segments from {}",
        tracks.len(),
        traj_path.display()
    );
    Ok(generate_per_track(&tracks, |trajectory_id, track| {
        let target = TargetTrack {
            waypoints: track.waypoints.clone(),
            class_id: track.class_id,
            size_override: None,
        };
        generate_detector_samples(
            trajectory_id,
            std::slice::from_ref(&target),
            &config,
            2_000 + trajectory_id as u64,
        )
    }))
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Real-data default: one snapshot every 5 s bounds the dataset size while
    // still yielding hundreds of snapshots per captured track.
    let args = GenArgs::parse(DEFAULT_OUT, 0.2);

    if let Some(traj_path) = &args.trajectories {
        let samples = samples_from_acquisition(traj_path, args.sample_rate_hz)?;
        write_detector_samples_parquet(&samples, &args.out)?;
        println!(
            "wrote {} detector samples to {}",
            samples.len(),
            args.out.display()
        );
        return Ok(());
    }
    let out = args.out;

    // Low sample rate + short scenes keep the committed sample tiny.
    let config = TrajectoryRadarConfig {
        sample_rate_hz: 2.0,
        ..TrajectoryRadarConfig::default()
    };

    // Scene 0: a single heavy-fixed-wing cruising. Scene 1: a light-fixed-wing
    // and a rotorcraft co-observed (multi-target snapshots).
    let scenes: Vec<(u32, Vec<TargetTrack>, u64)> = vec![
        (
            0,
            vec![cv_target(0, [5_000.0, 0.0, 1_000.0], [120.0, 0.0, 0.0], 1)],
            10,
        ),
        (
            1,
            vec![
                cv_target(0, [4_000.0, 1_000.0, 800.0], [90.0, 0.0, 0.0], 0),
                cv_target(1, [3_500.0, -800.0, 600.0], [0.0, 60.0, 0.0], 2),
            ],
            11,
        ),
    ];

    let mut samples: Vec<DetectorSample> = Vec::new();
    for (id, targets, seed) in &scenes {
        samples.extend(generate_detector_samples(*id, targets, &config, *seed)?);
    }

    write_detector_samples_parquet(&samples, &out)?;
    println!(
        "wrote {} detector samples to {}",
        samples.len(),
        out.display()
    );
    Ok(())
}
