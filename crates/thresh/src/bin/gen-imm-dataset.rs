//! Generate the Track B IMM-classifier training-sample Parquet (task 5.5).
//!
//! Built behind the `training-export` feature. Drives a couple of built-in
//! synthetic trajectories (a straight-line cruise and a banked coordinated
//! turn) through the full synth → IMM-tracker → feature pipeline and writes the
//! resulting samples to Parquet.
//!
//! ```sh
//! cargo run -p thresh --features training-export --bin gen-imm-dataset -- \
//!   test-data/training/imm-classifier/imm-samples.parquet
//! ```
//!
//! The small checked-in sample under `test-data/training/imm-classifier/` is
//! produced by this binary with the default arguments. The full dataset (over
//! real acquisition-layer trajectories) is wired up in Phase 6; this binary is
//! the entry point that Phase 6 extends to ingest those trajectories.

use thresh::synth::radar_trajectory::TrajectoryRadarConfig;
use thresh::synth::trajectory::{Segment, SegmentType, Trajectory};
use thresh::training::parquet_export::write_imm_samples_parquet;
use thresh::training::trajectory_ingest::{
    GenArgs, IngestConfig, generate_per_track, ingest_trajectories,
};
use thresh::training::{ImmTrainingParams, ImmTrainingSample, generate_imm_training_samples};

const DEFAULT_OUT: &str = "test-data/training/imm-classifier/imm-samples.parquet";

/// Straight-line cruise a few km from the sensor (mostly `CV`).
fn cruise() -> Trajectory {
    Trajectory {
        target_id: 0,
        initial_position: [5_000.0, 0.0, 1_000.0],
        initial_velocity: [120.0, 0.0, 0.0],
        segments: vec![Segment {
            segment_type: SegmentType::Cv,
            duration: 20.0,
        }],
        dt: 0.1,
    }
}

/// Cruise then a sustained banked turn (`CV` lead-in, then `coord_turn`).
fn maneuver() -> Trajectory {
    Trajectory {
        target_id: 1,
        initial_position: [6_000.0, 0.0, 1_500.0],
        initial_velocity: [100.0, 0.0, 0.0],
        segments: vec![
            Segment {
                segment_type: SegmentType::Cv,
                duration: 5.0,
            },
            Segment {
                segment_type: SegmentType::Ctrv { turn_rate: 0.1 },
                duration: 25.0,
            },
        ],
        dt: 0.1,
    }
}

/// Real-data mode: every ingested acquisition track becomes one trajectory run.
fn samples_from_acquisition(
    traj_path: &std::path::Path,
    sample_rate_hz: f64,
) -> Result<Vec<ImmTrainingSample>, Box<dyn std::error::Error>> {
    let ingest_config = IngestConfig {
        sample_rate_hz,
        ..IngestConfig::default()
    };
    let config = TrajectoryRadarConfig {
        sample_rate_hz,
        ..TrajectoryRadarConfig::default()
    };
    let params = ImmTrainingParams::default();
    let tracks = ingest_trajectories(traj_path, &ingest_config)?;
    println!(
        "ingested {} track segments from {}",
        tracks.len(),
        traj_path.display()
    );
    Ok(generate_per_track(&tracks, |trajectory_id, track| {
        generate_imm_training_samples(
            trajectory_id,
            &track.waypoints,
            track.class_id,
            &config,
            params,
            1_000 + trajectory_id as u64,
        )
    }))
}

fn synthetic_samples() -> Result<Vec<ImmTrainingSample>, Box<dyn std::error::Error>> {
    let config = TrajectoryRadarConfig {
        sample_rate_hz: 10.0,
        ..TrajectoryRadarConfig::default()
    };
    let params = ImmTrainingParams::default();
    let mut samples: Vec<ImmTrainingSample> = Vec::new();
    for (traj, seed) in [(cruise(), 1_u64), (maneuver(), 2_u64)] {
        let waypoints = traj.generate();
        samples.extend(generate_imm_training_samples(
            traj.target_id,
            &waypoints,
            1, // heavy-fixed-wing
            &config,
            params,
            seed,
        )?);
    }
    Ok(samples)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = GenArgs::parse(DEFAULT_OUT, 1.0);

    let samples = match &args.trajectories {
        Some(traj_path) => samples_from_acquisition(traj_path, args.sample_rate_hz)?,
        None => synthetic_samples()?,
    };

    write_imm_samples_parquet(&samples, &args.out)?;
    println!("wrote {} samples to {}", samples.len(), args.out.display());
    Ok(())
}
