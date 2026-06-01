//! Evaluate the analytic IMM tracker on synthetic scenarios and report MOT
//! metrics (flight-data-training-pipeline, Phase 9; design Decision 24).
//!
//! Rust-native eval driver: synth → analytic 4-model IMM tracker → MOTA / MOTP
//! / IDF1 (via `thresh-eval`) against ground-truth trajectories. Prints a JSON
//! line per scenario; `python/eval/run_tracker.py` wraps this binary.
//!
//! ```sh
//! cargo run -p thresh --bin eval-tracker            # built-in scenarios
//! cargo run -p thresh --bin eval-tracker -- --json  # one JSON object per scenario
//! ```
//!
//! `--learned-imm` / `--learned-detector` are accepted for forward
//! compatibility but currently fall back to the analytic baseline: the tracker
//! has no learned-IMM constructor yet (Phase 6's `LearnedImmFilter` is
//! filter-level) and the trained checkpoints are deferred.

use thresh::eval_harness::{DEFAULT_DISTANCE_THRESHOLD_M, EvalReport, run_eval_harness};
use thresh::synth::radar_trajectory::{TargetTrack, TrajectoryRadarConfig};
use thresh::synth::trajectory::{Segment, SegmentType, Trajectory};
use thresh::training::ImmTrainingParams;

const SEED: u64 = 42;

fn cv_target(target_id: u32, pos: [f64; 3], vel: [f64; 3], duration: f64) -> TargetTrack {
    let waypoints = Trajectory {
        target_id,
        initial_position: pos,
        initial_velocity: vel,
        segments: vec![Segment {
            segment_type: SegmentType::Cv,
            duration,
        }],
        dt: 0.1,
    }
    .generate();
    TargetTrack {
        waypoints,
        class_id: 1,
        size_override: None,
    }
}

fn maneuver_target(target_id: u32, pos: [f64; 3], vel: [f64; 3]) -> TargetTrack {
    let waypoints = Trajectory {
        target_id,
        initial_position: pos,
        initial_velocity: vel,
        segments: vec![
            Segment {
                segment_type: SegmentType::Cv,
                duration: 10.0,
            },
            Segment {
                segment_type: SegmentType::Ctrv { turn_rate: 0.1 },
                duration: 20.0,
            },
        ],
        dt: 0.1,
    }
    .generate();
    TargetTrack {
        waypoints,
        class_id: 1,
        size_override: None,
    }
}

fn scenarios() -> Vec<(&'static str, Vec<TargetTrack>)> {
    vec![
        (
            "single-cruise",
            vec![cv_target(
                0,
                [5_000.0, 0.0, 1_000.0],
                [120.0, 0.0, 0.0],
                30.0,
            )],
        ),
        (
            "two-separated",
            vec![
                cv_target(0, [5_000.0, 0.0, 1_000.0], [120.0, 0.0, 0.0], 30.0),
                cv_target(1, [5_000.0, 3_000.0, 1_000.0], [120.0, 0.0, 0.0], 30.0),
            ],
        ),
        (
            "maneuvering",
            vec![maneuver_target(
                0,
                [6_000.0, 0.0, 1_500.0],
                [100.0, 0.0, 0.0],
            )],
        ),
    ]
}

fn print_report(name: &str, r: &EvalReport, json: bool) {
    if json {
        println!(
            "{{\"scenario\":\"{name}\",\"mota\":{:.4},\"motp\":{:.4},\"idf1\":{:.4},\"id_switches\":{},\"frames\":{},\"distance_threshold_m\":{:.1}}}",
            r.mota, r.motp, r.idf1, r.id_switches, r.frames, r.distance_threshold_m
        );
    } else {
        println!(
            "{name:<16} MOTA={:.3}  MOTP={:.2}m  IDF1={:.3}  IDSW={}  frames={}",
            r.mota, r.motp, r.idf1, r.id_switches, r.frames
        );
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let json = args.iter().any(|a| a == "--json");
    let learned = args
        .iter()
        .any(|a| a == "--learned-imm" || a == "--learned-detector");
    if learned && !json {
        eprintln!(
            "note: --learned-* requested but not yet wired into MultiObjectTracker; \
             running the analytic baseline (see Phase 9 / design Decision 24)."
        );
    }

    let config = TrajectoryRadarConfig {
        sample_rate_hz: 10.0,
        ..TrajectoryRadarConfig::default()
    };
    for (name, targets) in scenarios() {
        let report = run_eval_harness(
            &targets,
            &config,
            ImmTrainingParams::default(),
            DEFAULT_DISTANCE_THRESHOLD_M,
            SEED,
        )?;
        print_report(name, &report, json);
    }
    Ok(())
}
