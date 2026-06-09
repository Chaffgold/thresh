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
//! With the `learned-imm` feature, `--learned-imm --model <onnx>` runs the
//! learned-IMM tracker (`MultiObjectTracker::new_imm_position_learned`) instead
//! of the analytic baseline. `--learned-detector` is accepted but not yet
//! implemented and falls back to the baseline.

#[cfg(feature = "learned-imm")]
use thresh::eval_harness::run_eval_harness_learned;
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

/// Value following `flag` in `args` (e.g. `--model path` → `Some("path")`).
fn flag_value<'a>(args: &'a [String], flag: &str) -> Option<&'a str> {
    args.iter()
        .position(|a| a == flag)
        .and_then(|i| args.get(i + 1))
        .map(String::as_str)
}

/// Run one scenario, choosing the analytic or learned-IMM tracker.
fn eval_one(
    targets: &[TargetTrack],
    config: &TrajectoryRadarConfig,
    learned_imm: bool,
    model: Option<&str>,
) -> Result<EvalReport, Box<dyn std::error::Error>> {
    #[cfg(feature = "learned-imm")]
    if learned_imm && let Some(path) = model {
        return run_eval_harness_learned(
            targets,
            config,
            ImmTrainingParams::default(),
            path,
            DEFAULT_DISTANCE_THRESHOLD_M,
            SEED,
        )
        .map_err(Into::into);
    }
    let _ = (learned_imm, model); // analytic baseline ignores these
    run_eval_harness(
        targets,
        config,
        ImmTrainingParams::default(),
        DEFAULT_DISTANCE_THRESHOLD_M,
        SEED,
    )
    .map_err(Into::into)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let json = args.iter().any(|a| a == "--json");
    // `--learned-imm` and `--learned-detector` are distinct flags: only the
    // former drives the learned-IMM tracker (and needs `--model`); the latter
    // (a future learned detector) is not implemented and runs the baseline.
    let learned_imm = args.iter().any(|a| a == "--learned-imm");
    let learned_detector = args.iter().any(|a| a == "--learned-detector");
    let model = flag_value(&args, "--model");

    if learned_detector && !json {
        eprintln!(
            "note: --learned-detector is not implemented yet; running the analytic baseline."
        );
    }
    #[cfg(not(feature = "learned-imm"))]
    if learned_imm && !json {
        eprintln!(
            "note: --learned-imm requested but the `learned-imm` feature is not enabled; \
             running the analytic baseline. Rebuild with `--features learned-imm`."
        );
    }
    #[cfg(feature = "learned-imm")]
    if learned_imm && model.is_none() {
        return Err("--learned-imm requires --model <path-to-imm_mode_classifier.onnx>".into());
    }

    let config = TrajectoryRadarConfig {
        sample_rate_hz: 10.0,
        ..TrajectoryRadarConfig::default()
    };
    for (name, targets) in scenarios() {
        let report = eval_one(&targets, &config, learned_imm, model)?;
        print_report(name, &report, json);
    }
    Ok(())
}
