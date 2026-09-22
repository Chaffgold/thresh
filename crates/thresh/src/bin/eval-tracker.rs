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
//! of the analytic baseline. With `onnx`, `--learned-detector --detector-model
//! <onnx>` runs point clouds through the detector before tracking. Optional
//! `--detector-baseline-model <onnx>` compares another detector on the exact
//! same materialized point clouds. Both learned modes may be combined.

#[cfg(feature = "learned-imm")]
use thresh::eval_harness::run_eval_harness_learned;
use thresh::eval_harness::{DEFAULT_DISTANCE_THRESHOLD_M, EvalReport, run_eval_harness};
use thresh::synth::radar_trajectory::{TargetTrack, TrajectoryRadarConfig};
use thresh::synth::trajectory::{Segment, SegmentType, Trajectory};
use thresh::training::ImmTrainingParams;
#[cfg(feature = "onnx")]
use thresh::{
    eval_harness::{DetectorEvalSequence, run_detector_eval},
    filter::imm::ImmConfig,
    inference::detection::{OnnxDetector, OnnxDetectorConfig},
    tracker::tracker::MultiObjectTracker,
};

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

fn maneuver_target(target_id: u32, pos: [f64; 3], vel: [f64; 3], duration: f64) -> TargetTrack {
    let waypoints = Trajectory {
        target_id,
        initial_position: pos,
        initial_velocity: vel,
        segments: vec![
            Segment {
                segment_type: SegmentType::Cv,
                duration: duration / 3.0,
            },
            Segment {
                segment_type: SegmentType::Ctrv { turn_rate: 0.1 },
                duration: duration * 2.0 / 3.0,
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

fn scenarios(duration: f64) -> Vec<(&'static str, Vec<TargetTrack>)> {
    vec![
        (
            "single-cruise",
            vec![cv_target(
                0,
                [5_000.0, 0.0, 1_000.0],
                [120.0, 0.0, 0.0],
                duration,
            )],
        ),
        (
            "two-separated",
            vec![
                cv_target(0, [5_000.0, 0.0, 1_000.0], [120.0, 0.0, 0.0], duration),
                cv_target(1, [5_000.0, 3_000.0, 1_000.0], [120.0, 0.0, 0.0], duration),
            ],
        ),
        (
            "maneuvering",
            vec![maneuver_target(
                0,
                [6_000.0, 0.0, 1_500.0],
                [100.0, 0.0, 0.0],
                duration,
            )],
        ),
    ]
}

fn print_report(name: &str, pipeline: &str, r: &EvalReport, json: bool) {
    if json {
        println!(
            "{{\"scenario\":\"{name}\",\"pipeline\":\"{pipeline}\",\"mota\":{:.4},\"motp\":{:.4},\"idf1\":{:.4},\"id_switches\":{},\"frames\":{},\"distance_threshold_m\":{:.1}}}",
            r.mota, r.motp, r.idf1, r.id_switches, r.frames, r.distance_threshold_m
        );
    } else {
        println!(
            "{name:<16} {pipeline} MOTA={:.3}  MOTP={:.2}m  IDF1={:.3}  IDSW={}  frames={}",
            r.mota, r.motp, r.idf1, r.id_switches, r.frames
        );
    }
}

#[derive(Debug, Default)]
struct Options {
    json: bool,
    learned_imm: bool,
    learned_detector: bool,
    imm_model: Option<String>,
    detector_model: Option<String>,
    detector_baseline_model: Option<String>,
    duration_seconds: Option<f64>,
}

fn read_model(args: &mut impl Iterator<Item = String>, flag: &str) -> Result<String, String> {
    args.next()
        .filter(|value| !value.starts_with("--"))
        .ok_or_else(|| format!("{flag} requires a model path"))
}

fn parse_options(args: impl Iterator<Item = String>) -> Result<Options, String> {
    let mut options = Options::default();
    let mut args = args;
    while let Some(flag) = args.next() {
        match flag.as_str() {
            "--json" => options.json = true,
            "--learned-imm" => options.learned_imm = true,
            "--learned-detector" => options.learned_detector = true,
            "--model" | "--imm-model" => options.imm_model = Some(read_model(&mut args, &flag)?),
            "--detector-model" => options.detector_model = Some(read_model(&mut args, &flag)?),
            "--detector-baseline-model" => {
                options.detector_baseline_model = Some(read_model(&mut args, &flag)?)
            }
            "--duration-seconds" => options.duration_seconds = Some(read_duration(&mut args)?),
            _ => return Err(format!("unknown argument: {flag}")),
        }
    }
    validate_modes(
        &options,
        cfg!(feature = "learned-imm"),
        cfg!(feature = "onnx"),
    )?;
    Ok(options)
}

fn read_duration(args: &mut impl Iterator<Item = String>) -> Result<f64, String> {
    let error = "--duration-seconds requires a positive finite number";
    let duration: f64 = args.next().ok_or(error)?.parse().map_err(|_| error)?;
    if !duration.is_finite() || duration <= 0.0 {
        return Err(error.into());
    }
    Ok(duration)
}

fn validate_model_option(
    enabled: bool,
    model: Option<&str>,
    mode: &str,
    flag: &str,
) -> Result<(), String> {
    match (enabled, model) {
        (true, None) => Err(format!("{mode} requires {flag} <path>")),
        (false, Some(_)) => Err(format!("{flag} requires {mode}")),
        _ => Ok(()),
    }
}

fn validate_modes(
    options: &Options,
    imm_available: bool,
    detector_available: bool,
) -> Result<(), String> {
    if options.learned_imm && !imm_available {
        return Err("--learned-imm requires rebuilding with --features learned-imm".into());
    }
    if options.learned_detector && !detector_available {
        return Err("--learned-detector requires rebuilding with --features onnx".into());
    }
    validate_model_option(
        options.learned_imm,
        options.imm_model.as_deref(),
        "--learned-imm",
        "--imm-model (or --model)",
    )?;
    validate_model_option(
        options.learned_detector,
        options.detector_model.as_deref(),
        "--learned-detector",
        "--detector-model",
    )?;
    if options.detector_baseline_model.is_some() && !options.learned_detector {
        return Err("--detector-baseline-model requires --learned-detector".into());
    }
    Ok(())
}

fn validate_model_files(options: &Options) -> Result<(), String> {
    for path in [
        &options.imm_model,
        &options.detector_model,
        &options.detector_baseline_model,
    ]
    .into_iter()
    .flatten()
    {
        if !std::path::Path::new(path).is_file() {
            return Err(format!("model file does not exist: {path}"));
        }
    }
    Ok(())
}

/// Run one scenario, choosing the analytic or learned-IMM tracker.
fn eval_one(
    targets: &[TargetTrack],
    config: &TrajectoryRadarConfig,
    model: Option<&str>,
) -> Result<EvalReport, Box<dyn std::error::Error>> {
    #[cfg(feature = "learned-imm")]
    if let Some(path) = model {
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
    #[cfg(not(feature = "learned-imm"))]
    if model.is_some() {
        return Err("learned-IMM feature is unavailable".into());
    }
    run_eval_harness(
        targets,
        config,
        ImmTrainingParams::default(),
        DEFAULT_DISTANCE_THRESHOLD_M,
        SEED,
    )
    .map_err(Into::into)
}

#[cfg(feature = "onnx")]
fn detector_tracker(imm_model: Option<&str>) -> Result<MultiObjectTracker, String> {
    let imm = ImmTrainingParams::default();
    let bank =
        move || ImmConfig::cv_ca_ctrv_ct(imm.sigma_a, imm.sigma_j, imm.sigma_v, imm.sigma_omega);
    #[cfg(feature = "learned-imm")]
    if let Some(path) = imm_model {
        return MultiObjectTracker::new_imm_position_learned(
            bank,
            path,
            imm.measurement_noise_sigma,
            imm.gate_threshold,
        );
    }
    #[cfg(not(feature = "learned-imm"))]
    if imm_model.is_some() {
        return Err("learned-IMM feature is unavailable".into());
    }
    Ok(MultiObjectTracker::new_imm_position(
        bank,
        imm.measurement_noise_sigma,
        imm.gate_threshold,
    ))
}

#[cfg(feature = "onnx")]
fn eval_detectors(
    name: &str,
    targets: &[TargetTrack],
    config: &TrajectoryRadarConfig,
    options: &Options,
) -> Result<(), Box<dyn std::error::Error>> {
    let sequence = DetectorEvalSequence::from_trajectories(targets, config, SEED)?;
    let defaults = OnnxDetectorConfig::default();
    for (role, path) in [
        ("detector-candidate", &options.detector_model),
        ("detector-baseline", &options.detector_baseline_model),
    ] {
        if let Some(path) = path {
            let detector = OnnxDetector::load(
                path,
                defaults.confidence_threshold,
                defaults.nms_iou_threshold,
            )?;
            let report = run_detector_eval(
                &sequence,
                &detector,
                detector_tracker(options.imm_model.as_deref())?,
                DEFAULT_DISTANCE_THRESHOLD_M,
            );
            let imm = if options.learned_imm {
                "learned-imm"
            } else {
                "analytic-imm"
            };
            print_report(name, &format!("{role}/{imm}"), &report, options.json);
        }
    }
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let options = parse_options(std::env::args().skip(1))?;
    validate_model_files(&options)?;

    let config = TrajectoryRadarConfig {
        sample_rate_hz: 10.0,
        ..TrajectoryRadarConfig::default()
    };
    for (name, targets) in scenarios(options.duration_seconds.unwrap_or(30.0)) {
        #[cfg(feature = "onnx")]
        if options.learned_detector {
            eval_detectors(name, &targets, &config, &options)?;
            continue;
        }
        let report = eval_one(&targets, &config, options.imm_model.as_deref())?;
        let pipeline = if options.learned_imm {
            "radar-measurements/learned-imm"
        } else {
            "radar-measurements/analytic-imm"
        };
        print_report(name, pipeline, &report, options.json);
    }
    Ok(())
}
