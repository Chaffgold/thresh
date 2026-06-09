//! Evaluate the automotive tracker over nuScenes scenes and report per-class
//! MOT metrics (automotive-tracking-pipeline, Phase 4, task 4.2).
//!
//! Drives `MultiObjectTracker::new_automotive_enu` over nuScenes keyframes via
//! `NuScenesBridge`: per sample, the scene's annotation boxes become
//! class-tagged detections (optionally Gaussian-perturbed), the tracker steps
//! in the nuScenes global frame, and confirmed tracks are scored against the
//! instance ground truth with `thresh-eval`'s class-blind + per-class metrics.
//!
//! Requires `--features nuscenes-eval`, the Python `nuscenes-devkit`, and a
//! local nuScenes download (see `docs/eval/automotive-tracking.md`); the
//! dataset is never fetched in CI.
//!
//! ```sh
//! cargo run -p thresh --features nuscenes-eval --bin eval-nuscenes -- \
//!   --dataroot /data/nuscenes --version v1.0-mini [--scenes 2] \
//!   [--noise-sigma 0.3] [--json]
//! ```

use nalgebra::DVector;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Normal};
use thresh::core::track::{TargetClass, TrackState};
use thresh::data::nuscenes::NuScenesBridge;
use thresh::eval::per_class::{ClassedFrameData, build_classed_report, class_averaged_mota};
use thresh::tracker::tracker::MultiObjectTracker;

const SEED: u64 = 42;
const DEFAULT_DISTANCE_THRESHOLD_M: f64 = 2.0;

fn flag_value<'a>(args: &'a [String], flag: &str) -> Option<&'a str> {
    args.iter()
        .position(|a| a == flag)
        .and_then(|i| args.get(i + 1))
        .map(String::as_str)
}

/// Per-sample ground truth: `(target_id, class, center)` per annotated object.
type SampleGt = Vec<(u64, TargetClass, [f64; 3])>;

/// Evaluate one scene; returns its classed evaluation frames. Track ids are
/// offset by `track_id_base` so concatenating scenes cannot alias ids across
/// scene boundaries (which would fabricate ID switches).
fn eval_scene(
    bridge: &NuScenesBridge,
    scene_token: &str,
    noise_sigma: f64,
    track_id_base: u64,
    rng: &mut StdRng,
) -> Result<Vec<ClassedFrameData>, Box<dyn std::error::Error>> {
    let samples = bridge.iter_samples(scene_token)?;
    let instances = bridge.scene_instance_tracks(scene_token)?;

    // Index ground truth per sample token.
    let mut gt_by_sample: std::collections::HashMap<&str, SampleGt> =
        std::collections::HashMap::new();
    for inst in &instances {
        for (sample_token, bbox) in &inst.samples {
            gt_by_sample
                .entry(sample_token.as_str())
                .or_default()
                .push((inst.target_id, inst.class, bbox.center()));
        }
    }

    let noise = Normal::new(0.0, noise_sigma.max(0.0))?;
    // The filter's measurement-noise R is deliberately matched to the injected
    // detection noise (floored at 0.5 m so the ideal-detection case keeps a
    // sane gate) — a well-specified filter assumes the noise it actually sees.
    // Decouple with a separate flag if mis-specification studies are wanted.
    let mut tracker = MultiObjectTracker::new_automotive_enu(noise_sigma.max(0.5), 16.0);
    let mut frames = Vec::with_capacity(samples.len());
    let mut prev_time_us: Option<i64> = None;

    for sample in &samples {
        let dt = prev_time_us
            .map(|p| (sample.timestamp_us - p) as f64 / 1e6)
            .filter(|d| *d > 0.0)
            .unwrap_or(0.5); // nuScenes keyframes are 2 Hz
        prev_time_us = Some(sample.timestamp_us);

        let gt = gt_by_sample
            .get(sample.token.as_str())
            .cloned()
            .unwrap_or_default();

        let detections: Vec<(DVector<f64>, TargetClass)> = gt
            .iter()
            .map(|&(_, class, p)| {
                let dx = noise.sample(rng);
                let dy = noise.sample(rng);
                let dz = noise.sample(rng);
                (
                    DVector::from_column_slice(&[p[0] + dx, p[1] + dy, p[2] + dz]),
                    class,
                )
            })
            .collect();
        tracker.step_classed(&detections, dt);

        frames.push(ClassedFrameData {
            gt: gt.iter().map(|&(id, class, p)| (id, p, class)).collect(),
            tracks: tracker
                .tracks
                .iter()
                .filter(|t| t.lifecycle == TrackState::Confirmed)
                .map(|t| {
                    (
                        track_id_base + t.id.0,
                        [t.state[0], t.state[2], t.state[4]],
                        t.class,
                    )
                })
                .collect(),
        });
    }
    Ok(frames)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let dataroot = flag_value(&args, "--dataroot").ok_or(
        "--dataroot <path-to-nuscenes> is required (see docs/eval/automotive-tracking.md)",
    )?;
    let version = flag_value(&args, "--version").unwrap_or("v1.0-mini");
    let max_scenes: usize = flag_value(&args, "--scenes")
        .map(str::parse)
        .transpose()?
        .unwrap_or(usize::MAX);
    let noise_sigma: f64 = flag_value(&args, "--noise-sigma")
        .map(str::parse)
        .transpose()?
        .unwrap_or(0.0);
    let json = args.iter().any(|a| a == "--json");

    let bridge = NuScenesBridge::new(version, dataroot)?;
    let scene_tokens = bridge.scene_tokens()?;
    let mut rng = StdRng::seed_from_u64(SEED);

    let mut frames = Vec::new();
    for (i, token) in scene_tokens.iter().take(max_scenes).enumerate() {
        // 2^32 per scene leaves room for any realistic per-scene track count.
        let base = (i as u64) << 32;
        frames.extend(eval_scene(&bridge, token, noise_sigma, base, &mut rng)?);
    }

    let report = build_classed_report(&frames, DEFAULT_DISTANCE_THRESHOLD_M);
    let class_avg = class_averaged_mota(&report.per_class);
    if json {
        println!("{}", report.to_json());
        println!("{{\"class_averaged_mota\":{class_avg:.4}}}");
    } else {
        println!("{}", report.to_table());
        println!("Class-averaged MOTA: {class_avg:.4}");
        println!(
            "({} scenes, noise σ = {noise_sigma} m, threshold = {DEFAULT_DISTANCE_THRESHOLD_M} m)",
            scene_tokens.len().min(max_scenes)
        );
    }
    Ok(())
}
