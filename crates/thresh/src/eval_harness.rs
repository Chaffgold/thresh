//! Rust-native tracker evaluation (flight-data-training-pipeline, Phase 9).
//!
//! Runs synth measurements from a set of target trajectories through the
//! analytic 4-model IMM tracker and reports MOTA / MOTP / IDF1 (via
//! `thresh-eval`) against ground truth. This is the Rust-native eval driver
//! (design Decision 24): the `thresh-py` path is CV-only and the Python synth
//! binding is deferred, so the evaluation engine lives here and produces real
//! baseline numbers locally; `python/eval/run_tracker.py` is a thin wrapper
//! over the `eval-tracker` binary.
//!
//! **Ground truth** is built independently from the trajectory waypoints (the
//! target's input index is its stable GT id), translated into the sensor frame
//! and range-gated — *not* from `from_trajectory`'s `gt_boxes`, whose slot
//! assignment is unstable across ticks. A target that is in range but missed by
//! the sensor that tick therefore correctly counts as a false negative.
//!
//! **Learned A/B (task 9.3):** [`run_eval_harness`] computes the analytic
//! baseline; with the `learned-imm` feature, `run_eval_harness_learned` runs
//! the same scenario through `MultiObjectTracker::new_imm_position_learned`
//! (the ONNX mode classifier, with analytic fallback) so the two can be
//! compared on identical inputs. A *representative* comparison still needs a
//! trained checkpoint (the committed model is a random-weight stub).
//! [`DetectorEvalSequence`] and [`run_detector_eval`] evaluate detector outputs
//! against independent truth, reusing identical point clouds across detector
//! candidates. The analytic radar-measurement path is a distinct sensor input,
//! not a same-observation detector baseline.

use nalgebra::DVector;
use rand::SeedableRng;
use rand::rngs::StdRng;

use thresh_core::coords::spherical_to_cartesian;
use thresh_core::measurement::Measurement;
use thresh_core::track::TrackState;
use thresh_eval::matching::FrameData;
use thresh_eval::metrics::{compute_idf1, compute_mot_metrics};
use thresh_filter::imm::ImmConfig;
use thresh_inference::detection::{DetectionPipeline, SensorInput};
use thresh_synth::radar_trajectory::{
    POINT_DIM, TargetTrack, TrajectoryRadarConfig, TrajectoryRadarError, from_trajectory,
    measurements_from_trajectory,
};
use thresh_synth::trajectory::Waypoint;
use thresh_tracker::tracker::MultiObjectTracker;

use crate::training::ImmTrainingParams;

/// MOT evaluation result for one tracker run.
#[derive(Debug, Clone, PartialEq)]
pub struct EvalReport {
    /// Multiple-Object Tracking Accuracy (1 − (FN + FP + IDSW)/GT).
    pub mota: f64,
    /// Multiple-Object Tracking Precision (mean matched-pair distance, m).
    pub motp: f64,
    /// Identity F1.
    pub idf1: f64,
    /// Total ID switches.
    pub id_switches: usize,
    /// Number of evaluated frames.
    pub frames: usize,
    /// Matching distance gate used (m).
    pub distance_threshold_m: f64,
}

/// Default matching gate (m). Synth radar noise (~10 m range, ~1 mrad angular)
/// plus tracker convergence error sits comfortably under this.
pub const DEFAULT_DISTANCE_THRESHOLD_M: f64 = 50.0;

/// Materialized observations and independent truth for detector comparisons.
///
/// Generate this once, then pass the same sequence to every detector under
/// comparison. Only point clouds (never GT boxes/classes) reach the detector.
pub struct DetectorEvalSequence {
    inputs: Vec<SensorInput>,
    truth: Vec<Vec<(u64, [f64; 3])>>,
    dt: f64,
}

impl DetectorEvalSequence {
    /// Synthesize sensor-frame point clouds; derive stable-ID truth directly
    /// from the trajectories, including targets missed by the sensor.
    pub fn from_trajectories(
        targets: &[TargetTrack],
        config: &TrajectoryRadarConfig,
        seed: u64,
    ) -> Result<Self, TrajectoryRadarError> {
        let snapshots = from_trajectory(targets, config, &mut StdRng::seed_from_u64(seed))?;
        let mut inputs = Vec::with_capacity(snapshots.len());
        let mut truth = Vec::with_capacity(snapshots.len());
        for snapshot in snapshots {
            let (points, _) = snapshot.point_cloud.as_chunks::<POINT_DIM>();
            inputs.push(SensorInput {
                points: points
                    .iter()
                    .map(|p| [p[0] as f64, p[1] as f64, p[2] as f64])
                    .collect(),
                intensities: Some(points.iter().map(|p| p[3] as f64).collect()),
                timestamp: snapshot.time_s,
            });
            truth.push(ground_truth(targets, config, snapshot.time_s));
        }
        Ok(Self {
            inputs,
            truth,
            dt: 1.0 / config.sample_rate_hz,
        })
    }
}

/// Run a detector and a fresh position tracker on a shared observation sequence.
///
/// The caller may supply either an analytic or learned-IMM tracker. Supplying a
/// fresh tracker for each run prevents history leaking between comparisons.
pub fn run_detector_eval(
    sequence: &DetectorEvalSequence,
    detector: &dyn DetectionPipeline,
    mut tracker: MultiObjectTracker,
    distance_threshold_m: f64,
) -> EvalReport {
    let frames: Vec<_> = sequence
        .inputs
        .iter()
        .zip(&sequence.truth)
        .map(|(input, gt)| {
            let detections = detector.detect(input);
            tracker.step_detections(&detections, sequence.dt);
            FrameData {
                gt: gt.clone(),
                tracks: confirmed_positions(&tracker),
            }
        })
        .collect();
    report_from_frames(&frames, distance_threshold_m)
}

fn ground_truth(
    targets: &[TargetTrack],
    config: &TrajectoryRadarConfig,
    time: f64,
) -> Vec<(u64, [f64; 3])> {
    targets
        .iter()
        .enumerate()
        .filter_map(|(id, target)| {
            gt_position_at(
                target,
                time,
                config.sensor.position_enu_m,
                config.max_range_m,
            )
            .map(|position| (id as u64, position))
        })
        .collect()
}

fn confirmed_positions(tracker: &MultiObjectTracker) -> Vec<(u64, [f64; 3])> {
    tracker
        .tracks
        .iter()
        .filter(|track| track.lifecycle == TrackState::Confirmed)
        .map(|track| (track.id.0, [track.state[0], track.state[2], track.state[4]]))
        .collect()
}

fn report_from_frames(frames: &[FrameData], distance_threshold_m: f64) -> EvalReport {
    let (mota, motp, id_switches) = compute_mot_metrics(frames, distance_threshold_m);
    EvalReport {
        mota,
        motp,
        idf1: compute_idf1(frames, distance_threshold_m),
        id_switches,
        frames: frames.len(),
        distance_threshold_m,
    }
}

/// Convert a sensor-frame radar measurement `(range, az, el)` into a Cartesian
/// `[x, y, z]` detection (the tracker's position-only input).
fn radar_to_detection(measurement: &Measurement) -> Option<DVector<f64>> {
    match measurement {
        Measurement::Radar {
            range,
            azimuth,
            elevation,
            ..
        } => {
            let p = spherical_to_cartesian(*range, *azimuth, *elevation);
            Some(DVector::from_column_slice(&[p.x, p.y, p.z]))
        }
        _ => None,
    }
}

/// Ground-truth position of `target` at time `t` in the sensor frame, if the
/// target is active at `t` (within its waypoint time window) and within
/// `max_range_m`. Linearly interpolates the bracketing waypoints.
fn gt_position_at(
    target: &TargetTrack,
    t: f64,
    sensor: [f64; 3],
    max_range_m: f64,
) -> Option<[f64; 3]> {
    let wps = &target.waypoints;
    if wps.len() < 2 || t < wps[0].time || t > wps[wps.len() - 1].time {
        return None;
    }
    let hi = wps
        .iter()
        .position(|w| w.time >= t)
        .unwrap_or(wps.len() - 1)
        .max(1);
    let wp = Waypoint::interpolate(&wps[hi - 1], &wps[hi], t);
    let pos = [
        wp.position[0] - sensor[0],
        wp.position[1] - sensor[1],
        wp.position[2] - sensor[2],
    ];
    let range = (pos[0] * pos[0] + pos[1] * pos[1] + pos[2] * pos[2]).sqrt();
    (range <= max_range_m).then_some(pos)
}

/// Run the analytic 4-model IMM tracker over synth measurements from `targets`
/// and compute MOT metrics against the ground-truth trajectories.
///
/// `targets` should be sampled at the config's `sample_rate_hz` (so the GT tick
/// grid matches the synth's). Deterministic given `seed`.
pub fn run_eval_harness(
    targets: &[TargetTrack],
    config: &TrajectoryRadarConfig,
    imm: ImmTrainingParams,
    distance_threshold_m: f64,
    seed: u64,
) -> Result<EvalReport, TrajectoryRadarError> {
    let tracker = MultiObjectTracker::new_imm_position(
        move || ImmConfig::cv_ca_ctrv_ct(imm.sigma_a, imm.sigma_j, imm.sigma_v, imm.sigma_omega),
        imm.measurement_noise_sigma,
        imm.gate_threshold,
    );
    run_eval_with_tracker(tracker, targets, config, distance_threshold_m, seed)
}

/// Build a learned-IMM tracker (ONNX mode classifier at `onnx_path`) and run the
/// same evaluation as [`run_eval_harness`]. Each track wraps its IMM bank in a
/// `LearnedImmFilter`, which falls back to the analytic mode update during the
/// warm-up window and on any classifier error.
///
/// # Errors
/// Returns the message if the classifier or IMM config is rejected, or the
/// synth error (stringified) if measurement generation fails.
#[cfg(feature = "learned-imm")]
pub fn run_eval_harness_learned(
    targets: &[TargetTrack],
    config: &TrajectoryRadarConfig,
    imm: ImmTrainingParams,
    onnx_path: impl AsRef<std::path::Path>,
    distance_threshold_m: f64,
    seed: u64,
) -> Result<EvalReport, String> {
    let tracker = MultiObjectTracker::new_imm_position_learned(
        move || ImmConfig::cv_ca_ctrv_ct(imm.sigma_a, imm.sigma_j, imm.sigma_v, imm.sigma_omega),
        onnx_path,
        imm.measurement_noise_sigma,
        imm.gate_threshold,
    )?;
    run_eval_with_tracker(tracker, targets, config, distance_threshold_m, seed)
        .map_err(|e| e.to_string())
}

/// Shared evaluation loop: drive `tracker` over the synth measurement stream and
/// compute MOT metrics (MOTA/MOTP/IDF1) against the ground-truth trajectories.
fn run_eval_with_tracker(
    mut tracker: MultiObjectTracker,
    targets: &[TargetTrack],
    config: &TrajectoryRadarConfig,
    distance_threshold_m: f64,
    seed: u64,
) -> Result<EvalReport, TrajectoryRadarError> {
    let mut rng = StdRng::seed_from_u64(seed);
    let per_tick = measurements_from_trajectory(targets, config, &mut rng)?;
    let dt = 1.0 / config.sample_rate_hz;
    let t_start = targets
        .iter()
        .filter_map(|t| t.waypoints.first())
        .map(|w| w.time)
        .fold(f64::INFINITY, f64::min);

    let mut frames = Vec::with_capacity(per_tick.len());
    for (i, tick) in per_tick.iter().enumerate() {
        let detections: Vec<DVector<f64>> = tick.iter().filter_map(radar_to_detection).collect();
        tracker.step(&detections, dt);
        let t = t_start + i as f64 * dt;

        let gt = ground_truth(targets, config, t);
        let tracks = confirmed_positions(&tracker);
        frames.push(FrameData { gt, tracks });
    }

    Ok(report_from_frames(&frames, distance_threshold_m))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::RefCell;
    use thresh_core::detection::Detection3D;
    use thresh_synth::trajectory::{Segment, SegmentType, Trajectory};

    fn cv_target(pos: [f64; 3], vel: [f64; 3]) -> TargetTrack {
        let waypoints = Trajectory {
            target_id: 0,
            initial_position: pos,
            initial_velocity: vel,
            segments: vec![Segment {
                segment_type: SegmentType::Cv,
                duration: 30.0,
            }],
            dt: 0.1, // matches 10 Hz
        }
        .generate();
        TargetTrack {
            waypoints,
            class_id: 1,
            size_override: None,
        }
    }

    fn config_10hz() -> TrajectoryRadarConfig {
        TrajectoryRadarConfig {
            sample_rate_hz: 10.0,
            ..TrajectoryRadarConfig::default()
        }
    }

    struct RecordingDetector {
        detections: Vec<Detection3D>,
        seen: RefCell<Vec<SensorInput>>,
    }

    impl DetectionPipeline for RecordingDetector {
        fn detect(&self, input: &SensorInput) -> Vec<Detection3D> {
            self.seen.borrow_mut().push(input.clone());
            self.detections.clone()
        }

        fn name(&self) -> &str {
            "recording"
        }
    }

    fn test_tracker() -> MultiObjectTracker {
        let params = ImmTrainingParams::default();
        MultiObjectTracker::new_imm_position(
            move || {
                ImmConfig::cv_ca_ctrv_ct(
                    params.sigma_a,
                    params.sigma_j,
                    params.sigma_v,
                    params.sigma_omega,
                )
            },
            params.measurement_noise_sigma,
            params.gate_threshold,
        )
    }

    #[test]
    fn detector_outputs_drive_metrics_on_identical_observations() {
        let config = config_10hz();
        let target = cv_target([5_000.0, 0.0, 1_000.0], [0.0; 3]);
        let sequence = DetectorEvalSequence::from_trajectories(&[target], &config, 42).unwrap();
        let perfect = RecordingDetector {
            detections: vec![Detection3D {
                position: [5_000.0, 0.0, 1_000.0],
                dimensions: [10.0; 3],
                yaw: 0.0,
                class_id: 1,
                confidence: 1.0,
            }],
            seen: RefCell::default(),
        };
        let empty = RecordingDetector {
            detections: vec![],
            seen: RefCell::default(),
        };
        let good = run_detector_eval(
            &sequence,
            &perfect,
            test_tracker(),
            DEFAULT_DISTANCE_THRESHOLD_M,
        );
        let bad = run_detector_eval(
            &sequence,
            &empty,
            test_tracker(),
            DEFAULT_DISTANCE_THRESHOLD_M,
        );
        assert!(good.mota > 0.95, "perfect detector MOTA: {}", good.mota);
        assert_eq!(bad.mota, 0.0, "empty detector must miss trajectory truth");
        assert!(good.mota > bad.mota);
        assert_eq!(good.frames, sequence.inputs.len());
        assert_eq!(perfect.seen.borrow().len(), good.frames);
        assert_eq!(empty.seen.borrow().len(), bad.frames);
        for (a, b) in perfect.seen.borrow().iter().zip(empty.seen.borrow().iter()) {
            assert_eq!(a.points, b.points);
            assert_eq!(a.intensities, b.intensities);
            assert_eq!(a.timestamp, b.timestamp);
            assert_eq!(a.points.len(), 1000);
        }
    }

    #[test]
    fn detector_truth_keeps_identity_across_target_arrival_and_sensor_offset() {
        let config = TrajectoryRadarConfig {
            sensor: thresh_synth::radar_trajectory::SensorPose {
                position_enu_m: [100.0, 200.0, 300.0],
            },
            ..config_10hz()
        };
        let first = cv_target([1_000.0, 2_000.0, 3_000.0], [0.0; 3]);
        let mut later = cv_target([5_000.0, 6_000.0, 7_000.0], [0.0; 3]);
        for waypoint in &mut later.waypoints {
            waypoint.time += 1.0;
        }
        let sequence =
            DetectorEvalSequence::from_trajectories(&[first, later], &config, 7).unwrap();
        assert_eq!(sequence.truth[0], vec![(0, [900.0, 1_800.0, 2_700.0])]);
        assert_eq!(
            sequence.truth[10],
            vec![
                (0, [900.0, 1_800.0, 2_700.0]),
                (1, [4_900.0, 5_800.0, 6_700.0])
            ]
        );
        assert_eq!(
            sequence.truth.last().unwrap(),
            &vec![(1, [4_900.0, 5_800.0, 6_700.0])]
        );
    }

    #[test]
    fn detector_truth_includes_targets_with_no_sensor_returns() {
        let mut config = config_10hz();
        config.radar.p_detection = 0.0;
        config.radar.radar_equation = None;
        let target = cv_target([5_000.0, 0.0, 1_000.0], [0.0; 3]);
        let sequence = DetectorEvalSequence::from_trajectories(&[target], &config, 42).unwrap();
        assert!(
            sequence
                .truth
                .iter()
                .all(|gt| gt == &vec![(0, [5_000.0, 0.0, 1_000.0])])
        );
        assert!(sequence.inputs.iter().all(|input| {
            input
                .intensities
                .as_ref()
                .unwrap()
                .iter()
                .all(|&intensity| intensity <= config.clutter_intensity_max as f64)
        }));
    }

    #[test]
    fn single_cv_target_tracks_well() {
        let targets = vec![cv_target([5_000.0, 0.0, 1_000.0], [120.0, 0.0, 0.0])];
        let report = run_eval_harness(
            &targets,
            &config_10hz(),
            ImmTrainingParams::default(),
            DEFAULT_DISTANCE_THRESHOLD_M,
            42,
        )
        .expect("eval");

        assert!(report.frames > 100, "should evaluate the full trajectory");
        // A clean single CV target should track accurately after warm-up; a few
        // pre-confirmation frames are false negatives, so allow some slack.
        assert!(
            report.mota > 0.7,
            "MOTA should be high, got {}",
            report.mota
        );
        assert!(
            report.motp <= DEFAULT_DISTANCE_THRESHOLD_M,
            "MOTP within gate, got {}",
            report.motp
        );
        assert!((0.0..=1.0).contains(&report.idf1));
    }

    #[test]
    fn two_separated_targets_keep_distinct_ids() {
        let targets = vec![
            cv_target([5_000.0, 0.0, 1_000.0], [120.0, 0.0, 0.0]),
            cv_target([5_000.0, 3_000.0, 1_000.0], [120.0, 0.0, 0.0]),
        ];
        let report = run_eval_harness(
            &targets,
            &config_10hz(),
            ImmTrainingParams::default(),
            DEFAULT_DISTANCE_THRESHOLD_M,
            7,
        )
        .expect("eval");
        assert!(
            report.mota > 0.6,
            "two well-separated targets, MOTA {}",
            report.mota
        );
    }

    #[test]
    fn radar_to_detection_handles_radar_and_non_radar() {
        let radar = Measurement::Radar {
            range: 1_000.0,
            azimuth: 0.1,
            elevation: 0.05,
            range_rate: None,
            time: 0.0,
            sensor_id: 1,
        };
        assert!(radar_to_detection(&radar).is_some());
        let eoir = Measurement::EoIr {
            azimuth: 0.1,
            elevation: 0.0,
            time: 0.0,
            sensor_id: 1,
        };
        assert!(radar_to_detection(&eoir).is_none());
    }

    #[test]
    fn gt_position_at_handles_edges() {
        let target = cv_target([1_000.0, 0.0, 0.0], [10.0, 0.0, 0.0]); // active 0..30 s
        // In-window, in-range → Some.
        assert!(gt_position_at(&target, 1.0, [0.0, 0.0, 0.0], 1.0e9).is_some());
        // Outside the target's time window → None.
        assert!(gt_position_at(&target, -1.0, [0.0, 0.0, 0.0], 1.0e9).is_none());
        assert!(gt_position_at(&target, 1.0e6, [0.0, 0.0, 0.0], 1.0e9).is_none());
        // Beyond max range → None.
        assert!(gt_position_at(&target, 1.0, [0.0, 0.0, 0.0], 1.0).is_none());
        // Too few waypoints → None.
        let empty = TargetTrack {
            waypoints: vec![],
            class_id: 0,
            size_override: None,
        };
        assert!(gt_position_at(&empty, 1.0, [0.0, 0.0, 0.0], 1.0e9).is_none());
    }
}
