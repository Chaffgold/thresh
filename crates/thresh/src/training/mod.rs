//! Track B training-data generation (flight-data-training-pipeline, Phase 5).
//!
//! This module drives the deployment-shaped pipeline that produces training
//! examples for the IMM mode classifier:
//!
//! ```text
//! canonical trajectory (waypoints)
//!   └─▶ thresh_synth::radar::measurements_from_trajectory   (5.1)
//!         └─▶ MultiObjectTracker (analytic 4-model IMM)      (5.2)
//!               └─▶ project_filter_state  (12-dim feature)   (5.3)
//!                     └─▶ paired with analytic_mode_label    (5.4)
//!                           └─▶ ImmTrainingSample  → Parquet  (5.5)
//! ```
//!
//! Per design Decisions 4 and 5, the classifier is trained on **filter state**
//! (state + covariance produced by running the classical tracker over
//! synthesised measurements), never on raw ADS-B state. Two safeguards enforce
//! this (task 5.7): the tracker is only ever fed sensor-level radar detections
//! ([`ensure_sensor_level`] rejects [`Measurement::AdsB`]), and every feature
//! vector is read from a tracker [`Track`](thresh_tracker::track::Track), not
//! from a measurement.
//!
//! The Parquet writer lives in the `parquet_export` module behind the
//! `training-export` feature so default and CI builds stay free of the
//! `arrow`/`parquet` deps.

use nalgebra::DVector;
use rand::SeedableRng;
use rand::rngs::StdRng;

use thresh_core::coords::spherical_to_cartesian;
use thresh_core::measurement::Measurement;
use thresh_core::motion_mode::MotionModeLabel;
use thresh_core::track::TrackState;
use thresh_filter::imm::{CLASSIFIER_FEATURE_DIM, ImmConfig, project_filter_state};
use thresh_synth::mode_label::analytic_mode_label;
use thresh_synth::radar_trajectory::{
    TargetTrack, TrajectoryRadarConfig, TrajectoryRadarError, measurements_from_trajectory,
};
use thresh_synth::trajectory::Waypoint;
use thresh_tracker::tracker::MultiObjectTracker;

pub mod detector;

#[cfg(feature = "training-export")]
pub mod parquet_export;

/// One training example: a confirmed-track filter-state feature vector paired
/// with the analytic ground-truth motion mode at that timestep.
#[derive(Debug, Clone, PartialEq)]
pub struct ImmTrainingSample {
    /// Source trajectory identifier (lets Phase 6 group windows by trajectory).
    pub trajectory_id: u32,
    /// Tracker track identifier the feature was snapshot from.
    pub track_id: u64,
    /// Tick time (seconds since the trajectory epoch).
    pub time_s: f64,
    /// 12-dim feature `[x, vx, y, vy, z, vz, Pxx, Pvxvx, Pyy, Pvyvy, Pzz, Pvzvz]`.
    pub feature: Vec<f64>,
    /// Analytic mode label derived from trajectory kinematics (the target).
    pub label: MotionModeLabel,
    /// The analytic tracker's own dominant IMM mode at this step, for reference
    /// and eval only — never used as a training label.
    pub imm_dominant_mode: Option<usize>,
}

/// Process-noise and gating parameters for the analytic IMM tracker used to
/// produce training features. Field order mirrors
/// [`ImmConfig::cv_ca_ctrv_ct`](thresh_filter::imm::ImmConfig::cv_ca_ctrv_ct).
#[derive(Debug, Clone, Copy)]
pub struct ImmTrainingParams {
    /// CV acceleration-noise spectral density (m/s²).
    pub sigma_a: f64,
    /// CA jerk-noise spectral density (m/s³).
    pub sigma_j: f64,
    /// CTRV/CT velocity-noise spectral density.
    pub sigma_v: f64,
    /// CTRV/CT turn-rate-noise spectral density.
    pub sigma_omega: f64,
    /// Measurement-noise sigma (m) used to build the tracker's `R`.
    pub measurement_noise_sigma: f64,
    /// Chi-squared gating threshold.
    pub gate_threshold: f64,
}

impl Default for ImmTrainingParams {
    fn default() -> Self {
        Self {
            sigma_a: 5.0,
            sigma_j: 1.0,
            sigma_v: 2.0,
            sigma_omega: 0.1,
            measurement_noise_sigma: 30.0,
            gate_threshold: 100.0,
        }
    }
}

/// Errors from the training-data generation pipeline.
#[derive(Debug)]
pub enum TrainingError {
    /// The synth measurement generator failed (e.g. invalid config or
    /// unsorted waypoints).
    Synth(TrajectoryRadarError),
    /// A raw ADS-B (or otherwise non-sensor-level) measurement reached the
    /// tracker input — forbidden by design Decision 4 (see task 5.7).
    AdsbContamination,
}

impl std::fmt::Display for TrainingError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Synth(e) => write!(f, "synth error: {e}"),
            Self::AdsbContamination => write!(
                f,
                "classifier input integrity violation: a raw ADS-B measurement \
                 reached the tracker input; training features must derive only \
                 from synthesised sensor measurements (design Decision 4)"
            ),
        }
    }
}

impl std::error::Error for TrainingError {}

impl From<TrajectoryRadarError> for TrainingError {
    fn from(e: TrajectoryRadarError) -> Self {
        Self::Synth(e)
    }
}

/// Build-failing guard (task 5.7): assert every measurement destined for the
/// tracker is sensor-level. Raw [`Measurement::AdsB`] is system/track-level
/// truth and must never be fed to the tracker that produces training features.
pub fn ensure_sensor_level(measurements: &[Measurement]) -> Result<(), TrainingError> {
    if measurements
        .iter()
        .any(|m| matches!(m, Measurement::AdsB { .. }))
    {
        return Err(TrainingError::AdsbContamination);
    }
    Ok(())
}

/// Convert a sensor-frame radar measurement `(range, az, el)` into a Cartesian
/// position detection `[x, y, z]` for the position-only IMM tracker. Returns
/// `None` for non-radar measurements (which are not tracker inputs here).
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

/// Run the full Track B pipeline over one canonical trajectory and return the
/// per-step training samples (tasks 5.1–5.4).
///
/// `waypoints` must be sorted by time and sampled at the config's
/// `sample_rate_hz` (so synth tick `i` aligns with waypoint `i`); this is how
/// the in-memory `Trajectory::generate()` output and the acquisition-layer
/// trajectories are produced. Analytic labels are computed from these
/// ground-truth waypoints; features are computed only from confirmed-track
/// filter state.
pub fn generate_imm_training_samples(
    trajectory_id: u32,
    waypoints: &[Waypoint],
    class_id: u32,
    radar_config: &TrajectoryRadarConfig,
    imm: ImmTrainingParams,
    seed: u64,
) -> Result<Vec<ImmTrainingSample>, TrainingError> {
    if waypoints.is_empty() {
        return Err(TrainingError::Synth(TrajectoryRadarError::EmptyTrajectory));
    }

    let target = TargetTrack {
        waypoints: waypoints.to_vec(),
        class_id,
        size_override: None,
    };
    let mut rng = StdRng::seed_from_u64(seed);
    let per_tick =
        measurements_from_trajectory(std::slice::from_ref(&target), radar_config, &mut rng)?;

    let dt = 1.0 / radar_config.sample_rate_hz;
    let t_start = waypoints[0].time;

    let mut tracker = MultiObjectTracker::new_imm_position(
        move || ImmConfig::cv_ca_ctrv_ct(imm.sigma_a, imm.sigma_j, imm.sigma_v, imm.sigma_omega),
        imm.measurement_noise_sigma,
        imm.gate_threshold,
    );

    let mut samples = Vec::new();
    for (i, tick) in per_tick.iter().enumerate() {
        // Task 5.7 guard: only sensor-level measurements reach the tracker.
        ensure_sensor_level(tick)?;

        let detections: Vec<DVector<f64>> = tick.iter().filter_map(radar_to_detection).collect();
        tracker.step(&detections, dt);

        let time_s = t_start + i as f64 * dt;
        let label = analytic_mode_label(waypoints, i.min(waypoints.len() - 1));

        for track in &tracker.tracks {
            if track.lifecycle != TrackState::Confirmed {
                continue;
            }
            // Feature is read from filter state — never from a measurement.
            let feature = project_filter_state(&track.state, &track.covariance);
            debug_assert_eq!(feature.len(), CLASSIFIER_FEATURE_DIM);
            samples.push(ImmTrainingSample {
                trajectory_id,
                track_id: track.id.0,
                time_s,
                feature,
                label,
                imm_dominant_mode: track.dominant_mode,
            });
        }
    }

    Ok(samples)
}

#[cfg(test)]
mod tests {
    use super::*;
    use thresh_synth::trajectory::{Segment, SegmentType, Trajectory};

    /// 10 Hz config with the target a few km from the origin sensor so it is
    /// comfortably in range and reliably detected.
    fn config_10hz() -> TrajectoryRadarConfig {
        TrajectoryRadarConfig {
            sample_rate_hz: 10.0,
            ..TrajectoryRadarConfig::default()
        }
    }

    fn cv_trajectory() -> Vec<Waypoint> {
        Trajectory {
            target_id: 0,
            initial_position: [5_000.0, 0.0, 1_000.0],
            initial_velocity: [120.0, 0.0, 0.0],
            segments: vec![Segment {
                segment_type: SegmentType::Cv,
                duration: 20.0,
            }],
            dt: 0.1, // matches 10 Hz
        }
        .generate()
    }

    /// Task 5.2/5.3: the pipeline produces confirmed-track samples whose
    /// features are 12-dim filter state (positive covariance diagonal proves
    /// the value came through the filter, not from a raw measurement).
    #[test]
    fn cv_trajectory_produces_filter_state_samples() {
        let wps = cv_trajectory();
        let samples = generate_imm_training_samples(
            7,
            &wps,
            1,
            &config_10hz(),
            ImmTrainingParams::default(),
            42,
        )
        .expect("pipeline should succeed");

        assert!(
            !samples.is_empty(),
            "a confirmed track should yield samples"
        );
        for s in &samples {
            assert_eq!(s.feature.len(), CLASSIFIER_FEATURE_DIM);
            assert_eq!(s.trajectory_id, 7);
            // Covariance-diagonal entries (indices 6..12) are strictly positive:
            // a raw ADS-B/world state would have no such filter covariance.
            assert!(
                s.feature[6..12].iter().all(|&v| v > 0.0),
                "covariance diagonal must be positive (proves filter output): {:?}",
                &s.feature[6..12]
            );
        }
    }

    /// Task 5.6 (end-to-end): a straight-line target is labelled overwhelmingly
    /// `CV` through the full synth → tracker → label pipeline.
    #[test]
    fn cv_trajectory_labels_are_mostly_cv() {
        let wps = cv_trajectory();
        let samples = generate_imm_training_samples(
            0,
            &wps,
            1,
            &config_10hz(),
            ImmTrainingParams::default(),
            7,
        )
        .unwrap();
        let cv = samples
            .iter()
            .filter(|s| s.label == MotionModeLabel::Cv)
            .count();
        let frac = cv as f64 / samples.len() as f64;
        assert!(
            frac > 0.85,
            "straight-line target should be mostly CV, got {frac:.2}"
        );
    }

    /// Task 5.7: the build-failing guard rejects raw ADS-B contamination and
    /// accepts sensor-level radar.
    #[test]
    fn guard_rejects_adsb_but_accepts_radar() {
        let radar = Measurement::Radar {
            range: 1_000.0,
            azimuth: 0.1,
            elevation: 0.05,
            range_rate: None,
            time: 0.0,
            sensor_id: 1,
        };
        let adsb = Measurement::AdsB {
            lat: 47.6,
            lon: -122.3,
            alt: 1_000.0,
            velocity: None,
            time: 0.0,
        };
        assert!(ensure_sensor_level(std::slice::from_ref(&radar)).is_ok());
        assert!(matches!(
            ensure_sensor_level(&[radar, adsb]),
            Err(TrainingError::AdsbContamination)
        ));
    }

    #[test]
    fn training_error_displays_both_arms() {
        // `from` exercises the `From<TrajectoryRadarError>` impl; Display
        // covers the `Synth` arm (which delegates to the inner error).
        let synth = TrainingError::from(TrajectoryRadarError::InvalidSampleRate(0.0));
        assert!(format!("{synth}").contains("synth error"));
        let contam = TrainingError::AdsbContamination;
        assert!(format!("{contam}").contains("integrity violation"));
    }

    #[test]
    fn radar_to_detection_ignores_non_radar() {
        let eoir = Measurement::EoIr {
            azimuth: 0.1,
            elevation: 0.0,
            time: 0.0,
            sensor_id: 1,
        };
        assert!(radar_to_detection(&eoir).is_none());
        let radar = Measurement::Radar {
            range: 100.0,
            azimuth: 0.0,
            elevation: 0.0,
            range_rate: None,
            time: 0.0,
            sensor_id: 1,
        };
        assert!(radar_to_detection(&radar).is_some());
    }

    #[test]
    fn invalid_sample_rate_propagates_synth_error() {
        let wps = cv_trajectory();
        let bad = TrajectoryRadarConfig {
            sample_rate_hz: 0.0,
            ..TrajectoryRadarConfig::default()
        };
        let err = generate_imm_training_samples(0, &wps, 1, &bad, ImmTrainingParams::default(), 1);
        assert!(matches!(err, Err(TrainingError::Synth(_))));
    }

    #[test]
    fn empty_trajectory_is_an_error() {
        let err = generate_imm_training_samples(
            0,
            &[],
            0,
            &config_10hz(),
            ImmTrainingParams::default(),
            1,
        );
        assert!(matches!(
            err,
            Err(TrainingError::Synth(TrajectoryRadarError::EmptyTrajectory))
        ));
    }
}
