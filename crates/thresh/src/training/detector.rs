//! Track A detector training-sample generation (flight-data-training-pipeline,
//! Phase 7 part 2, task 7.2).
//!
//! Bridges the Phase 4 synth pairing into detector training data: each
//! [`RadarSnapshot`](thresh_synth::radar_trajectory::RadarSnapshot) from
//! `from_trajectory` (a `(point_cloud, gt_boxes, gt_valid, gt_classes)` tuple in
//! the ONNX detector's shapes) becomes one [`DetectorSample`], tagged with its
//! source trajectory. The Rust-native Parquet writer for these lives in
//! [`parquet_export`](super::parquet_export) behind the `training-export`
//! feature; the torch-free `python/training/detector_dataset.py` reads it.

use rand::SeedableRng;
use rand::rngs::StdRng;

use thresh_synth::radar_trajectory::{
    TargetTrack, TrajectoryRadarConfig, TrajectoryRadarError, from_trajectory,
};

/// One detector training example: a synthesised radar snapshot (point cloud +
/// ground-truth boxes / validity / classes, all in the ONNX detector shapes)
/// tagged with its source trajectory and snapshot index.
#[derive(Debug, Clone, PartialEq)]
pub struct DetectorSample {
    /// Source trajectory / scene identifier.
    pub trajectory_id: u32,
    /// Index of this snapshot within the scene's snapshot sequence.
    pub snapshot_index: u64,
    /// Snapshot time (seconds since the trajectory epoch).
    pub time_s: f64,
    /// Flattened point cloud, `1000 × 4` `[x, y, z, intensity]` (4000 floats).
    pub point_cloud: Vec<f32>,
    /// Flattened ground-truth boxes, `100 × 7` `[x, y, z, L, W, H, yaw]` (700 floats).
    pub gt_boxes: Vec<f32>,
    /// Per-box validity mask (length 100).
    pub gt_valid: Vec<bool>,
    /// Per-box class index (length 100).
    pub gt_classes: Vec<i64>,
}

/// Generate detector samples for one scene (a set of co-observed targets) via
/// the Phase 4 synth pairing. Each emitted `RadarSnapshot` becomes one
/// [`DetectorSample`].
///
/// `targets` is the full set of targets in the scene (their union time window
/// drives the snapshot sequence). Determinism is seeded by `seed`.
pub fn generate_detector_samples(
    trajectory_id: u32,
    targets: &[TargetTrack],
    config: &TrajectoryRadarConfig,
    seed: u64,
) -> Result<Vec<DetectorSample>, TrajectoryRadarError> {
    let mut rng = StdRng::seed_from_u64(seed);
    let snapshots = from_trajectory(targets, config, &mut rng)?;
    Ok(snapshots
        .into_iter()
        .enumerate()
        .map(|(i, s)| DetectorSample {
            trajectory_id,
            snapshot_index: i as u64,
            time_s: s.time_s,
            point_cloud: s.point_cloud,
            gt_boxes: s.gt_boxes,
            gt_valid: s.gt_valid,
            gt_classes: s.gt_classes,
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use thresh_synth::trajectory::{Segment, SegmentType, Trajectory};

    fn one_target_scene() -> Vec<TargetTrack> {
        let waypoints = Trajectory {
            target_id: 0,
            initial_position: [5_000.0, 0.0, 1_000.0],
            initial_velocity: [120.0, 0.0, 0.0],
            segments: vec![Segment {
                segment_type: SegmentType::Cv,
                duration: 4.0,
            }],
            dt: 0.5,
        }
        .generate();
        vec![TargetTrack {
            waypoints,
            class_id: 1, // heavy-fixed-wing
            size_override: None,
        }]
    }

    #[test]
    fn generates_onnx_shaped_snapshots() {
        let config = TrajectoryRadarConfig {
            sample_rate_hz: 2.0,
            ..TrajectoryRadarConfig::default()
        };
        let samples = generate_detector_samples(0, &one_target_scene(), &config, 7).unwrap();
        assert!(!samples.is_empty());
        for (i, s) in samples.iter().enumerate() {
            assert_eq!(s.trajectory_id, 0);
            assert_eq!(s.snapshot_index, i as u64);
            assert_eq!(s.point_cloud.len(), 1000 * 4);
            assert_eq!(s.gt_boxes.len(), 100 * 7);
            assert_eq!(s.gt_valid.len(), 100);
            assert_eq!(s.gt_classes.len(), 100);
        }
    }

    #[test]
    fn cv_target_has_a_valid_in_range_box() {
        let config = TrajectoryRadarConfig {
            sample_rate_hz: 2.0,
            ..TrajectoryRadarConfig::default()
        };
        let samples = generate_detector_samples(3, &one_target_scene(), &config, 1).unwrap();
        // At least one snapshot should carry a valid ground-truth box for the
        // (in-range) target, tagged with its class.
        let any_valid = samples
            .iter()
            .any(|s| s.gt_valid.contains(&true) && s.gt_classes.contains(&1));
        assert!(any_valid, "expected a valid class-1 ground-truth box");
    }
}
