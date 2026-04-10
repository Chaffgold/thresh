//! Synthetic dataset adapter for pre-generated data.

use std::collections::BTreeMap;

use crate::dataset::{CoordinateFrame, Dataset, DatasetMetadata};
use crate::frame::{Frame, GroundTruthEntry};
use thresh_core::measurement::Measurement;

/// A dataset built from pre-generated synthetic data.
pub struct SyntheticDataset {
    /// Human-readable name for this dataset.
    pub name: String,
    /// Pre-loaded frames sorted by timestamp.
    pub frames: Vec<Frame>,
}

impl SyntheticDataset {
    /// Create from a list of pre-built frames.
    pub fn from_frames(name: String, frames: Vec<Frame>) -> Self {
        let mut dataset = Self { name, frames };
        dataset
            .frames
            .sort_by(|a, b| a.timestamp.partial_cmp(&b.timestamp).unwrap());
        dataset
    }

    /// Create from raw measurements and ground truth, bucketed by time step.
    ///
    /// Measurements and ground truth entries are grouped into frames using
    /// `dt` as the bucket width. Each bucket's timestamp is the midpoint of
    /// the bucket interval.
    pub fn from_measurements(
        name: String,
        measurements: Vec<(f64, Measurement)>,
        ground_truth: Vec<(f64, GroundTruthEntry)>,
        dt: f64,
    ) -> Self {
        // Bucket index -> (measurements, ground_truth)
        let mut buckets: BTreeMap<i64, (Vec<Measurement>, Vec<GroundTruthEntry>)> = BTreeMap::new();

        for (t, m) in measurements {
            let idx = (t / dt).floor() as i64;
            buckets.entry(idx).or_default().0.push(m);
        }

        for (t, gt) in ground_truth {
            let idx = (t / dt).floor() as i64;
            buckets.entry(idx).or_default().1.push(gt);
        }

        let frames = buckets
            .into_iter()
            .map(|(idx, (meas, gt))| {
                let timestamp = (idx as f64 + 0.5) * dt;
                Frame {
                    timestamp,
                    measurements: meas,
                    ground_truth: if gt.is_empty() { None } else { Some(gt) },
                    sensor_metadata: None,
                }
            })
            .collect();

        Self { name, frames }
    }
}

impl Dataset for SyntheticDataset {
    fn metadata(&self) -> DatasetMetadata {
        let time_span = if self.frames.is_empty() {
            None
        } else {
            Some((
                self.frames.first().unwrap().timestamp,
                self.frames.last().unwrap().timestamp,
            ))
        };

        DatasetMetadata {
            name: self.name.clone(),
            source: "synthetic".to_string(),
            target_count: None,
            time_span,
            coordinate_frame: CoordinateFrame::Enu,
        }
    }

    fn frames(&self) -> Box<dyn Iterator<Item = Frame> + '_> {
        Box::new(self.frames.iter().cloned())
    }

    fn ground_truth(&self) -> Option<Box<dyn Iterator<Item = Frame> + '_>> {
        let gt_frames: Vec<Frame> = self
            .frames
            .iter()
            .filter(|f| f.ground_truth.is_some())
            .cloned()
            .collect();

        if gt_frames.is_empty() {
            None
        } else {
            Some(Box::new(gt_frames.into_iter()))
        }
    }
}
