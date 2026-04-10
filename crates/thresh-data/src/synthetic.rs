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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::frame::GroundTruthEntry;
    use thresh_core::measurement::Measurement;

    fn radar_measurement(time: f64) -> Measurement {
        Measurement::Radar {
            range: 10000.0,
            azimuth: 0.5,
            elevation: 0.1,
            range_rate: None,
            time,
            sensor_id: 0,
        }
    }

    fn gt_entry(id: u64) -> GroundTruthEntry {
        GroundTruthEntry {
            target_id: id,
            position: [100.0, 200.0, 300.0],
            velocity: Some([10.0, 20.0, 0.0]),
            class: None,
        }
    }

    fn make_frame(timestamp: f64, gt: Option<Vec<GroundTruthEntry>>) -> Frame {
        Frame {
            timestamp,
            measurements: vec![radar_measurement(timestamp)],
            ground_truth: gt,
            sensor_metadata: None,
        }
    }

    #[test]
    fn from_frames_sorts_by_timestamp() {
        let ds = SyntheticDataset::from_frames(
            "test".into(),
            vec![
                make_frame(3.0, None),
                make_frame(1.0, None),
                make_frame(2.0, None),
            ],
        );
        let timestamps: Vec<f64> = ds.frames().map(|f| f.timestamp).collect();
        assert_eq!(timestamps, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn from_measurements_buckets_correctly() {
        let measurements = vec![
            (0.0, radar_measurement(0.0)),
            (0.5, radar_measurement(0.5)),
            (1.0, radar_measurement(1.0)),
            (1.3, radar_measurement(1.3)),
        ];
        let ground_truth = vec![(0.0, gt_entry(1)), (1.0, gt_entry(2))];

        let ds =
            SyntheticDataset::from_measurements("test".into(), measurements, ground_truth, 1.0);
        let frames: Vec<Frame> = ds.frames().collect();

        assert_eq!(frames.len(), 2);
        // Bucket 0: t=0.0, t=0.5 → 2 measurements
        assert_eq!(frames[0].measurements.len(), 2);
        // Bucket 1: t=1.0, t=1.3 → 2 measurements
        assert_eq!(frames[1].measurements.len(), 2);
        // GT: bucket 0 has target 1, bucket 1 has target 2
        assert_eq!(frames[0].ground_truth.as_ref().unwrap().len(), 1);
        assert_eq!(frames[1].ground_truth.as_ref().unwrap().len(), 1);
    }

    #[test]
    fn from_measurements_empty_gt_is_none() {
        let measurements = vec![(0.0, radar_measurement(0.0))];
        let ds = SyntheticDataset::from_measurements("test".into(), measurements, vec![], 1.0);
        let frames: Vec<Frame> = ds.frames().collect();
        assert_eq!(frames.len(), 1);
        assert!(frames[0].ground_truth.is_none());
    }

    #[test]
    fn metadata_returns_correct_values() {
        let ds = SyntheticDataset::from_frames(
            "my_scenario".into(),
            vec![make_frame(1.0, None), make_frame(5.0, None)],
        );
        let meta = ds.metadata();
        assert_eq!(meta.name, "my_scenario");
        assert_eq!(meta.source, "synthetic");
        assert!(meta.target_count.is_none());
        let (start, end) = meta.time_span.unwrap();
        assert!((start - 1.0).abs() < 1e-10);
        assert!((end - 5.0).abs() < 1e-10);
    }

    #[test]
    fn metadata_empty_dataset() {
        let ds = SyntheticDataset::from_frames("empty".into(), vec![]);
        let meta = ds.metadata();
        assert!(meta.time_span.is_none());
    }

    #[test]
    fn ground_truth_returns_only_gt_frames() {
        let ds = SyntheticDataset::from_frames(
            "test".into(),
            vec![
                make_frame(0.0, Some(vec![gt_entry(1)])),
                make_frame(1.0, None),
                make_frame(2.0, Some(vec![gt_entry(2)])),
            ],
        );
        let gt: Vec<Frame> = ds.ground_truth().unwrap().collect();
        assert_eq!(gt.len(), 2);
        assert!((gt[0].timestamp - 0.0).abs() < 1e-10);
        assert!((gt[1].timestamp - 2.0).abs() < 1e-10);
    }

    #[test]
    fn ground_truth_returns_none_when_no_gt() {
        let ds = SyntheticDataset::from_frames(
            "test".into(),
            vec![make_frame(0.0, None), make_frame(1.0, None)],
        );
        assert!(ds.ground_truth().is_none());
    }

    #[test]
    fn frames_iterator_yields_all() {
        let ds = SyntheticDataset::from_frames(
            "test".into(),
            vec![
                make_frame(0.0, None),
                make_frame(1.0, None),
                make_frame(2.0, None),
            ],
        );
        assert_eq!(ds.frames().count(), 3);
    }
}
