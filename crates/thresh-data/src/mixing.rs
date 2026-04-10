//! Dataset mixing and temporal bucketing.

use crate::dataset::{CoordinateFrame, Dataset, DatasetMetadata};
use crate::frame::Frame;

/// Combines multiple datasets into a single time-ordered stream.
pub struct MixedDataset {
    datasets: Vec<Box<dyn Dataset>>,
    name: String,
}

impl MixedDataset {
    /// Create a mixed dataset from multiple sources.
    pub fn new(name: String, datasets: Vec<Box<dyn Dataset>>) -> Self {
        Self { datasets, name }
    }
}

impl Dataset for MixedDataset {
    fn metadata(&self) -> DatasetMetadata {
        let mut target_count: Option<usize> = None;
        let mut time_start = f64::INFINITY;
        let mut time_end = f64::NEG_INFINITY;
        let mut has_time_span = false;

        for ds in &self.datasets {
            let meta = ds.metadata();
            if let Some(tc) = meta.target_count {
                *target_count.get_or_insert(0) += tc;
            }
            if let Some((start, end)) = meta.time_span {
                has_time_span = true;
                if start < time_start {
                    time_start = start;
                }
                if end > time_end {
                    time_end = end;
                }
            }
        }

        DatasetMetadata {
            name: self.name.clone(),
            source: "mixed".to_string(),
            target_count,
            time_span: if has_time_span {
                Some((time_start, time_end))
            } else {
                None
            },
            coordinate_frame: CoordinateFrame::Enu,
        }
    }

    fn frames(&self) -> Box<dyn Iterator<Item = Frame> + '_> {
        Box::new(MergeIter::new(
            self.datasets.iter().map(|ds| ds.frames()).collect(),
        ))
    }

    fn ground_truth(&self) -> Option<Box<dyn Iterator<Item = Frame> + '_>> {
        let iters: Vec<_> = self
            .datasets
            .iter()
            .filter_map(|ds| ds.ground_truth())
            .collect();

        if iters.is_empty() {
            None
        } else {
            Some(Box::new(MergeIter::new(iters)))
        }
    }
}

/// A lazy k-way merge iterator that pulls from multiple sorted iterators
/// in timestamp order.
struct MergeIter<'a> {
    /// Each source is an iterator with an optional peeked value.
    sources: Vec<(Box<dyn Iterator<Item = Frame> + 'a>, Option<Frame>)>,
}

impl<'a> MergeIter<'a> {
    fn new(iters: Vec<Box<dyn Iterator<Item = Frame> + 'a>>) -> Self {
        let sources = iters
            .into_iter()
            .map(|mut it| {
                let peeked = it.next();
                (it, peeked)
            })
            .collect();
        Self { sources }
    }
}

impl Iterator for MergeIter<'_> {
    type Item = Frame;

    fn next(&mut self) -> Option<Frame> {
        // Find the source with the smallest peeked timestamp.
        let mut best_idx: Option<usize> = None;
        let mut best_ts = f64::INFINITY;

        for (i, (_, peeked)) in self.sources.iter().enumerate() {
            if let Some(frame) = peeked
                && frame.timestamp < best_ts
            {
                best_ts = frame.timestamp;
                best_idx = Some(i);
            }
        }

        let idx = best_idx?;
        let (ref mut iter, ref mut peeked) = self.sources[idx];
        let result = peeked.take();
        *peeked = iter.next();
        result
    }
}

/// Group frames into time buckets of `window` seconds.
///
/// Measurements within `window` seconds of each other are merged into a
/// single frame. The bucket's timestamp is the midpoint of the bucket
/// interval. Ground truth entries from all merged frames are combined.
pub fn bucket_frames(frames: Vec<Frame>, window: f64) -> Vec<Frame> {
    if frames.is_empty() {
        return Vec::new();
    }

    // Sort by timestamp first.
    let mut frames = frames;
    frames.sort_by(|a, b| a.timestamp.partial_cmp(&b.timestamp).unwrap());

    let t_start = frames[0].timestamp;
    let mut buckets: std::collections::BTreeMap<i64, Vec<Frame>> =
        std::collections::BTreeMap::new();

    for frame in frames {
        let idx = ((frame.timestamp - t_start) / window).floor() as i64;
        buckets.entry(idx).or_default().push(frame);
    }

    buckets
        .into_iter()
        .map(|(idx, bucket_frames)| {
            let timestamp = t_start + (idx as f64 + 0.5) * window;

            let mut measurements = Vec::new();
            let mut ground_truth: Vec<crate::frame::GroundTruthEntry> = Vec::new();
            let mut sensor_metadata = None;

            for f in bucket_frames {
                measurements.extend(f.measurements);
                if let Some(gt) = f.ground_truth {
                    ground_truth.extend(gt);
                }
                if sensor_metadata.is_none() {
                    sensor_metadata = f.sensor_metadata;
                }
            }

            Frame {
                timestamp,
                measurements,
                ground_truth: if ground_truth.is_empty() {
                    None
                } else {
                    Some(ground_truth)
                },
                sensor_metadata,
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::frame::{Frame, GroundTruthEntry};
    use crate::synthetic::SyntheticDataset;

    fn make_frame(timestamp: f64, n_meas: usize, target_id: Option<u64>) -> Frame {
        use thresh_core::measurement::Measurement;
        let measurements = (0..n_meas)
            .map(|_| Measurement::Radar {
                range: 1000.0,
                azimuth: 0.5,
                elevation: 0.1,
                range_rate: None,
                time: timestamp,
                sensor_id: 0,
            })
            .collect();

        let ground_truth = target_id.map(|id| {
            vec![GroundTruthEntry {
                target_id: id,
                position: [1.0, 2.0, 3.0],
                velocity: None,
                class: None,
            }]
        });

        Frame {
            timestamp,
            measurements,
            ground_truth,
            sensor_metadata: None,
        }
    }

    #[test]
    fn mixed_dataset_merges_in_time_order() {
        let ds1 = SyntheticDataset::from_frames(
            "ds1".into(),
            vec![make_frame(0.0, 1, Some(1)), make_frame(0.2, 1, Some(1))],
        );
        let ds2 = SyntheticDataset::from_frames(
            "ds2".into(),
            vec![make_frame(0.1, 1, Some(2)), make_frame(0.3, 1, Some(2))],
        );

        let mixed = MixedDataset::new("merged".into(), vec![Box::new(ds1), Box::new(ds2)]);

        let frames: Vec<Frame> = mixed.frames().collect();
        assert_eq!(frames.len(), 4);
        assert!((frames[0].timestamp - 0.0).abs() < 1e-9);
        assert!((frames[1].timestamp - 0.1).abs() < 1e-9);
        assert!((frames[2].timestamp - 0.2).abs() < 1e-9);
        assert!((frames[3].timestamp - 0.3).abs() < 1e-9);
    }

    #[test]
    fn temporal_bucketing_groups_within_window() {
        // 4 frames at 0ms, 10ms, 20ms, 100ms with a 50ms window
        let frames = vec![
            make_frame(0.000, 1, Some(1)),
            make_frame(0.010, 1, Some(2)),
            make_frame(0.020, 1, Some(3)),
            make_frame(0.100, 1, Some(4)),
        ];

        let bucketed = bucket_frames(frames, 0.050);
        // First bucket: [0.0, 0.05) contains 0ms, 10ms, 20ms
        // Second bucket: [0.05, 0.10) empty
        // Third bucket: [0.10, 0.15) contains 100ms
        assert_eq!(bucketed.len(), 2);
        assert_eq!(bucketed[0].measurements.len(), 3);
        assert_eq!(bucketed[1].measurements.len(), 1);

        // Ground truth entries are combined in first bucket.
        let gt = bucketed[0].ground_truth.as_ref().unwrap();
        assert_eq!(gt.len(), 3);
    }

    #[test]
    fn frame_ordering_after_mixing() {
        let ds1 = SyntheticDataset::from_frames(
            "a".into(),
            vec![
                make_frame(1.0, 1, None),
                make_frame(3.0, 1, None),
                make_frame(5.0, 1, None),
            ],
        );
        let ds2 = SyntheticDataset::from_frames(
            "b".into(),
            vec![make_frame(2.0, 1, None), make_frame(4.0, 1, None)],
        );

        let mixed = MixedDataset::new("test".into(), vec![Box::new(ds1), Box::new(ds2)]);
        let timestamps: Vec<f64> = mixed.frames().map(|f| f.timestamp).collect();
        assert_eq!(timestamps, vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    }

    #[test]
    fn bucket_counts_are_correct() {
        // 10 frames at 0.0, 0.01, 0.02, ..., 0.09 with window 0.05
        let frames: Vec<Frame> = (0..10)
            .map(|i| make_frame(i as f64 * 0.01, 1, None))
            .collect();

        let bucketed = bucket_frames(frames, 0.05);
        assert_eq!(bucketed.len(), 2);
        assert_eq!(bucketed[0].measurements.len(), 5); // 0.00..0.04
        assert_eq!(bucketed[1].measurements.len(), 5); // 0.05..0.09
    }

    #[test]
    fn metadata_combines_correctly() {
        let ds1 = SyntheticDataset {
            name: "ds1".into(),
            frames: vec![make_frame(0.0, 1, None), make_frame(1.0, 1, None)],
        };
        let ds2 = SyntheticDataset {
            name: "ds2".into(),
            frames: vec![make_frame(0.5, 1, None), make_frame(2.0, 1, None)],
        };

        let mixed = MixedDataset::new("combined".into(), vec![Box::new(ds1), Box::new(ds2)]);
        let meta = mixed.metadata();

        assert_eq!(meta.name, "combined");
        assert_eq!(meta.source, "mixed");
        let (start, end) = meta.time_span.unwrap();
        assert!((start - 0.0).abs() < 1e-9);
        assert!((end - 2.0).abs() < 1e-9);
    }
}
