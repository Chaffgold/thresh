//! Per-class MOT evaluation (automotive-tracking-pipeline, task 4.1).
//!
//! Follows the nuScenes convention: evaluation is performed **per class** by
//! filtering ground truth and tracks down to one class and running the
//! standard metrics on each filtered stream (filter-then-evaluate), rather
//! than mixing classes inside one association problem. The aggregate report
//! also carries the class-averaged MOTA (the mean over per-class MOTAs, akin
//! to how nuScenes averages its tracking metrics over classes).

use std::collections::BTreeMap;

use thresh_core::track::TargetClass;

use crate::hota::compute_hota_at_threshold;
use crate::matching::FrameData;
use crate::metrics::{compute_idf1, compute_mot_metrics};
use crate::report::{ClassReport, EvalReport};

/// One evaluation frame with class-tagged ground truth and tracks.
#[derive(Debug, Clone, Default)]
pub struct ClassedFrameData {
    /// Ground truth `[(id, [x,y,z], class)]`.
    pub gt: Vec<(u64, [f64; 3], TargetClass)>,
    /// Tracker output `[(id, [x,y,z], class)]`.
    pub tracks: Vec<(u64, [f64; 3], TargetClass)>,
}

impl ClassedFrameData {
    /// Drop the class tags (the aggregate, class-blind view).
    pub fn untagged(&self) -> FrameData {
        FrameData {
            gt: self.gt.iter().map(|&(id, p, _)| (id, p)).collect(),
            tracks: self.tracks.iter().map(|&(id, p, _)| (id, p)).collect(),
        }
    }
}

/// Split classed frames into one class-blind `FrameData` stream per class
/// (every class that appears in either ground truth or tracks gets a stream;
/// frames are kept aligned so temporal metrics like ID switches still work).
pub fn split_frames_by_class(frames: &[ClassedFrameData]) -> BTreeMap<String, Vec<FrameData>> {
    let mut classes: Vec<TargetClass> = frames
        .iter()
        .flat_map(|f| {
            f.gt.iter()
                .map(|&(_, _, c)| c)
                .chain(f.tracks.iter().map(|&(_, _, c)| c))
        })
        .collect();
    classes.sort_by_key(|c| format!("{c:?}"));
    classes.dedup();

    classes
        .into_iter()
        .map(|class| {
            let stream: Vec<FrameData> = frames
                .iter()
                .map(|f| FrameData {
                    gt: f
                        .gt
                        .iter()
                        .filter(|&&(_, _, c)| c == class)
                        .map(|&(id, p, _)| (id, p))
                        .collect(),
                    tracks: f
                        .tracks
                        .iter()
                        .filter(|&&(_, _, c)| c == class)
                        .map(|&(id, p, _)| (id, p))
                        .collect(),
                })
                .collect();
            (format!("{class:?}"), stream)
        })
        .collect()
}

/// Compute per-class MOT metrics (filter-then-evaluate per class).
pub fn compute_per_class_mot(
    frames: &[ClassedFrameData],
    distance_threshold: f64,
) -> Vec<ClassReport> {
    split_frames_by_class(frames)
        .into_iter()
        .map(|(class_name, stream)| {
            let (mota, _motp, _idsw) = compute_mot_metrics(&stream, distance_threshold);
            let (hota, _deta, _assa) = compute_hota_at_threshold(&stream, distance_threshold);
            let count = stream.iter().map(|f| f.gt.len()).sum();
            ClassReport {
                class_name,
                mota,
                hota,
                count,
            }
        })
        .collect()
}

/// Mean of the per-class MOTAs over classes that have ground truth (the
/// class-averaged aggregate used alongside the class-blind overall MOTA).
pub fn class_averaged_mota(per_class: &[ClassReport]) -> f64 {
    let with_gt: Vec<&ClassReport> = per_class.iter().filter(|c| c.count > 0).collect();
    if with_gt.is_empty() {
        return 0.0;
    }
    with_gt.iter().map(|c| c.mota).sum::<f64>() / with_gt.len() as f64
}

/// Build a full [`EvalReport`] from classed frames: class-blind overall
/// metrics plus the per-class breakdown.
pub fn build_classed_report(frames: &[ClassedFrameData], distance_threshold: f64) -> EvalReport {
    let untagged: Vec<FrameData> = frames.iter().map(ClassedFrameData::untagged).collect();
    let (mota, motp, id_switches) = compute_mot_metrics(&untagged, distance_threshold);
    let idf1 = compute_idf1(&untagged, distance_threshold);
    let (hota, _deta, _assa) = compute_hota_at_threshold(&untagged, distance_threshold);
    let per_class = compute_per_class_mot(frames, distance_threshold);
    EvalReport {
        mota,
        motp,
        idf1,
        hota,
        id_switches,
        total_gt: untagged.iter().map(|f| f.gt.len()).sum(),
        total_tracks: untagged.iter().map(|f| f.tracks.len()).sum(),
        per_class,
        gospa: None,
        consistency: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Two cars tracked perfectly; one pedestrian missed entirely.
    fn mixed_frames() -> Vec<ClassedFrameData> {
        (0..10)
            .map(|t| {
                let t = t as f64;
                ClassedFrameData {
                    gt: vec![
                        (1, [t * 10.0, 0.0, 0.0], TargetClass::Car),
                        (2, [t * 10.0, 20.0, 0.0], TargetClass::Car),
                        (3, [5.0, t * 1.0, 0.0], TargetClass::Pedestrian),
                    ],
                    tracks: vec![
                        (101, [t * 10.0, 0.0, 0.0], TargetClass::Car),
                        (102, [t * 10.0, 20.0, 0.0], TargetClass::Car),
                    ],
                }
            })
            .collect()
    }

    #[test]
    fn per_class_separates_perfect_cars_from_missed_pedestrian() {
        let reports = compute_per_class_mot(&mixed_frames(), 1.0);
        assert_eq!(reports.len(), 2);
        let car = reports.iter().find(|c| c.class_name == "Car").unwrap();
        let ped = reports
            .iter()
            .find(|c| c.class_name == "Pedestrian")
            .unwrap();
        assert!((car.mota - 1.0).abs() < 1e-10, "cars perfect: {}", car.mota);
        assert_eq!(car.count, 20);
        assert!(
            (ped.mota - 0.0).abs() < 1e-10,
            "pedestrian missed: {}",
            ped.mota
        );
        assert_eq!(ped.count, 10);
    }

    #[test]
    fn class_blind_overall_differs_from_class_average() {
        let frames = mixed_frames();
        let report = build_classed_report(&frames, 1.0);
        // Overall (class-blind): 10 FN out of 30 GT → MOTA = 2/3.
        assert!((report.mota - 2.0 / 3.0).abs() < 1e-10, "{}", report.mota);
        // Class-averaged: (1.0 + 0.0) / 2 = 0.5.
        let avg = class_averaged_mota(&report.per_class);
        assert!((avg - 0.5).abs() < 1e-10, "{avg}");
        assert_eq!(report.total_gt, 30);
        assert_eq!(report.total_tracks, 20);
    }

    #[test]
    fn cross_class_id_collisions_do_not_leak_between_streams() {
        // The same numeric id used by a Car GT and a Pedestrian GT must stay
        // in separate per-class streams (no spurious matches or switches).
        let frames: Vec<ClassedFrameData> = (0..5)
            .map(|t| {
                let t = t as f64;
                ClassedFrameData {
                    gt: vec![
                        (7, [t * 10.0, 0.0, 0.0], TargetClass::Car),
                        (7, [0.0, t * 1.0, 0.0], TargetClass::Pedestrian),
                    ],
                    tracks: vec![
                        (7, [t * 10.0, 0.0, 0.0], TargetClass::Car),
                        (7, [0.0, t * 1.0, 0.0], TargetClass::Pedestrian),
                    ],
                }
            })
            .collect();
        for report in compute_per_class_mot(&frames, 1.0) {
            assert!(
                (report.mota - 1.0).abs() < 1e-10,
                "{}: {}",
                report.class_name,
                report.mota
            );
        }
    }

    #[test]
    fn class_average_ignores_track_only_classes() {
        // A class that appears only in tracker output (all false positives)
        // has no GT; it must not dilute the class-averaged MOTA.
        let frames = vec![ClassedFrameData {
            gt: vec![(1, [0.0, 0.0, 0.0], TargetClass::Car)],
            tracks: vec![
                (101, [0.0, 0.0, 0.0], TargetClass::Car),
                (201, [50.0, 0.0, 0.0], TargetClass::Bus),
            ],
        }];
        let reports = compute_per_class_mot(&frames, 1.0);
        assert_eq!(reports.len(), 2, "Bus stream still reported");
        let avg = class_averaged_mota(&reports);
        assert!(
            (avg - 1.0).abs() < 1e-10,
            "only Car (with GT) counts: {avg}"
        );
    }
}
