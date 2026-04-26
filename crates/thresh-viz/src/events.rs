//! Track lifecycle event derivation from snapshot diffs.
//!
//! `MultiObjectTracker` does not currently emit explicit "born" / "died"
//! events — that information is implicit in the diff between consecutive
//! snapshots. This module computes the diff GUI-side so the dashboard
//! can display a lifecycle event log without requiring tracker changes.
//!
//! Births and deaths are exact (set difference on track IDs). ID
//! switches are heuristic: when one ID disappears and another appears in
//! the same timestep within `id_switch_position_tolerance` distance, we
//! report it as a switch rather than as separate death + birth events.

use crate::recording::{LifecycleEvent, VizFrame};

/// Tolerance below which a death + birth pair is treated as an ID switch.
/// Default chosen for typical aerospace scenarios where 5m is well within
/// per-frame motion of fast targets.
pub const DEFAULT_ID_SWITCH_TOLERANCE_METERS: f64 = 5.0;

/// Derive lifecycle events from the diff of two consecutive frames.
///
/// `prev` is the earlier frame; `next` is the later one. Returns events
/// in deterministic order: id-switches first (sorted by `from` ID), then
/// births (sorted by ID), then deaths (sorted by ID).
pub fn diff_snapshots(prev: &VizFrame, next: &VizFrame) -> Vec<LifecycleEvent> {
    diff_snapshots_with_tolerance(prev, next, DEFAULT_ID_SWITCH_TOLERANCE_METERS)
}

/// Like [`diff_snapshots`] but with a caller-specified ID-switch position
/// tolerance.
pub fn diff_snapshots_with_tolerance(
    prev: &VizFrame,
    next: &VizFrame,
    tolerance: f64,
) -> Vec<LifecycleEvent> {
    use std::collections::BTreeMap;

    let prev_by_id: BTreeMap<u64, [f64; 3]> =
        prev.tracks.iter().map(|t| (t.id, t.position)).collect();
    let next_by_id: BTreeMap<u64, [f64; 3]> =
        next.tracks.iter().map(|t| (t.id, t.position)).collect();

    let disappeared: Vec<(u64, [f64; 3])> = prev_by_id
        .iter()
        .filter(|(id, _)| !next_by_id.contains_key(id))
        .map(|(id, pos)| (*id, *pos))
        .collect();
    let appeared: Vec<(u64, [f64; 3])> = next_by_id
        .iter()
        .filter(|(id, _)| !prev_by_id.contains_key(id))
        .map(|(id, pos)| (*id, *pos))
        .collect();

    let mut id_switches: Vec<(u64, u64)> = Vec::new();
    let mut paired_disappeared: Vec<bool> = vec![false; disappeared.len()];
    let mut paired_appeared: Vec<bool> = vec![false; appeared.len()];

    // Match disappeared to appeared within the tolerance distance — earliest
    // disappeared ID gets first pick of the closest appeared ID, breaking
    // ties by appeared ID for determinism.
    for (i, (from_id, from_pos)) in disappeared.iter().enumerate() {
        let mut best: Option<(usize, f64)> = None;
        for (j, (_, to_pos)) in appeared.iter().enumerate() {
            if paired_appeared[j] {
                continue;
            }
            let d = euclidean(*from_pos, *to_pos);
            if d <= tolerance && best.is_none_or(|(_, bd)| d < bd) {
                best = Some((j, d));
            }
        }
        if let Some((j, _)) = best {
            id_switches.push((*from_id, appeared[j].0));
            paired_disappeared[i] = true;
            paired_appeared[j] = true;
        }
    }

    let mut events: Vec<LifecycleEvent> = Vec::new();
    id_switches.sort_unstable();
    for (from, to) in id_switches {
        events.push(LifecycleEvent::IdSwitched { from, to });
    }
    let mut births: Vec<u64> = appeared
        .iter()
        .enumerate()
        .filter_map(|(j, (id, _))| (!paired_appeared[j]).then_some(*id))
        .collect();
    births.sort_unstable();
    for id in births {
        events.push(LifecycleEvent::Born { id });
    }
    let mut deaths: Vec<u64> = disappeared
        .iter()
        .enumerate()
        .filter_map(|(i, (id, _))| (!paired_disappeared[i]).then_some(*id))
        .collect();
    deaths.sort_unstable();
    for id in deaths {
        events.push(LifecycleEvent::Died { id });
    }
    events
}

fn euclidean(a: [f64; 3], b: [f64; 3]) -> f64 {
    ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::recording::{VizDetection, VizGroundTruth, VizTrack};

    fn track(id: u64, x: f64) -> VizTrack {
        VizTrack {
            id,
            position: [x, 0.0, 0.0],
            velocity: [0.0; 3],
            covariance_diag: [1.0; 6],
            is_confirmed: true,
            class_label: None,
        }
    }

    fn frame(timestamp: f64, tracks: Vec<VizTrack>) -> VizFrame {
        VizFrame {
            timestamp,
            tracks,
            detections: Vec::<VizDetection>::new(),
            ground_truth: Vec::<VizGroundTruth>::new(),
            associations: Vec::new(),
            events: Vec::new(),
        }
    }

    #[test]
    fn births_only_when_only_new_ids_appear() {
        let prev = frame(0.0, vec![track(1, 0.0)]);
        let next = frame(1.0, vec![track(1, 0.0), track(2, 100.0), track(3, 200.0)]);
        let events = diff_snapshots(&prev, &next);
        assert_eq!(
            events,
            vec![
                LifecycleEvent::Born { id: 2 },
                LifecycleEvent::Born { id: 3 },
            ]
        );
    }

    #[test]
    fn deaths_only_when_only_old_ids_disappear() {
        let prev = frame(0.0, vec![track(1, 0.0), track(2, 50.0), track(3, 100.0)]);
        let next = frame(1.0, vec![track(1, 0.0)]);
        let events = diff_snapshots(&prev, &next);
        assert_eq!(
            events,
            vec![
                LifecycleEvent::Died { id: 2 },
                LifecycleEvent::Died { id: 3 },
            ]
        );
    }

    #[test]
    fn id_switch_pairs_close_death_with_close_birth() {
        // Track 5 disappears at x=100, track 9 appears at x=101 — within tolerance.
        let prev = frame(0.0, vec![track(5, 100.0)]);
        let next = frame(1.0, vec![track(9, 101.0)]);
        let events = diff_snapshots(&prev, &next);
        assert_eq!(events, vec![LifecycleEvent::IdSwitched { from: 5, to: 9 }]);
    }

    #[test]
    fn id_switch_does_not_pair_far_appearance() {
        // Track 5 disappears at x=100, track 9 appears at x=1000 — too far.
        let prev = frame(0.0, vec![track(5, 100.0)]);
        let next = frame(1.0, vec![track(9, 1000.0)]);
        let events = diff_snapshots(&prev, &next);
        assert_eq!(
            events,
            vec![
                LifecycleEvent::Born { id: 9 },
                LifecycleEvent::Died { id: 5 },
            ]
        );
    }

    #[test]
    fn no_events_when_nothing_changes() {
        let prev = frame(0.0, vec![track(1, 10.0), track(2, 20.0)]);
        let next = frame(1.0, vec![track(1, 11.0), track(2, 19.0)]);
        let events = diff_snapshots(&prev, &next);
        assert!(events.is_empty());
    }

    #[test]
    fn deterministic_order_across_runs() {
        let prev = frame(
            0.0,
            vec![
                track(10, 0.0),
                track(20, 100.0),
                track(30, 200.0),
                track(40, 300.0),
            ],
        );
        let next = frame(
            1.0,
            vec![
                track(10, 0.0),
                track(50, 101.0), // matches 20 within tolerance
                track(60, 305.0), // matches 40 within tolerance
                track(70, 500.0), // pure birth
            ],
        );

        let a = diff_snapshots(&prev, &next);
        let b = diff_snapshots(&prev, &next);
        assert_eq!(a, b);

        // Expected order: id-switches sorted by `from`, then births, then deaths.
        assert_eq!(
            a,
            vec![
                LifecycleEvent::IdSwitched { from: 20, to: 50 },
                LifecycleEvent::IdSwitched { from: 40, to: 60 },
                LifecycleEvent::Born { id: 70 },
                LifecycleEvent::Died { id: 30 },
            ]
        );
    }
}
