//! End-to-end automotive tracking over synthetic road scenarios
//! (automotive-tracking-pipeline, Phase 3 wiring Phase 2's tracker entry).
//!
//! Drives `MultiObjectTracker::new_automotive_enu` over the `intersection`
//! preset: once with world-frame detections (`step_classed`) and once with the
//! same scene observed from a moving ego (`step_with_ego`), asserting the
//! tracker holds one confirmed track per agent and the two views agree.

use nalgebra::DVector;
use thresh::core::ego::{EgoMotion, EgoPose};
use thresh::core::track::{TargetClass, TrackState};
use thresh::synth::road_scenarios::{ROAD_DT_S, RoadAgent, intersection};
use thresh::synth::trajectory::Waypoint;
use thresh::tracker::tracker::MultiObjectTracker;

/// Ground-truth (position, class) per agent at tick `k`, while the agent's
/// trajectory still has samples.
fn detections_at(
    agents: &[(RoadAgent, Vec<Waypoint>)],
    k: usize,
) -> Vec<(DVector<f64>, TargetClass)> {
    agents
        .iter()
        .filter_map(|(agent, wps)| {
            wps.get(k)
                .map(|w| (DVector::from_column_slice(&w.position), agent.class))
        })
        .collect()
}

fn scenario() -> (Vec<(RoadAgent, Vec<Waypoint>)>, usize) {
    let agents: Vec<(RoadAgent, Vec<Waypoint>)> = intersection()
        .into_iter()
        .map(|a| {
            let wps = a.waypoints();
            (a, wps)
        })
        .collect();
    let ticks = agents.iter().map(|(_, w)| w.len()).min().unwrap();
    (agents, ticks)
}

#[test]
fn intersection_scene_is_fully_tracked() {
    let (agents, ticks) = scenario();
    let mut tracker = MultiObjectTracker::new_automotive_enu(0.5, 16.0);
    for k in 0..ticks {
        tracker.step_classed(&detections_at(&agents, k), ROAD_DT_S);
    }
    assert_eq!(
        tracker.confirmed_count(),
        agents.len(),
        "every intersection agent should be a confirmed track"
    );
    // Classes survive birth-through-confirmation.
    let mut tracked: Vec<TargetClass> = tracker
        .tracks
        .iter()
        .filter(|t| t.lifecycle == TrackState::Confirmed)
        .map(|t| t.class)
        .collect();
    tracked.sort_by_key(|c| format!("{c:?}"));
    let mut expected: Vec<TargetClass> = agents.iter().map(|(a, _)| a.class).collect();
    expected.sort_by_key(|c| format!("{c:?}"));
    assert_eq!(tracked, expected);
}

#[test]
fn moving_ego_view_matches_world_view() {
    let (agents, ticks) = scenario();

    // World-frame reference run.
    let mut world_tracker = MultiObjectTracker::new_automotive_enu(0.5, 16.0);
    // Ego run: the same scene observed from a platform driving through it.
    let mut ego_tracker = MultiObjectTracker::new_automotive_enu(0.5, 16.0);

    let mut prev: Option<EgoPose> = None;
    for k in 0..ticks {
        let world_dets = detections_at(&agents, k);
        world_tracker.step_classed(&world_dets, ROAD_DT_S);

        // Ego drives north through the junction at 8 m/s.
        let t = k as f64 * ROAD_DT_S;
        let yaw = std::f64::consts::FRAC_PI_2;
        let pose = EgoPose {
            translation_m: [(-2.0), -50.0 + 8.0 * t, 0.0],
            rotation_wxyz: [(yaw / 2.0).cos(), 0.0, 0.0, (yaw / 2.0).sin()],
            time_s: t,
        };
        let motion = prev
            .and_then(|p| EgoMotion::from_pose_pair(&p, &pose))
            .unwrap_or_else(|| EgoMotion::stationary(pose));
        // Express each world detection in the ego frame (inverse transform
        // with a yaw-only pose: rotate by -yaw after translating).
        let (s, c) = (-yaw).sin_cos();
        let ego_dets: Vec<(DVector<f64>, TargetClass)> = world_dets
            .iter()
            .map(|(d, class)| {
                let dx = d[0] - pose.translation_m[0];
                let dy = d[1] - pose.translation_m[1];
                (
                    DVector::from_column_slice(&[c * dx - s * dy, s * dx + c * dy, d[2]]),
                    *class,
                )
            })
            .collect();
        ego_tracker.step_with_ego(&ego_dets, ROAD_DT_S, &motion);
        prev = Some(pose);
    }

    assert_eq!(
        world_tracker.confirmed_count(),
        ego_tracker.confirmed_count()
    );
    // Each confirmed ego-view track should coincide with a world-view track.
    for et in ego_tracker
        .tracks
        .iter()
        .filter(|t| t.lifecycle == TrackState::Confirmed)
    {
        let (ex, ey) = (et.state[0], et.state[2]);
        let nearest = world_tracker
            .tracks
            .iter()
            .filter(|t| t.lifecycle == TrackState::Confirmed)
            .map(|t| ((t.state[0] - ex).powi(2) + (t.state[2] - ey).powi(2)).sqrt())
            .fold(f64::INFINITY, f64::min);
        assert!(
            nearest < 1e-6,
            "ego-view track at ({ex:.2},{ey:.2}) should coincide with a world-view track (nearest {nearest:.3} m)"
        );
    }
}
