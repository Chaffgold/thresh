# Automotive tracking — getting started

thresh's automotive support (the `automotive-tracking-pipeline` change) extends
the aerospace tracker to road scenes: native automotive object classes, LiDAR
and camera measurements, ego-motion compensation for a moving sensor platform,
synthetic road scenarios, and per-class evaluation benchmarked against
nuScenes. Everything is additive — aerospace tracking is untouched.

## Class taxonomy

`thresh_core::track::TargetClass` carries both domains:

| Automotive | nuScenes categories (via `thresh_data::category::map_category`) |
|---|---|
| `Car` | `vehicle.car`, `vehicle.emergency.*` |
| `Truck` | `vehicle.truck`, `vehicle.trailer`, `vehicle.construction` |
| `Bus` | `vehicle.bus.*` |
| `Motorcycle` | `vehicle.motorcycle` |
| `Bicycle` | `vehicle.bicycle` |
| `Pedestrian` | `human.pedestrian.*` |
| `Unknown` | non-automotive categories (`animal`, `movable_object.*`, …) |

The mapping is lossless: no automotive category resolves to an aerospace class
(`Aircraft`, `Ballistic`, `Uav`, `Orbital` remain aerospace-only). The enum is
expected to grow (e.g. `Trailer`, `EmergencyVehicle`); match it with a
wildcard rather than exhaustively.

Each class has a tracking head (`thresh_tracker::heads::TrackHead`) expressing
its motion priors: speed prior as initial velocity covariance (pedestrian ≪
car), agility as process noise (motorcycle > car > truck > bus), and
class-appropriate confirmation/deletion (pedestrians confirm and delete fast).

## Coordinate frames and ego-motion

Tracking happens in a fixed **world ENU frame** (metres; `z = 0` is the road
plane in the synthetic presets; nuScenes uses its global frame). Automotive
sensors ride on a moving platform, so measurements typically arrive in the
**ego/body frame**. `thresh_core::ego` carries the platform state:

- `EgoPose` — the ego→world transform: `p_world = R(q) · p_ego + t`, with the
  quaternion in `[w, x, y, z]` (scalar-first, matching the nuScenes `ego_pose`
  record) and translation in metres.
- `EgoMotion` — a pose plus world-frame linear (m/s) and angular (rad/s)
  velocity; `EgoMotion::from_pose_pair` finite-differences two consecutive
  poses.

Ego compensation happens **at the measurement boundary**: `step_with_ego`
lifts ego-frame detections into the world frame and then runs the standard
cycle, so a stationary world object never drifts due to platform motion and
the Kalman predict step needs no platform-specific changes. (See design
Decision 7 in the OpenSpec change.)

## Quickstart: synthetic road scene

The synthetic presets need no external data:

```rust
use thresh::synth::road_scenarios::{intersection, ROAD_DT_S};
use thresh::tracker::tracker::MultiObjectTracker;
use nalgebra::DVector;

// Class-tagged ground-truth agents: crossing car + truck, a left-turning
// car, and a crosswalk pedestrian.
let agents: Vec<_> = intersection()
    .into_iter()
    .map(|a| { let wps = a.waypoints(); (a, wps) })
    .collect();
let ticks = agents.iter().map(|(_, w)| w.len()).min().unwrap();

let mut tracker = MultiObjectTracker::new_automotive_enu(0.5, 16.0);
for k in 0..ticks {
    let detections: Vec<_> = agents
        .iter()
        .filter_map(|(agent, wps)| wps.get(k).map(|w| {
            (DVector::from_column_slice(&w.position), agent.class)
        }))
        .collect();
    tracker.step_classed(&detections, ROAD_DT_S);
}
assert_eq!(tracker.confirmed_count(), agents.len());
```

`crates/thresh/tests/automotive_integration.rs` is the living version of this
example (including the moving-ego variant via `step_with_ego`), and
`thresh-synth`'s `road_scenarios` tests enforce per-class physical
plausibility bounds. The trajectory generator's
`SegmentType::KinematicBicycle { steering_angle, acceleration, wheelbase }`
provides ground-vehicle steering kinematics for building custom scenarios.

## Per-class evaluation

`thresh_eval::per_class` evaluates the nuScenes way — filter ground truth and
tracks to one class, score each stream independently:

```rust
use thresh::eval::per_class::{build_classed_report, class_averaged_mota, ClassedFrameData};

let frames: Vec<ClassedFrameData> = collect_frames(); // (id, [x,y,z], class) per frame
let report = build_classed_report(&frames, 2.0);
println!("{}", report.to_table());           // overall + per-class MOTA/HOTA
println!("{}", class_averaged_mota(&report.per_class));
```

## nuScenes-mini example

With a local nuScenes download and the devkit installed (acquisition,
licensing, and flags are documented in `docs/eval/automotive-tracking.md`):

```sh
pip install nuscenes-devkit
cargo run -p thresh --features nuscenes-eval --bin eval-nuscenes -- \
  --dataroot /data/nuscenes --version v1.0-mini --noise-sigma 0.3
```

This drives the tracker over every keyframe of each scene and prints the
class-blind and per-class report. The dataset is never fetched in CI; the
`nuscenes-eval` feature is opt-in.

## What's deferred

- **Learned automotive detector** (and with it score-thresholded AMOTA and
  detector-fed baseline comparisons — see the baselines table in
  `docs/eval/automotive-tracking.md`).
- **BEV fusion frame** — world ENU + measurement-boundary ego compensation
  suffices for the current scope.
- **nuScenes benchmark numbers** — the harness is complete; producing the
  numbers needs the dataset locally (data-gated).
