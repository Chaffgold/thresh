# Tasks — Automotive Tracking Pipeline

> Phases are dependency-ordered. Phase 1 (class taxonomy) is implemented;
> Phases 1–2 unblock everything else, and Phase 3 (synth) and Phase 4 (eval)
> can proceed in parallel once the types land.

## 1. Class taxonomy

- [x] 1.1 Extend `TargetClass` (`crates/thresh-core/src/track.rs`) with automotive variants: `Car`, `Truck`, `Bus`, `Motorcycle`, `Bicycle`, `Pedestrian`, keeping all aerospace variants and the `serde`/`Hash` derives. _Concrete classes only (per maintainer decision); doc note flags that the set is expected to grow (Trailer/ConstructionVehicle/EmergencyVehicle/Animal) and SHOULD be matched with a wildcard. No exhaustive `match` sites needed updating (`HeadRegistry::get` falls back to the `Unknown` head; `thresh-viz` uses a wildcard arm)._
- [x] 1.2 Rewrite `thresh-data`'s `map_category()` (`crates/thresh-data/src/nuscenes.rs`) to a lossless nuScenes→`TargetClass` mapping (no more `vehicle.car → Aircraft`); cover all nuScenes categories with an explicit `Unknown` fallback only for genuinely non-automotive ones. _Coarse types folded into the nearest concrete class for now (`trailer`/`construction` → `Truck`, `emergency.*` → `Car`), prefix-matched most-specific-first._
- [x] 1.3 Unit-test the mapping over the full nuScenes category list; assert no automotive category resolves to an aerospace class. _`map_category_no_automotive_maps_to_aerospace` iterates the full nuScenes detection category list and asserts none map to an aerospace class or to `Unknown`; plus per-group mapping tests._

## 2. Sensor measurements and ego-motion

- [x] 2.1 Add a LiDAR 3D-detection measurement variant (position + 3D box + yaw + class) to `crates/thresh-core/src/measurement.rs`. _`Measurement::Lidar { position, extent, yaw, class, velocity, time, sensor_id }`; arms added to `time`/`to_vector`/`dim`/`default_noise` (3D, or 6D with velocity; ~0.2 m std position noise). The one exhaustive external match (`thresh-data::benchmark::measurement_to_cartesian`) extended._
- [x] 2.2 Add a camera (monocular 3D / image-plane) measurement variant. _`Measurement::Camera { center_px, extent_px, position: Option<[f64;3]>, yaw: Option<f64>, class, time, sensor_id }` — image-plane box plus optional monocular 3D estimate (per the spec); `to_vector` prefers the metric estimate, else the pixel center._
- [x] 2.3 Add an ego-motion type (ego pose + linear & angular velocity) and surface it from `NuScenesBridge` (ego-pose deltas). _`thresh_core::ego::{EgoPose, EgoMotion}` (un-gated, fully unit-tested quaternion math; `[w,x,y,z]` matching nuScenes; `EgoMotion::from_pose_pair` finite-differences two poses). `NuScenesBridge::ego_pose(sample, channel)` resolves `sample_data → ego_pose` (µs → s). Design Decision 8._
- [x] 2.4 Add an automotive ENU tracker entry point (e.g. `MultiObjectTracker::new_automotive_enu(...)`) that feeds ego-motion into the prediction step; keep it additive/opt-in so aerospace trackers are unchanged. _`new_automotive_enu` + `step_with_ego(&[(detection, class)], dt, &EgoMotion)` lift ego-frame detections into world ENU at the measurement boundary (Decision 7: with world-frame tracks, lifting *is* the prediction compensation — verified by `stationary_object_does_not_drift_under_ego_motion`). `step`/`step_detections` byte-identical; `identity_ego_matches_world_frame_step` + `plain_step_is_unchanged_by_automotive_entry_points` guard the aerospace path._
- [x] 2.5 Class-specific motion priors (e.g. pedestrian ≤ ~6 m/s, car ≤ ~40 m/s) via the existing head/lifecycle registry. _Six automotive `TrackHead` factories (all `state_dim=6`, position-tracker compatible): speed priors as velocity covariance (pedestrian var 4 ≪ car var 225), agility as `process_noise_sigma` (motorcycle 3.0 > car 2.0 > truck 1.5 > bus 1.0), fast confirm/delete for pedestrians. Engaged at birth via `step_classed`. Ordering + compatibility tests in `heads.rs`._

## 3. Synthetic automotive scenarios

- [ ] 3.1 Add a `KinematicBicycle { steering_angle, acceleration, wheelbase }` `SegmentType` to `crates/thresh-synth/src/trajectory.rs` (2-DOF; reuse `Ctrv` math where applicable).
- [ ] 3.2 Add road-scenario presets (lane-follow, intersection, multi-agent) with configurable clutter.
- [ ] 3.3 Generate a synthetic automotive dataset and verify trajectories are physically plausible (speed/turn-rate bounds per class).

## 4. nuScenes-benchmarked evaluation

- [ ] 4.1 Add per-class MOTA aggregation to `crates/thresh-eval` (alongside the existing `compute_mot_metrics` / `compute_amota`).
- [ ] 4.2 Add a nuScenes eval driver (over held-out scenes; start with the `mini` split) that runs the automotive tracker through `NuScenesBridge` and reports per-class MOTA + AMOTA.
- [ ] 4.3 Record published baselines (SORT / DeepSORT / AB3DMOT) for context in an eval report under `docs/eval/`.
- [ ] 4.4 Document nuScenes acquisition (mirroring `TRAINING.md`); keep heavy data out of CI.

## 5. Documentation and wrap-up

- [ ] 5.1 `docs/automotive/getting-started.md`: class taxonomy, ego-motion semantics, coordinate-frame conventions, a runnable nuScenes-mini example.
- [ ] 5.2 Final lint/test/`openspec validate` pass; update `design.md` Open Questions with any decisions made during implementation.

## Exit criteria

- [ ] E.1 `TargetClass` includes native automotive classes and `map_category()` is lossless (no automotive category maps to an aerospace class).
- [ ] E.2 LiDAR and ego-motion are first-class in the type system; the automotive tracker compensates for ego-motion.
- [ ] E.3 The tracker runs end-to-end over a nuScenes split via `NuScenesBridge`, reporting per-class MOTA + AMOTA, with a credible proof point (e.g. MOTA ≥ 0.50 on the `val` split; prototype on `mini`).
- [ ] E.4 All code passes clippy/tests/CI; no aerospace regression.
