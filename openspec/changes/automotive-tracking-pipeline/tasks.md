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

- [x] 3.1 Add a `KinematicBicycle { steering_angle, acceleration, wheelbase }` `SegmentType` to `crates/thresh-synth/src/trajectory.rs` (2-DOF; reuse `Ctrv` math where applicable). _Heading/speed derived from planar velocity like `Ctrv`; `ω = v·tan(δ)/L` evaluated mid-step with midpoint position integration (O(dt²)); speed clamps at standstill (no reversing through a braking segment). Analytic tests: straight acceleration, constant-steering circle of radius `L/tan(δ)`, brake-to-stop distance `v²/2a`._
- [x] 3.2 Add road-scenario presets (lane-follow, intersection, multi-agent) with configurable clutter. _`thresh-synth::road_scenarios`: `lane_follow(n)` (platoon + lead-car brake-to-stop), `intersection()` (crossing car/truck, left-turning car, crosswalk pedestrian), `multi_agent()` (both scenes + a cyclist, ≥4 classes). Deterministic class-tagged `RoadAgent`s; measurement clutter/noise stays the synth sensor layer's job (`from_trajectory`), keeping GT presets pure._
- [x] 3.3 Generate a synthetic automotive dataset and verify trajectories are physically plausible (speed/turn-rate bounds per class). _`plausibility::{max_speed_mps, max_turn_rate_radps}` per class; `presets_are_physically_plausible` checks every preset agent against its class bounds (plus on-road-plane and uniqueness tests). End-to-end: `crates/thresh/tests/automotive_integration.rs` tracks the full intersection with `new_automotive_enu` (every agent confirmed with its class) and proves a moving-ego view coincides with the world-frame view._

## 4. nuScenes-benchmarked evaluation

- [x] 4.1 Add per-class MOTA aggregation to `crates/thresh-eval` (alongside the existing `compute_mot_metrics` / `compute_amota`). _New `per_class` module: `ClassedFrameData`, `split_frames_by_class` (filter-then-evaluate per the nuScenes convention), `compute_per_class_mot` (populates the existing `ClassReport`), `class_averaged_mota`, and `build_classed_report` (fills `EvalReport.per_class`, previously never populated). Tests cover the class-blind-vs-class-averaged split (2/3 vs 0.5 analytic case), cross-class id-collision isolation, and track-only classes._
- [x] 4.2 Add a nuScenes eval driver (over held-out scenes; start with the `mini` split) that runs the automotive tracker through `NuScenesBridge` and reports per-class MOTA + AMOTA. _`eval-nuscenes` binary behind the new `nuscenes-eval` feature: annotations → class-tagged detections (optional `--noise-sigma` jitter) → `step_classed` in the global frame → class-blind + per-class + class-averaged MOTA. `InstanceTrack` now carries `TargetClass` directly (no numeric-discriminant round-trip). Per-scene track-id offsetting prevents cross-scene id aliasing. Compile/clippy-verified under the feature; running needs a local nuScenes download (data-gated, Decision 11)._
- [x] 4.3 Record published baselines (SORT / DeepSORT / AB3DMOT) for context in an eval report under `docs/eval/`. _`docs/eval/automotive-tracking.md`: web-verified, cited AMOTA numbers (AB3DMOT 0.151 test / 0.586 sAMOTA val; CenterPoint 0.638; SimpleTrack 0.668; EagerMOT 0.712) with the sAMOTA-vs-AMOTA caveat, and SORT/DeepSORT flagged as 2D-MOT context only._
- [x] 4.4 Document nuScenes acquisition (mirroring `TRAINING.md`); keep heavy data out of CI. _Same doc: registration/download (mini → trainval), layout, `pip install nuscenes-devkit`, CC BY-NC-SA licensing note; the `nuscenes-eval` feature is opt-in and CI never fetches the dataset._

## 5. Documentation and wrap-up

- [ ] 5.1 `docs/automotive/getting-started.md`: class taxonomy, ego-motion semantics, coordinate-frame conventions, a runnable nuScenes-mini example.
- [ ] 5.2 Final lint/test/`openspec validate` pass; update `design.md` Open Questions with any decisions made during implementation.

## Exit criteria

- [ ] E.1 `TargetClass` includes native automotive classes and `map_category()` is lossless (no automotive category maps to an aerospace class).
- [ ] E.2 LiDAR and ego-motion are first-class in the type system; the automotive tracker compensates for ego-motion.
- [ ] E.3 The tracker runs end-to-end over a nuScenes split via `NuScenesBridge`, reporting per-class MOTA + AMOTA, with a credible proof point (e.g. MOTA ≥ 0.50 on the `val` split; prototype on `mini`).
- [ ] E.4 All code passes clippy/tests/CI; no aerospace regression.
