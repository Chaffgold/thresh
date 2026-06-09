# Design — Automotive Tracking Pipeline

## Context

thresh is an aerospace-focused multi-sensor MOT framework, but most of its
machinery is domain-agnostic. A survey of the codebase (and the automotive MOT
literature) shows the *filtering, association, and evaluation* layers already
generalize to automotive; the gaps are in *types, sensor models, and synthetic
data*. Notably, `thresh-data` already ships a `NuScenesBridge`
(`crates/thresh-data/src/nuscenes.rs`) with scene iteration, ego-pose, and
`SensorCalibration` (translation + rotation quaternion + camera intrinsics) — so
nuScenes ingestion is a rewrite-the-mapping problem, not a build-from-scratch
one.

This proposal also resolves a question the maintainer raised: whether NVIDIA
**Alpamayo** belongs here. It does not — see Decision 5.

## Goals / Non-Goals

**Goals**
- Native automotive classes with a lossless nuScenes mapping.
- LiDAR/camera measurements and ego-motion as first-class inputs.
- Automotive synthetic scenarios (vehicle dynamics) for the synth-first workflow.
- nuScenes-benchmarked evaluation (per-class MOTA + AMOTA), with public baselines
  for context.

**Non-Goals**
- A learned automotive detector or BEV fusion (possible follow-on change).
- Planning/prediction (and therefore Alpamayo).
- Waymo/KITTI, embedded deployment, real-time guarantees.

## Decisions

### 1. Extend `TargetClass`; rewrite `map_category()` losslessly

`TargetClass` (`crates/thresh-core/src/track.rs:47`) currently has `Aircraft`,
`Ballistic`, `Uav`, `Orbital`, `Unknown`. Add automotive variants (`Car`,
`Truck`, `Bus`, `Motorcycle`, `Bicycle`, `Pedestrian`; consider a generic
`Vehicle`). Then rewrite `thresh-data`'s `map_category()` (`nuscenes.rs:377`),
which today collapses `vehicle.car|truck|bus|trailer|… → Aircraft` and
`motorcycle|bicycle|pedestrian → Uav`, into a faithful mapping. Keep all
aerospace variants so existing tracking is untouched.

**Why:** per-class metrics and class-specific motion priors are impossible while
cars are typed as aircraft. Additive enum variants are backward-compatible.

### 2. Add LiDAR/camera measurements + an ego-motion input

Measurements today are `Radar | EoIr | AdsB | Othr`
(`crates/thresh-core/src/measurement.rs`). Add a LiDAR 3D-detection variant
(position + 3D box + yaw + class) and a camera variant; `SensorInputType` already
has a `PointCloud` for detector *input* (`detection.rs`). Separately, add an
**ego-motion** value (ego pose + linear/angular velocity) that the automotive
tracker feeds into prediction so the moving sensor platform is compensated —
this is the single biggest automotive-vs-aerospace difference (stationary/inertial
vs vehicle-mounted).

### 3. Kinematic bicycle model in `thresh-synth`

`SegmentType` (`crates/thresh-synth/src/trajectory.rs:53`) has `Cv | Ca | Ctrv |
Ballistic`. `Ctrv` already models constant-turn-rate (the 2D steering case), so
add a `KinematicBicycle { steering_angle, acceleration, wheelbase }` segment plus
road-scenario presets (lane-follow, intersection). Start with a 2-DOF bicycle
model; higher fidelity (tire slip, roll) is a later refinement.

### 4. Evaluate on nuScenes with per-class MOTA + AMOTA

`thresh-eval` already computes MOTA/MOTP/IDF1 and `compute_amota`
(`src/metrics.rs`). Add per-class aggregation and a nuScenes eval driver over
held-out scenes (start with the `mini` split), reporting AMOTA alongside
published baselines (SORT/DeepSORT/AB3DMOT) for context. Target a credible proof
point (e.g. MOTA ≥ 0.50 on the `val` split) rather than SOTA.

### 5. Alpamayo is out of scope (verified)

Alpamayo is a Vision-Language-Action model: 4-camera video + ego-motion history →
a 6.4 s trajectory (64 waypoints) + Chain-of-Causation natural-language rationale;
a planning/reasoning teacher model for distillation onto DRIVE Thor. It does no
detection, association, or state estimation and emits no tracks. It is orthogonal
to thresh's detection+tracking mission and provides zero value as a perception
component. Excluded.

### 6. Reuse `thresh-data`, don't rebuild ingestion

Build on the existing `NuScenesBridge`/`SensorCalibration` rather than a new
ingestion path. The work is: lossless class mapping, surfacing LiDAR/camera
measurements and ego-motion through the bridge, and the eval driver.

## Risks

- **Ego-motion plumbing** touches the filter prediction step; must stay opt-in so
  aerospace trackers are unaffected (mirror the additive, feature-gated approach
  used for learned-IMM).
- **Dataset availability** — nuScenes is large; prototype on `mini`, document
  acquisition, keep heavy data out of CI (consistent with project convention).
- **Scope creep toward a learned detector** — explicitly deferred to a follow-on.

## Open Questions

- Separate `TrackerVariant`/constructor for automotive, or a config flag on the
  ENU tracker? (Leaning: an `new_automotive_enu(...)` constructor + ego-motion,
  paralleling `new_imm_position`.)
- Generic `Vehicle` superclass vs only concrete classes? (nuScenes has both fine
  and coarse categories.)
- Should a future change add a learned automotive detector (Track-A analog) and
  is a BEV frame warranted, or does ENU + ego-motion suffice for the first cut?
