# Automotive Tracking Pipeline

## What

Extend thresh from its aerospace focus to first-class **automotive** multi-object
tracking, benchmarked on **nuScenes**, by closing the domain-specific gaps the
codebase has today. Concretely:

- A native automotive **class taxonomy** (`Car`, `Truck`, `Bus`, `Motorcycle`,
  `Bicycle`, `Pedestrian`, …) that replaces the lossy nuScenes→aerospace mapping
  currently in `thresh-data`.
- Automotive **sensor measurement types** (LiDAR 3D detections, monocular/3D
  camera detections) and an **ego-motion input** so the moving-platform geometry
  is handled.
- Automotive **synthetic scenarios** in `thresh-synth` (a kinematic bicycle
  model, road/clutter layouts) to mirror the flight-data synth approach.
- An automotive **evaluation mode** in `thresh-eval` (per-class MOTA, nuScenes
  AMOTA) run over held-out nuScenes scenes.

This is a **design proposal**; no implementation tasks are checked.

## Why

thresh already has the hard parts of automotive MOT: Kalman/EKF/UKF/CKF + IMM
filters, Hungarian/JPDA/MHT association, an ENU tracker, MOTA/MOTP/IDF1/**AMOTA**
metrics, and — crucially — a `thresh-data` crate with a working `NuScenesBridge`
(scene iteration, ego-pose and sensor-calibration transforms). What it lacks is
domain *fit*: the only way to ingest a car today is `thresh-data`'s
`map_category()`, which maps `vehicle.car`/`truck`/`bus` → `TargetClass::Aircraft`
and `motorcycle`/`bicycle`/`pedestrian` → `TargetClass::Uav`
(`crates/thresh-data/src/nuscenes.rs:377`). That workaround is explicitly
"best-effort … the closest rigid-body analog" and makes per-class evaluation
meaningless.

Automotive is also a strategically low-risk expansion: it is a public,
well-benchmarked domain (nuScenes/Waymo/KITTI) with published baselines
(SORT/DeepSORT/AB3DMOT), so it both validates that the core filter library
generalizes and gives the project a comparable proof point.

**On NVIDIA Alpamayo:** investigated and explicitly **out of scope**. Alpamayo is
a Vision-Language-Action *planning/reasoning* model (multi-camera video + ego
history → a 6.4 s driving trajectory + natural-language rationale). It performs
no detection, association, or state estimation and emits no object tracks, so it
is orthogonal to thresh's detection+tracking niche. It would only ever be
relevant if thresh grew a trajectory-*planning* module, which this change does
not propose.

## How

Follow the same `synth → (train) → ingest → track → eval` spine the
`flight-data-training-pipeline` used, but reuse the existing `thresh-data`
nuScenes plumbing instead of building ingestion from scratch:

1. **Class taxonomy** — add automotive variants to `TargetClass`; rewrite
   `map_category()` to a lossless mapping; keep aerospace variants intact.
2. **Measurements & ego-motion** — add LiDAR/camera measurement variants and an
   ego-motion channel so tracker prediction can compensate for the moving
   platform.
3. **Synth** — add a kinematic bicycle `SegmentType` and automotive scenario
   presets to `thresh-synth`.
4. **Eval** — add per-class MOTA + nuScenes AMOTA reporting over held-out scenes,
   with SORT/AB3DMOT baselines for context.

## Out of scope

- NVIDIA Alpamayo or any planning/trajectory-prediction capability.
- Bird's-Eye-View (BEV) learned fusion or a learned automotive detector (a future
  change may add an automotive detector mirroring Track A).
- Real-time/embedded deployment (DRIVE Thor, etc.).
- Waymo/KITTI ingestion (nuScenes is the initial benchmark target).

## Affected crates and paths

- `crates/thresh-core` — `TargetClass` (`src/track.rs:47`); measurement variants
  (`src/measurement.rs`); ego-motion type.
- `crates/thresh-data` — `map_category()` rewrite + LiDAR/camera plumbing through
  `NuScenesBridge` (`src/nuscenes.rs`).
- `crates/thresh-tracker` — an automotive ENU constructor wiring ego-motion into
  prediction.
- `crates/thresh-synth` — kinematic bicycle `SegmentType` + scenario presets
  (`src/trajectory.rs`).
- `crates/thresh-eval` — per-class MOTA + AMOTA reporting (`src/metrics.rs` already
  has `compute_amota`).

## Dependencies

- Existing `thresh-data` `NuScenesBridge` + `SensorCalibration` (ego/sensor
  transforms already implemented).
- nuScenes dataset (or its `mini` split) for benchmarking — not committed; an
  acquisition note will mirror `TRAINING.md`.
