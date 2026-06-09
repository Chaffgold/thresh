# Automotive tracking evaluation

Status of the `automotive-tracking-pipeline` evaluation (Phase 4). This report
covers the evaluation harness, how to run it, published baselines for context,
and what remains gated on data availability.

## Harness

Two drivers exercise the automotive tracker (`MultiObjectTracker::new_automotive_enu`):

1. **Synthetic road scenarios** (no external data; runs in CI as tests).
   `thresh-synth::road_scenarios` presets feed `step_classed` /
   `step_with_ego`; `crates/thresh/tests/automotive_integration.rs` asserts the
   full intersection scene is tracked (every agent confirmed with its own
   class) and that a moving-ego view coincides with the world-frame view.

2. **nuScenes scenes** (`eval-nuscenes`, requires a local download):

   ```sh
   cargo run -p thresh --features nuscenes-eval --bin eval-nuscenes -- \
     --dataroot /data/nuscenes --version v1.0-mini [--scenes N] \
     [--noise-sigma 0.3] [--json]
   ```

   Per keyframe, the scene's annotation boxes become class-tagged detections
   (optionally perturbed with Gaussian noise of `--noise-sigma` metres), the
   tracker steps in the nuScenes global frame, and confirmed tracks are scored
   with `thresh-eval`: class-blind MOTA/MOTP/IDF1/HOTA plus the per-class
   breakdown and the class-averaged MOTA
   (`thresh-eval::per_class`, filter-then-evaluate per the nuScenes
   convention).

   **Scope note:** the harness currently feeds ground-truth-derived detections
   (ideal or noise-perturbed), so its numbers measure the *tracking* layer —
   association, lifecycle, per-class priors — not detection quality. Scores
   from a real detector (and with it score-thresholded AMOTA over recall
   points) arrive with the learned automotive detector follow-on;
   `compute_amota` documents this simplification.

## Published baselines (context)

Published nuScenes 3D MOT results, for context once the harness runs
detector-fed (numbers are AMOTA on a 0–1 scale; the nuScenes metric averages
MOTA over 40 recall thresholds — see the
[nuScenes devkit tracking docs](https://github.com/nutonomy/nuscenes-devkit/blob/master/python-sdk/nuscenes/eval/tracking/README.md)):

| Method | Split | AMOTA | Detections | Source |
|---|---|---|---|---|
| AB3DMOT | nuScenes test | 0.151 | Megvii | [SimpleTrack paper, Table 7](https://arxiv.org/abs/2111.09621) |
| AB3DMOT | nuScenes val | 0.586 (sAMOTA) | Megvii | [AB3DMOT repo](https://github.com/xinshuoweng/AB3DMOT/blob/master/docs/nuScenes.md) |
| CenterPoint | nuScenes test | 0.638 | self (center-based) | [CenterPoint, CVPR 2021, Table 4](https://arxiv.org/abs/2006.11275) |
| SimpleTrack (10 Hz) | nuScenes test | 0.668 | CenterPoint | [SimpleTrack, Table 7](https://arxiv.org/abs/2111.09621) |
| EagerMOT | nuScenes val | 0.712 | CenterPoint + Cascade R-CNN | EagerMOT (ICRA 2021) |

For historical context only: **SORT** (~0.41 MOTA) and **DeepSORT** (~0.61
MOTA) are 2D pedestrian-tracking baselines on MOT16/17 — a different task and
metric, listed because the proposal named them
([SORT](https://arxiv.org/abs/1602.00763),
[DeepSORT](https://arxiv.org/abs/1703.07402)).

Note AB3DMOT's val number is **sAMOTA** (a scaled variant) while its test
number is challenge AMOTA — they are not directly comparable; both are listed
as published.

## Getting the data (task 4.4)

The nuScenes dataset is **never** fetched in CI; the `nuscenes-eval` feature is
opt-in and compile-verified only.

1. Register at <https://www.nuscenes.org/nuscenes> and download a split —
   start with **v1.0-mini** (~4 GB; 10 scenes) for prototyping; `trainval`
   (~300 GB) for the real benchmark.
2. Extract so that `<dataroot>/v1.0-mini/` contains the JSON tables and
   `<dataroot>/samples/`, `<dataroot>/sweeps/` the sensor data.
3. Install the devkit into the Python environment PyO3 will use:
   `pip install nuscenes-devkit` (the bridge imports `nuscenes.nuscenes`).
4. Run `eval-nuscenes` as above. Licensing: nuScenes is CC BY-NC-SA 4.0
   (non-commercial); the dataset and any derived caches stay out of the
   repository (consistent with `LICENSING.md`).

## Exit-criterion status (E.3)

> The tracker runs end-to-end over a nuScenes split via `NuScenesBridge`,
> reporting per-class MOTA + AMOTA, with a credible proof point (e.g. MOTA ≥
> 0.50 on the `val` split; prototype on `mini`).

The driver, per-class metrics, and reporting exist and are compile-verified
under `--features nuscenes-eval`; the synthetic-scenario harness passes in CI.
Producing the benchmark *numbers* requires the nuScenes download on a local
machine (data-gated, like the flight-data pipeline's GPU-gated training runs)
— and a *detector-fed* comparison against the baselines above additionally
requires the learned automotive detector follow-on.
