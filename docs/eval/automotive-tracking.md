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

## Results: v1.0-mini prototype (E.3)

Run on 2026-06-10 over all 10 `v1.0-mini` scenes (18,538 GT objects, 2 m
threshold, CPU-only; the mini download is ~4 GB and publicly fetchable —
no registration or GPU required):

| Run | Overall MOTA | Class-avg MOTA | MOTP | IDF1 | HOTA | IDSW |
|---|---|---|---|---|---|---|
| Ideal detections (σ = 0) | **0.9251** | 0.8740 | 0.016 m | 0.9708 | 0.9318 | 336 |
| Perturbed (σ = 0.3 m) | **0.8484** | 0.7566 | 0.442 m | 0.9709 | 0.8273 | 1761 |

Per-class MOTA (ideal): Truck 0.950, Car 0.918, Pedestrian 0.898, Bus 0.873,
Motorcycle 0.866, Bicycle 0.860, Unknown 0.753 (non-automotive nuScenes
categories — barriers, cones, debris — correctly bucketed to `Unknown` and,
as expected, hardest to track). The residual ID switches at σ = 0 come from
annotation gaps (occlusion) interacting with the confirm/delete lifecycle.

This satisfies E.3's prototype clause (MOTA ≥ 0.50 by a wide margin, per-class
+ class-averaged reporting end-to-end through `NuScenesBridge`). The full
`val`-split run (~300 GB trainval download) and a *detector-fed* comparison
against the baselines above remain follow-ons.

### Runtime notes (macOS)

The embedding binary must initialize Python itself — `thresh-data`'s `pyo3`
carries the `auto-initialize` feature for this. On macOS, build against a
concrete interpreter and point the embedded runtime at the devkit's
site-packages:

```sh
python3.12 -m venv ~/data/nuscenes/venv && ~/data/nuscenes/venv/bin/pip install nuscenes-devkit
PYO3_PYTHON=$(brew --prefix python@3.12)/bin/python3.12 \
  cargo build -p thresh --features nuscenes-eval --bin eval-nuscenes
PYTHONPATH=~/data/nuscenes/venv/lib/python3.12/site-packages \
  ./target/debug/eval-nuscenes --dataroot ~/data/nuscenes --version v1.0-mini
```

(Linking the macOS system Python 3.9 fails at load with a
`@rpath/Python3.framework` dyld error; Homebrew Python links by absolute
path.)
