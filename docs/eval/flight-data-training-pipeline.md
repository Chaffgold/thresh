# Evaluation — flight-data-training-pipeline

MOT evaluation of the thresh tracker on synthetic radar scenarios, for the
`flight-data-training-pipeline` change (Phase 9, task 9.4).

## Harness

The evaluation engine is **Rust-native** (`thresh::eval_harness`, design
Decision 24): each scenario's target trajectories are turned into synthetic
radar measurements (`thresh-synth`), run through the analytic 4-model IMM
tracker (`thresh-tracker`, `cv_ca_ctrv_ct`), and the confirmed tracks are scored
against ground truth with `thresh-eval` (MOTA / MOTP / IDF1). Ground truth is
built independently from the trajectory waypoints (stable per-target IDs,
sensor-frame, range-gated) — a missed detection is a true false negative.

Reproduce:

```sh
cargo run -p thresh --bin eval-tracker            # human-readable
cargo run -p thresh --bin eval-tracker -- --json  # machine-readable
# or via the Python wrapper:
uv run -C python python -m eval.run_tracker
```

Settings: 10 Hz, seed 42, matching gate 50 m, `TrajectoryRadarConfig::default()`
radar (≈10 m range / 1 mrad angular noise, P_d ≈ 0.9).

## Analytic baseline (current)

| Scenario | MOTA | MOTP (m) | IDF1 | ID switches | frames |
|---|---|---|---|---|---|
| single-cruise (1 CV target) | 0.904 | 20.6 | 0.949 | 0 | 301 |
| two-separated (2 CV targets) | 0.894 | 20.7 | 0.944 | 0 | 301 |
| maneuvering (CV → coordinated turn) | 0.890 | 36.7 | 0.942 | 0 | 301 |

These are the **analytic-IMM baseline** the learned components must beat (or at
least not regress). MOTA ≈ 0.9 — the ~10% gap is dominated by the pre-confirmation
warm-up (the first few frames have ground truth but no confirmed track) plus
occasional missed detections; **0 ID switches** and IDF1 ≈ 0.94 indicate stable
identity. MOTP rises during the turn (≈37 m vs ≈21 m), as expected for an IMM
tracking a maneuver.

## Exit criteria — status

| Track | Criterion | Status |
|---|---|---|
| **B** (IMM mode classifier) | held-out accuracy ≥ 0.70; MOTA no worse than analytic (task 6.8) | **Pending** — needs a trained classifier + tracker-level learned-IMM integration (see below) |
| **A** (detector) | mAP@0.5 ≥ 0.30; downstream MOTA improvement (task 7.9) | **Pending** — needs a trained detector |

## Learned vs. analytic A/B — what's blocking it

`run_tracker.py` / `eval-tracker` accept `--learned-imm` / `--learned-detector`,
but the A/B comparison is not yet runnable, for two independent reasons:

1. **No tracker-level learned path.** Phase 6 added `LearnedImmFilter` as a
   *filter-level* wrapper; `MultiObjectTracker` has no learned-IMM constructor,
   so the eval's tracker can't yet swap in the learned classifier. Adding a
   `MultiObjectTracker::new_imm_position_learned(...)` is the missing piece.
2. **No trained checkpoints.** The committed models are random-weight stubs;
   real training (tasks 6.8 / 7.5 / 7.9) is deferred (non-CI, GPU-gated — design
   Decision 7). Until then the flags fall back to the analytic baseline.

When both land, this report gains an analytic-vs-learned column per scenario and
the exit-criteria rows are filled in.
