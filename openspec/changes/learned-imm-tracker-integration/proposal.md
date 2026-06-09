# Learned-IMM Tracker Integration

## What

Add a tracker-level entry point for the learned IMM mode classifier so the
multi-object tracker can run the learned mode-probability update end-to-end,
and make the evaluation harness's learned-vs-analytic A/B actually runnable.

Concretely:
- `MultiObjectTracker::new_imm_position_learned(config_factory, onnx_path, …)`
  (behind a new `learned-imm` feature on `thresh-tracker`) — births each track
  with a `LearnedImmFilter` instead of a plain `ImmFilter`.
- `thresh::eval_harness::run_eval_harness_learned(…)` and an `eval-tracker
  --learned-imm --model <path>` flag (behind `thresh`'s `learned-imm` feature)
  that route through the learned constructor.

## Why

The `flight-data-training-pipeline` change built Track B down to a filter-level
`LearnedImmFilter` (the `learned-imm` feature on `thresh-filter`) but stopped
short of the tracker: `MultiObjectTracker` had no learned constructor, so task
9.3's "A/B test the classical and learned pipelines" could not run — the
`eval-tracker --learned-imm` flag silently fell back to the analytic baseline.
This change closes that seam. It is the smallest addition that turns the
deferred A/B into a runnable comparison; a *representative* result still depends
on a trained checkpoint (the committed model is a random-weight stub), which
remains gated on the GPU training run.

## How

`LearnedImmFilter` already wraps an `ImmFilter` and exposes the same
`predict` / `update_with_measurement` surface, falling back to the analytic
update during the warm-up window and on any classifier error. The tracker keeps
a parallel `learned_imm_filters` map (a track lives in exactly one map) and, at
birth, wraps the IMM bank when the learned path is active. The predict / update
/ lifecycle / removal paths prefer the learned map when present. All of this is
gated on `learned-imm`; default builds are unchanged.

## Out of scope

- Training a representative classifier or committing trained weights (gated on
  the GPU run; the random-weight stub is used only to exercise the wiring).
- Sharing one ONNX session across tracks (each track loads its own session;
  fine for the small target counts in evaluation scenarios).
- Any change to the analytic IMM path or to single-model KF tracking.

## Affected crates and paths

- `crates/thresh-tracker` — new `learned-imm` feature; `new_imm_position_learned`;
  learned-filter map + birth/predict/update/removal wiring (`src/tracker.rs`).
- `crates/thresh` — new `learned-imm` feature; `run_eval_harness_learned`
  (`src/eval_harness.rs`); `eval-tracker --learned-imm --model` (`src/bin/eval-tracker.rs`).
- `.github/workflows/ci.yml` — `onnx-tests` runs the learned-IMM tracker tests.

## Dependencies

- `thresh-filter`'s existing `learned-imm` feature (`ImmModeAdapter`,
  `LearnedImmFilter`) and, transitively, `thresh-inference`'s `onnx` (ort).
