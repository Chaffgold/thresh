# Design — Learned-IMM Tracker Integration

## Context

`flight-data-training-pipeline` delivered `LearnedImmFilter` (in `thresh-filter`,
feature `learned-imm`): an `ImmFilter` wrapper whose mode probabilities come
from an ONNX classifier once a `WINDOW_LEN`-step history of filter-state
projections exists, with analytic fallback during warm-up and on error. But
`MultiObjectTracker` only ever constructs plain `ImmFilter`s, so nothing in the
tracker or the eval harness exercised the learned path. This change adds the
tracker-level seam.

## Goals / Non-Goals

**Goals**
- A learned-IMM tracker constructor with the same ergonomics as
  `new_imm_position`.
- The eval harness can run the learned tracker on identical inputs to the
  analytic one (a real A/B), behind a feature flag.
- Zero impact on default (non-`learned-imm`) builds.

**Non-Goals**
- Training / shipping a representative model (gated on the GPU run).
- Refactoring `LearnedImmFilter` to share one session across tracks.

## Decisions

### 1. Parallel per-track map, not an enum

The tracker keeps `imm_filters: HashMap<usize, ImmFilter>` and adds (gated)
`learned_imm_filters: HashMap<usize, LearnedImmFilter>`. A track lives in exactly
one map; predict/update/removal check the learned map first. This is more
localized than replacing the value type with an `enum { Analytic, Learned }`
(which would touch every match site) and keeps the default build byte-identical.

### 2. Per-track classifier session

Each track's `LearnedImmFilter` owns its own `ImmModeAdapter` (ONNX session),
loaded from the stored path at birth. `OnnxModel` is not shareable across the
per-track wrappers without a larger refactor, and evaluation scenarios have few
targets, so per-track load is acceptable. Documented as a known cost.

### 3. Fail fast in the constructor, fall back at birth

`new_imm_position_learned` validates the config, checks the bank has the
4-model `cv_ca_ctrv_ct` shape the classifier expects, and loads the ONNX once up
front (returning `Err` on any of these). At birth, if the classifier fails to
load (e.g. file removed mid-run) the track falls back to a plain analytic
`ImmFilter`, so tracking degrades gracefully rather than dropping the track.

### 4. Learned eval returns a stringified error

`run_eval_harness_learned` returns `Result<EvalReport, String>` (vs the analytic
`run_eval_harness`'s `TrajectoryRadarError`) because the learned constructor's
failure mode is a setup/string error; the binary prints either way.

## Risks

- **Per-track session load cost** scales with target count; only relevant for
  large scenarios, which the eval driver does not use.
- **Random-weight stub** makes the learned numbers non-representative (it badly
  degrades maneuvering MOTA, as expected) — this is a fixture limitation, not a
  defect; the seam itself is verified by the wiring producing *different*
  metrics from the analytic path.

## Open Questions

- Should a future change share one classifier session across tracks (an
  `Arc<Mutex<ImmModeAdapter>>` or a `&mut` borrow in `LearnedImmFilter`)? Only
  worth it if learned tracking is run on dense scenes.
