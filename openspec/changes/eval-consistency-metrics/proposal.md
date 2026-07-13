# Evaluation Consistency Metrics (NEES / NIS / GOSPA)

## Why

thresh's accuracy claims are unfalsifiable at the covariance level: `thresh-eval` measures only assignment-quality MOT metrics (MOTA/MOTP/IDF1/HOTA/AMOTA), so a filter that reports wildly overconfident or underconfident covariances scores identically to an honest one as long as its point estimates land inside the matcher's distance threshold. This gap becomes acute with the in-flight `orbital-ballistic-filter-models` change: it introduces tunable process-noise parameters (`sigma_accel`, `sigma_beta`) and its section 6 recalibrates the orbital/ballistic MOTA benchmark gates — without consistency metrics there is no way to tell whether a given Q tuning produces a statistically honest filter or merely games MOTA. This change adds the standard falsifiability layer (Bar-Shalom NEES/NIS with chi-squared bounds; GOSPA per Rahmathullah/García-Fernández/Svensson 2017) so the gate recalibration can be accepted on evidence.

## What Changes

- **thresh-eval**: NEES (normalized estimation error squared, truth-required) and NIS (normalized innovation squared, truth-free) with two-sided chi-squared consistency bounds at a configurable confidence level — both time-averaged (ANEES over Monte Carlo runs / track lifetime) and per-track variants, following Bar-Shalom's formulation.
- **thresh-eval**: GOSPA metric with the alpha=2 convention and its exact localization / missed-target / false-track decomposition, computed per frame over the existing `FrameData` sequence via optimal assignment (the Hungarian solver already in `thresh-association`).
- **thresh-filter**: minimal, non-breaking update-diagnostics exposure. Today only the linear KF offers `innovation()` / `innovation_covariance()` helpers (and only *pre*-update, recomputed from H and R); EKF has neither, and the UKF/CKF sigma-point `update()` computes the innovation and S internally and discards them — S there is sigma-point-derived and **cannot** be reconstructed by a caller. All four filters expose the post-update innovation and innovation covariance S consumed during the last update (exact mechanism — `UpdateOutcome` return value vs. stored getters — is a design.md decision), plus the predicted state/covariance hooks NEES needs.
- **thresh-data**: scenario TOMLs can assert NEES/NIS consistency bounds alongside MOTA — `Baselines` (currently `mota`/`hota`/`idf1` only) gains optional consistency fields and `check_regression` enforces them, so the orbital change's section-6 calibration can adopt them. No existing scenario baseline values change here.

**Out of scope** (stated explicitly): trajectory-GOSPA (deferred until a trajectory-level tracker exists), OSPA(2), PCRLB, NIS whiteness / Ljung-Box tests (belongs to the later time-correlated-noise work), any tracker or association algorithm changes beyond the minimal filter accessors, and any re-baselining of existing scenarios (the orbital change's section 6 owns that).

## Capabilities

### New Capabilities

- `filter-consistency-metrics`: NEES and NIS computation in thresh-eval with two-sided chi-squared bounds — per-track and time-averaged variants, a pass/fail consistency verdict at a configurable confidence level, and the truth-error / innovation input contracts the filters must satisfy.
- `gospa-metric`: GOSPA (alpha=2) with exact localization / missed / false decomposition over per-frame ground-truth-vs-track sets, reported alongside the existing MOT metrics.
- `consistency-benchmark-gates`: optional NEES/NIS bound assertions in benchmark scenario TOMLs, enforced by `check_regression` next to the existing MOTA/HOTA/IDF1 baselines.

### Modified Capabilities

- `state-estimation`: new requirement — KF, EKF, and UKF update paths SHALL expose the innovation and innovation covariance S actually used in the update (plus predicted state/covariance for NEES), non-breaking for existing callers.
- `cubature-kalman-filter`: the CKF SHALL expose the same update diagnostics, preserving its spec'd surface parity with the UKF.

Note: `evaluation-metrics` is intentionally **not** modified — its existing requirements (MOTA/IDF1/HOTA/AMOTA and report generation) are unchanged; the "Evaluation report generation" requirement already covers "all computed metrics", and the new metrics arrive as new capabilities rather than edits to the MOT requirements.

## Impact

- **Crates**: `thresh-eval` gains consistency and GOSPA modules plus report wiring; `thresh-filter` adds update-diagnostics exposure to `kf.rs` / `ekf.rs` / `ukf.rs` / `ckf.rs` (non-breaking for every in-repo caller — `update()` currently returns `()` everywhere and all call sites are statement-position); `thresh-data` extends `Baselines` (serde-defaulted optional fields, backward-compatible with existing TOMLs) and `check_regression` in `benchmark.rs`.
- **Dependencies**: none added — pure Rust, nalgebra math, chi-squared quantiles computed in pure f64 (Wilson–Hilferty-seeded Newton refinement of the regularized incomplete gamma — design.md Decision 2); GOSPA assignment reuses `thresh-association`'s Hungarian solver.
- **Consumers**: `run_orbital_benchmark` and the other runners converge on `build_benchmark_result`, which gains optional consistency outputs; the in-flight `orbital-ballistic-filter-models` change (its `orbital-ballistic-benchmarks` gate calibration) is the first intended consumer.
- **CI**: plain `cargo test`, no new feature gates, no GPU; clippy `-D warnings`; SonarCloud cognitive complexity ≤ 15 via phase-helper decomposition per the CLAUDE.md style guide.
- **Breaking**: none in-repo. The filter diagnostics arrive as a new `update` return value (`()` → struct is source-compatible at every in-repo call site, all statement-position), and existing scenario TOMLs parse unchanged. One public-surface caveat, flagged for completeness: the `LeafFilter` trait's `update_linear` return type changes in lockstep (design.md Decision 1), which is breaking only for external implementors of that trait — none exist (pre-1.0, org-internal).
