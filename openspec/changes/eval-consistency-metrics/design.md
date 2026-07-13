# Design — Evaluation Consistency Metrics (NEES / NIS / GOSPA)

## Context

`thresh-eval` measures assignment quality only. Its whole input surface is `FrameData { gt: Vec<(u64, [f64; 3])>, tracks: Vec<(u64, [f64; 3])> }` (`matching.rs`), matched per frame by Euclidean distance through `thresh_association::hungarian::hungarian_assignment(cost, gate)`, and every metric (MOTA/MOTP/IDF1 in `metrics.rs`, HOTA in `hota.rs`, the incremental `MotMetricsBuilder`) is a function of those matches. Covariances never enter: a filter reporting P ≈ 0 or P ≈ ∞ scores identically to an honest one.

On the filter side the diagnostics needed for consistency metrics are computed and thrown away:

- `KalmanFilter::update` (`kf.rs:39`) computes the innovation `y` and `S = H P Hᵀ + R` internally; the public `innovation()` / `innovation_covariance()` helpers are **pre-update recomputations** that require the caller to re-supply `h`/`r` and are only correct if called before `update` mutates `x`/`p`.
- `ExtendedKalmanFilter::update` (`ekf.rs:36`) has no diagnostics API at all.
- `UnscentedKalmanFilter::update` (`ukf.rs:133`) and `CubatureKalmanFilter::update` (`ckf.rs:103`) compute a **sigma-point/cubature-point S** (`Σ wc·Δz·Δzᵀ + R`) that a caller *cannot* reconstruct: it depends on the internal point spread and the `h_fn` closure evaluations, not on any `H` matrix.
- All four `update` paths return `()`, and all four filter structs have all-`pub` fields (externally struct-literal-constructible).

In the tracker, `apply_measurement_update` (`tracker.rs:621`) constructs an **ephemeral** `KalmanFilter` per track per update, calls `update`, copies `x`/`p` back onto the `Track`, and drops the filter — so post-update innovations are unreachable from outside `MultiObjectTracker` today.

In `thresh-data/src/benchmark.rs`, all four runners (synthetic / ADS-B / orbital / nuScenes) converge on `build_benchmark_result` (`:240`), and `check_regression` (`:354`) gates `BenchmarkResult` against `Baselines { mota, hota, idf1 }` — three `Option<f64>` floors.

The in-flight `orbital-ballistic-filter-models` change makes this gap acute: it introduces tunable process noise (`sigma_accel`, `sigma_beta`) and its section 6 recalibrates the orbital MOTA gates. Without NEES/NIS there is no evidence that a Q tuning that clears a MOTA floor is statistically honest rather than gamed.

Constraints inherited from the repo: pure Rust, nalgebra, **no new dependencies**; clippy `-D warnings`; SonarCloud cognitive complexity ≤ 15 via phase-helper decomposition; workspace layering: `thresh-eval` depends only on `thresh-core` and `thresh-association` — in particular **not** on `thresh-filter` or `thresh-tracker`, and nothing in the tracker can consume eval types (which is why Decision 5's tracker accumulator is a local field, not the eval crate's type) — while `thresh-data` depends on `thresh-tracker` and `thresh-eval` and is where the two meet.

## Goals / Non-Goals

**Goals:**

- Expose post-update innovation and innovation covariance S (the values *actually used* in the update) from KF, EKF, UKF, and CKF without breaking any existing caller.
- NEES and NIS per Bar-Shalom's standard formulation — per-track and time-averaged (ANEES/ANIS) — with two-sided chi-squared bounds at a configurable confidence level (95% default) and correct degrees-of-freedom accounting.
- GOSPA (Rahmathullah/García-Fernández/Svensson 2017, α = 2) with its exact localization / missed / false decomposition, computed over the existing `FrameData` sequence via the existing Hungarian solver.
- Optional NEES/NIS bound assertions in benchmark scenario TOMLs, enforced by `check_regression`, adoptable by the orbital change's section-6 calibration with a one-way (non-circular) ordering.

**Non-Goals:**

- Trajectory-GOSPA (needs a trajectory-level tracker; deferred), OSPA(2), PCRLB.
- NIS whiteness / Ljung-Box tests (belongs to the later time-correlated-noise work).
- NIS for the IMM and JPDA update paths (mixture/weighted innovations are not χ²-distributed under the single-Gaussian assumption — see Decision 5).
- Any tracker or association algorithm changes beyond the minimal diagnostics plumbing; any re-baselining of existing scenario MOTA/HOTA/IDF1 values (the orbital change's section 6 owns that).

## Decisions

### Decision 1: Filter diagnostics — `UpdateOutcome` returned by value from every `update` path

**Decision:** all four filters' `update` (and their `update_linear` convenience wrappers, which delegate) change return type from `()` to a shared struct in `thresh-filter`:

```rust
/// Diagnostics from a completed measurement update.
pub struct UpdateOutcome {
    /// Innovation y = z − ẑ actually applied (pre-fusion residual).
    pub innovation: DVector<f64>,
    /// Innovation covariance S actually used to form the gain
    /// (H P Hᵀ + R for KF/EKF; the sigma-/cubature-point moment for UKF/CKF).
    pub innovation_covariance: DMatrix<f64>,
    /// Normalized innovation squared yᵀ S⁻¹ y, computed where S's
    /// factorization already exists.
    pub nis: f64,
}
```

- **Non-breaking by construction.** Every in-repo call site invokes `update` in statement position (`kf.update(&z, &h, &r);`), and `()` → struct is source-compatible there. No `#[must_use]` — under the repo's `-D warnings` that would be a de-facto breaking change.
- **Why return-by-value and not stored `last_*` getters:** all four structs have all-`pub` fields, so they are externally constructible via struct literal (`KalmanFilter { x, p }`); adding a stored field — even a `pub` one — breaks every such literal. Stored diagnostics also cost a clone on every update whether or not anyone reads them, and go stale silently (a getter read after the next `predict` reports the previous update). Return-by-value has zero storage, zero staleness, and the values are moves of matrices `update` already owns.
- **Why this is the only honest exposure for UKF/CKF:** their S is `Σ wc·(zᵢ−ẑ)(zᵢ−ẑ)ᵀ + R` over projected sigma/cubature points (`ukf.rs:149-160`, `ckf.rs:119-127`). It is not `H P Hᵀ + R` for any `H`, and the caller cannot reproduce it without re-running the point projection through the `h_fn` closure against the *pre-update* `(x, p)` — which `update` has already overwritten.
- **NIS computed inside `update`:** KF/EKF already form `s_inv` for the gain; UKF/CKF hold S's LU factorization from the gain solve. Computing `yᵀ S⁻¹ y` there avoids a second O(m³) factorization downstream and guarantees the NIS uses exactly the S that produced the gain.
- **KF's pre-update helpers stay.** `innovation()` / `innovation_covariance()` remain for gating use (compute-before-commit); their doc comments gain a cross-reference distinguishing "hypothetical residual for a candidate z" from "diagnostics of the update that happened".
- **NEES hooks require no new API.** `x` and `p` are `pub` on all four filters; predicted-state NEES reads them after `predict` and before `update`, posterior NEES reads them after `update`. This is documented as the contract in the delta spec rather than adding accessors that alias public fields.
- **`LeafFilter` follows suit.** The pluggable-leaf trait (`imm.rs:267`) declares `fn update_linear(...)` returning `()` and its three impls delegate in tail position (`ExtendedKalmanFilter::update_linear(self, z, h, r)` as the final expression), so they would stop compiling if left alone. The trait method's return type changes to `UpdateOutcome` too, keeping the leaf trait signature-parallel with the inherent methods — and making per-mode NIS reachable when IMM consistency work lands later (Open Questions). Its only implementors are the three in-repo filters; the trait change is breaking only for hypothetical external leaf implementors (pre-1.0, org-internal). The IMM's own `update_step` likelihood math is untouched.

**Alternatives considered:**

- *Stored `last_innovation` / `last_s` fields + getters.* Breaks struct-literal construction (above), pays clone cost unconditionally, staleness hazard. Rejected.
- *Callback/observer parameter on `update`.* A genuinely breaking signature change on all callers, and over-engineered for "hand me two matrices".
- *Parallel `update_with_diagnostics` methods.* Doubles the update surface per filter (8 methods), and the two paths inevitably drift; the whole point is that diagnostics come from the *one* code path that ships.

### Decision 2: NEES/NIS formulation — Bar-Shalom scalars, ANEES/ANIS accumulators, Newton-refined chi-squared bounds

**Decision:** new module `thresh-eval/src/consistency.rs` (with a `chi2` submodule for the quantile math, unit-tested in isolation per the phase-helper testing convention):

- **Scalars:** `nees(error: &DVector, p: &DMatrix) -> Option<f64>` computing `eᵀ P⁻¹ e` (None on singular P — an eval crate must not panic on a degenerate input it is being asked to judge), and `nis(innovation: &DVector, s: &DMatrix) -> Option<f64>`. Callers holding an `UpdateOutcome` use its precomputed `nis` field; the free function serves recorded/serialized innovation streams.
- **Accumulator:** `ConsistencyAccumulator { sum: f64, n: usize, dof: usize }` with `push(sample)`, `mean()` (the ANEES/ANIS), and `verdict(alpha) -> ConsistencyVerdict`. Per-track variants are just one accumulator per track ID (a `HashMap<u64, ConsistencyAccumulator>` helper), time-averaged is one accumulator over everything — same type, no parallel code paths.
- **Bounds and dof accounting (Bar-Shalom, *Estimation with Applications to Tracking and Navigation*, §5.4):** if each sample is χ²(d) and N samples are averaged, `N·ANEES ~ χ²(N·d)`, so the two-sided acceptance interval at confidence `1−α` is `[χ²_{N·d}(α/2)/N, χ²_{N·d}(1−α/2)/N]`. Default `alpha = 0.05` (the two-sided 95% convention). The verdict is three-valued — `Consistent`, `Overconfident` (above the upper bound: reported covariance too small), `Underconfident` (below the lower bound: covariance inflated, e.g. Q pumped up to game association) — because *both* tails are failures for falsifiability purposes.
- **Chi-squared quantiles via a Wilson–Hilferty seed refined by Newton on the incomplete gamma:** the initial guess `χ²_k(q) ≈ k·(1 − 2/(9k) + z_q·√(2/(9k)))³` — with `z_q` from a standard rational approximation of the inverse normal CDF (Acklam's, ~15 lines, pure f64) — is refined by Newton iterations on the regularized lower incomplete gamma `P(k/2, x/2)` (series for `x < k/2 + 1`, continued fraction otherwise — the standard *Numerical Recipes* `gammp` split; ~60 lines, pure f64, fixed iteration cap and tolerance so evaluation is deterministic). No new dependency, valid for arbitrary dof — which matters because ANEES bounds need dof = N·d where N is a run's sample count (thousands). The refinement is not optional polish: raw Wilson–Hilferty upper-tail quantiles are within ~1% for all dof ≥ 2, but its **lower-tail** (α/2) quantiles degrade badly at small dof (≈ −7% at dof 4 and ≈ −48% at dof 2 for the 2.5% quantile, worse at 99% confidence), which would break both the spec's small-dof reference-quantile tests and per-track verdicts on short tracks. The refined quantile is accurate to near machine precision across the full dof range; unit tests pin hardcoded reference quantiles (e.g. χ²₃(0.975) = 9.3484, χ²₁₀₀(0.975) = 129.561), both tails, at the spec's 1% relative tolerance — which the refinement passes with orders-of-magnitude margin.

**Alternatives considered:**

- *Checked-in quantile table.* Cannot cover dof = N·d for arbitrary run lengths (or configurable confidence levels) without interpolation code that ends up larger than the quantile math itself.
- *Raw Wilson–Hilferty with no refinement.* Smallest possible code, but its small-dof lower-tail error (−48% at dof 2 for the 2.5% quantile) violates the delta spec's 1%-relative reference-quantile requirement at dof 2 and 4; kept only as the Newton seed.
- *Bisection instead of Newton for the refinement.* Equally dependency-free and guaranteed to converge, but needs the same incomplete-gamma CDF anyway and more iterations; Newton from the Wilson–Hilferty seed converges in a handful of steps, and the fixed iteration cap keeps it just as deterministic.
- *`statrs`/`special` crate.* Violates the no-new-deps constraint for one formula.
- *One-sided (upper) bounds only.* Rejected — the underconfidence tail is precisely the "inflate Q until MOTA passes" failure mode this change exists to catch.

### Decision 3: NEES truth alignment — reuse `match_frame`, new `EstimateFrame` input, position-marginal NEES in pipelines

**Decision:** NEES needs a truth↔estimate pairing plus the estimate covariance; NIS needs neither. The data flow splits accordingly:

- **New input type** in `consistency.rs` (not a change to `FrameData` — see alternatives):

  ```rust
  pub struct TrackEstimate {
      pub id: u64,
      pub state: DVector<f64>,       // full filter state
      pub covariance: DMatrix<f64>,  // full P
  }
  pub struct EstimateFrame {
      pub gt: Vec<(u64, [f64; 3])>,  // same shape as FrameData.gt
      pub tracks: Vec<TrackEstimate>,
  }
  ```

- **Alignment reuses the MOT matcher.** `sequence_anees(frames: &[EstimateFrame], pos_idx: [usize; 3], dist_threshold: f64)` projects each `TrackEstimate` to a position via `pos_idx` (the tracker's interleaved convention is `[0, 2, 4]`, exported as a const; the orbital change's 7D ballistic state uses the same first-six layout, so no change needed there), builds a `FrameData`, and calls the existing `match_frame` with the **same `dist_threshold` the runner uses for MOTA**. Consequence, stated in the spec: the NEES population is exactly the MOTA true-positive population. NEES over matched pairs only is the standard practice — unmatched truth/track cardinality errors are MOTA's and GOSPA's job, not NEES's.
- **Position-marginal NEES (dof = 3) is the pipeline variant.** Every benchmark runner's ground truth is position-only (`[f64; 3]`), so per-pair NEES is `eᵀ P_pos⁻¹ e` over the 3×3 position block of P extracted via `pos_idx` — inverting the extracted marginal block, which is the correct marginal covariance of the position components. Full-state NEES stays available as the plain `nees(e, P)` function for Monte-Carlo filter tests where synthetic truth includes velocities (and, later, β).
- **NIS is filter-internal** and never touches this machinery: its samples come straight from `UpdateOutcome` streams (Decision 5 for the benchmark plumbing).

**Alternatives considered:**

- *Extend `FrameData` with optional covariance.* `FrameData` has all-`pub` fields and is literal-constructed at ~10 sites across four runners and every eval test; an added field breaks all of them, and the MOT metrics would carry a field they never read. A parallel type keeps MOT plumbing untouched (explicit scope boundary).
- *Mahalanobis matching for the NEES pairing.* Statistically appealing (match by the covariance being judged) but circular — an overconfident filter would shrink its own gate and select its best samples; and it makes the NEES population diverge from the MOTA population, so gate failures become uninterpretable. Euclidean matching with the shared threshold keeps one population under both gates.
- *NEES in `thresh-filter`.* NEES needs truth and assignment machinery; the filter crate has neither and should stay estimation-only. Eval is where truth lives.

### Decision 4: GOSPA — α = 2 fixed, `(c, p)` parameters, assignment via existing Hungarian with cost `dᵖ` gated at `cᵖ`

**Decision:** new module `thresh-eval/src/gospa.rs`:

```rust
pub struct GospaParams { pub c: f64, pub p: f64 }   // cutoff (state units), order; default p = 2
pub struct GospaResult {
    pub gospa: f64,           // (loc + missed + false)^(1/p)
    pub localization: f64,    // Σ_matched d^p
    pub missed: f64,          // (c^p / 2) · |unmatched gt|
    pub false_tracks: f64,    // (c^p / 2) · |unmatched tracks|
    pub n_matched: usize, pub n_missed: usize, pub n_false: usize,
}
pub fn gospa_frame(frame: &FrameData, params: &GospaParams) -> GospaResult;
pub fn gospa_sequence(frames: &[FrameData], params: &GospaParams) -> GospaSummary; // per-frame results + order-p mean + summed decomposition
```

- **α = 2 is fixed, not a parameter.** Per Rahmathullah et al. 2017 (Proposition 1), α = 2 is the unique choice for which the metric decomposes exactly into localization + missed + false with per-target penalty `cᵖ/2` — the decomposition *is* the contract this change ships. Other α values are documented as intentionally unsupported.
- **Assignment reuses `hungarian_assignment` exactly.** Cost matrix `cost[i][j] = d(gtᵢ, trackⱼ)ᵖ`, gate `cᵖ`. This yields the GOSPA-optimal assignment, not an approximation: the solver treats `cost ≥ gate` as infeasible and pads to square with gate-valued dummies (`hungarian.rs:141-147`, `:660`), so its internal objective is `Σ_matched dᵖ + cᵖ·(dim − m)` for `m` real matches, which differs from the GOSPA objective `Σ_matched dᵖ + (cᵖ/2)(|X| + |Y| − 2m) = Σ_matched dᵖ − cᵖ·m + const` by a constant independent of the assignment. Minimizers coincide. A unit test pins this equivalence on a hand-solved 3×3 instance, and a property test asserts the decomposition identity `gospaᵖ = loc + missed + false` to 1e-12 on random frames.
- **Per-frame over the existing `FrameData`** — GOSPA is a set metric per time step; the sequence summary reports the per-frame values plus the order-p mean of the per-frame totals and the summed decomposition components (matching the delta spec's summary contract). No cross-frame assignment terms of any kind (that is trajectory-GOSPA, explicitly out of scope).
- **`c` is per-call, no global default.** The repo's regimes span metres (nuScenes) to tens of kilometres (orbital); each caller supplies `c` from its context — benchmark runners derive it from the same expression as their MOTA `dist_threshold` so the two metrics agree on what "close enough" means.

**Alternatives considered:**

- *Expose α.* Invites configurations whose decomposition fields would be silently wrong; fixed α = 2 with a doc note is safer than a validated-at-runtime parameter.
- *Auction algorithm for assignment.* New solver code when a gated Hungarian already exists in-tree and thresh-eval already depends on it; the equivalence argument above removes the only reason to want a bespoke solver.
- *Gate on `d ≥ c` with cost `d`.* Produces the optimal assignment for p = 1 only; costing `dᵖ` and gating `cᵖ` is correct for all p and identical at p = 1.

### Decision 5: Benchmark integration — tracker-level ANIS accumulator, `EstimateFrame` collection in runners, explicit numeric bounds in `Baselines`

**Decision:** four additive pieces:

1. **ANIS plumbing (the one tracker touch).** Post-update innovations exist only inside `apply_measurement_update`, which now receives an `UpdateOutcome` for free as the return value of `kf.update(...)`. `MultiObjectTracker` gains a **private** `nis_accumulator: ConsistencyAccumulator`-shaped field (sum, count, measurement dof) fed on the single-model KF path, a public read accessor, and a reset. This is the minimal plumbing the filter accessors exist to feed: one field, ~4 lines, no signature or behavior change. **The IMM path is excluded** — the moment-matched mixture innovation is not Gaussian with a single S, so its "NIS" is not χ²(m) and gating it would be pseudo-statistics; same for JPDA's probability-weighted updates. Every benchmark runner uses `MultiObjectTracker::new_cv_position` (the Hungarian + KF path), so all four runners get ANIS. Excluded paths are documented in the delta spec and left to the later time-correlated-noise / IMM-consistency work.
2. **ANEES collection in runners.** A sibling of `collect_confirmed_track_positions` — `collect_confirmed_track_estimates` — additionally copies `track.state` / `track.covariance` (both already `pub` on `thresh_tracker::track::Track`; no thresh-core or thresh-tracker change) into `TrackEstimate`s. Runners accumulate `Vec<EstimateFrame>` alongside the existing `Vec<FrameData>`, and `build_benchmark_result` (a `pub(crate)` helper — free to change) gains the estimates parameter and computes ANEES via Decision 3, ANIS from the tracker accessor, and the sequence GOSPA (reported, not gated).
3. **Schema:** `BenchmarkResult` gains `anees: Option<f64>`, `anees_samples: usize`, `anis: Option<f64>`, `anis_samples: usize`, `gospa: Option<f64>`; `Baselines` gains four serde-defaulted optional fields:

   ```toml
   [baselines]
   mota = 0.5
   anees_min = 1.8   # optional, two-sided
   anees_max = 4.6
   anis_min = 2.1
   anis_max = 3.9
   ```

   `check_regression` enforces each present bound (phase-helper: one `check_bound(name, value, min, max, &mut failures)` helper keeps complexity flat as the metric count grows). Existing TOMLs parse unchanged. **Fail-loud rule:** if a TOML asserts an `anis_*`/`anees_*` bound but the run produced zero samples, that is a gate *failure*, not a silent pass — otherwise a plumbing regression reads as consistency.
4. **Explicit numeric bounds, calibrated, not α-in-TOML.** The TOML carries concrete numbers like the existing `mota` floors — greppable, deterministic under the runners' seeded RNGs, and interpretable in review. To make calibration trivial, the runner prints the observed ANEES/ANIS **next to the theoretical χ² bounds for that run's N·d** (Decision 2 machinery), so setting bounds is "run, read, copy, add margin" — the same procedure the orbital change's Decision 7 established for MOTA floors, with the printed theoretical interval as the anchor.

**Adoption by `orbital-ballistic-filter-models` section 6 (no circular dependency):** this change edits **no existing baseline values** and adds bounds to exactly one scenario TOML — `synth-cv-clean`, where the CV tracker on CV truth with known noise is the textbook case whose ANEES/ANIS genuinely should sit inside the 95% interval — as the in-repo demonstration and CI exercise of the gate. The orbital change's section-6 calibration then *sets values* for the orbital/ballistic TOMLs: pure data edits against schema this change already landed. The code of the two changes is mutually additive (neither edits the other's files); only section 6's TOML data edits must land after this change's schema — a one-way ordering, not a cycle.

**Alternatives considered:**

- *Accumulate NIS on `Track` (thresh-core).* `Track` is an all-`pub`-field core type literal-constructed across the workspace; adding a field breaks every constructor for a value only eval reads. The tracker-level scalar accumulator has no such blast radius.
- *Return per-step diagnostics from `MultiObjectTracker::step`.* Breaking signature change on the hottest API in the repo to plumb an eval concern; rejected under the "no tracker changes beyond minimal plumbing" scope line.
- *Recompute NIS inside the benchmark from tracker outputs.* Impossible: the predicted state/covariance at each update instant is consumed inside `step` and gone by the time the runner regains control.
- *Confidence level in TOML with bounds computed at runtime from N·d.* Statistically purer, but the gate then silently moves whenever a scenario's duration or visibility window changes sample count, and review can't see what the gate actually is. The printed theoretical interval gives calibration the purity; the TOML keeps the gate explicit.

### Decision 6: Module layout and report wiring in `thresh-eval`

**Decision:** two new modules, no reshuffling of existing ones:

```
crates/thresh-eval/src/
  consistency.rs   // nees, nis, ConsistencyAccumulator, verdicts, EstimateFrame,
                   //   sequence_anees; chi2 quantiles as a `chi2` submodule
  gospa.rs         // GospaParams, gospa_frame, gospa_sequence, decomposition types
```

`lib.rs` re-exports the headline types (`ConsistencyVerdict`, `GospaResult`, …) alongside the existing `MotMetrics` re-export. `EvalReport` (`report.rs`) gains two optional sections:

```rust
#[serde(default, skip_serializing_if = "Option::is_none")]
pub gospa: Option<GospaReportSection>,
#[serde(default, skip_serializing_if = "Option::is_none")]
pub consistency: Option<ConsistencyReportSection>,
```

Old serialized reports deserialize unchanged (`default`); new reports without the metrics serialize byte-identically to today (`skip_serializing_if`); `to_table` appends the sections only when present. This satisfies the live `evaluation-metrics` spec's "report covers all computed metrics" requirement without modifying that spec — the new metrics arrive as new capabilities (`filter-consistency-metrics`, `gospa-metric`), matching the proposal's capability split.

**Alternatives considered:**

- *One `consistency` module containing GOSPA too.* GOSPA is a set-distance metric needing no covariances or filter diagnostics; lumping it with NEES/NIS obscures that its only input is the same `FrameData` the MOT metrics use. Two modules mirror the two new capabilities one-to-one.
- *Standalone report types not attached to `EvalReport`.* Forks the reporting surface; every consumer would need to merge two JSON documents. Optional sections keep one report.

## Risks / Trade-offs

- **[`()` → `UpdateOutcome` return-type churn]** — statement-position callers compile unchanged, but any downstream code binding the unit return (`let x: () = kf.update(...)`) breaks. → Mitigation: repo is pre-1.0 and org-internal (release-decisions memory); a workspace grep (done during design) found every direct call site in statement position — tracker (`tracker.rs:651`, `ecef_tracker.rs:255`, `cost_matrix.rs:200`), filter tests/benches, and the `thresh-py` wrapper (whose own `update` returns `PyResult<()>` and discards the inner return) — with the sole exception of the tail-position `LeafFilter` impls, handled in Decision 1; no `#[must_use]`.
- **[NEES population survivor bias]** — NEES over matched pairs only: a diverged filter whose tracks fall outside the match gate contributes no NEES samples, so a terrible tracker can look "consistent". → Mitigation: consistency gates are only meaningful **alongside** the MOTA floor (a scenario must pass both); `anees_samples` is reported and the fail-loud zero-sample rule (Decision 5) catches total collection failure.
- **[Quantile-refinement correctness and determinism]** — the Newton-refined incomplete-gamma quantile is more code than a bare approximation, and an unbounded iteration would be a determinism hazard. → Mitigation: fixed iteration cap and tolerance (no input-dependent convergence branching); reference-quantile unit tests pin both tails at dof 2/4/100/400 for 95% and 99% confidence; the `chi2` submodule is isolated and unit-tested per the phase-helper convention.
- **[ANIS covers only the single-model KF path]** — a future IMM-configured benchmark would assert `anis_*` bounds and get zero samples. → Mitigation: exactly the fail-loud rule — the gate errors rather than passing; the exclusion and its statistical reason are in the delta spec.
- **[`dᵖ` cost magnitudes in the orbital regime]** — km-scale distances squared (~1e8) are far from f64 limits but worth a glance at the Hungarian's reduction arithmetic. → Mitigation: a GOSPA unit test runs at orbital magnitudes (c = 5e3 m) and cross-checks against a brute-force assignment.
- **[Gate proliferation in `check_regression`]** — four new bound checks risk tripping the cognitive-complexity gate. → Mitigation: `check_bound` phase helper (Decision 5); the function becomes a linear list of helper calls.
- **[ANEES on interpolated truth]** — the ADS-B runner's ground truth is interpolated to a 1 Hz grid; interpolation error inflates NEES beyond the filter's fault. → Mitigation: nothing in the schema forces bounds onto ADS-B scenarios; this change gates only `synth-cv-clean` (exact truth) and the guidance note in the delta spec restricts ANEES gating to exact-truth sources (synthetic, SGP4-propagated).

## Migration Plan

1. **`thresh-filter`:** `UpdateOutcome` + return-type change on KF/EKF/UKF/CKF `update`/`update_linear`, NIS computed at the existing factorization. Tests: EKF-vs-KF S identity on a linear problem; UKF/CKF outcome S symmetric-PD and consistent with the applied gain; NIS of a zero innovation is 0.
2. **`thresh-eval`:** `consistency.rs` (chi2 submodule first, then scalars/accumulators/`EstimateFrame`/`sequence_anees`) and `gospa.rs`; report sections. Tests: reference quantiles; a Monte-Carlo KF on matched-Q synthetic truth lands inside the 95% ANEES/ANIS interval, a deliberately overconfident tuning (`Q × 0.01`) lands above the upper bound, and an underconfident one (`Q × 100`) below the lower bound (the falsifiability demonstration); GOSPA hand-solved instances + decomposition identity.
3. **`thresh-tracker`:** private ANIS accumulator + accessor fed by the KF path's `UpdateOutcome`.
4. **`thresh-data`:** `Baselines`/`BenchmarkResult`/`check_regression` extension, `collect_confirmed_track_estimates`, runner collection, calibration print (observed vs theoretical bounds), then calibrate and commit `synth-cv-clean` bounds last.

Each step is additive and independently revertible. **Rollback:** reverting step 4's schema requires deleting the four optional TOML keys from `synth-cv-clean` (data-only); steps 1–3 revert without data changes — step 1's return type reverts to `()` with no caller edits (all in-repo callers remain statement-position) beyond restoring the `LeafFilter` trait signature alongside it.

## Open Questions

- **GOSPA `c` and consistency bounds for orbital/ballistic scenarios** — owned by the orbital change's section-6 calibration runs; this change only guarantees the printed theoretical intervals make that calibration mechanical.
- **Two-point-difference birth covariance honesty** — track birth seeds P from head config, not from measurement geometry; early-life NEES samples may run hot for reasons unrelated to Q. Decide during calibration whether ANEES should skip the first k post-confirmation samples per track (a `warmup` knob on `sequence_anees`) or whether confirmed-only collection already suffices.
- **Per-class consistency breakdowns** — `per_class.rs` splits MOT metrics by `TargetClass`; extending ANEES the same way is mechanical but adds report surface. Defer until a multi-class scenario actually gates consistency.
- **IMM mixture consistency** — per-mode NIS against per-mode S is well-defined and could feed a mode-matched consistency test later; recorded here so the time-correlated-noise / IMM work picks it up rather than re-deriving the exclusion.
