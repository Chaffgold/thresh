# Design

## Context

See `proposal.md` for motivation and scope.

**Baseline.**
- The baseline is `develop` at `3fec105` (#167). Since the planning base `c2b9e4f` (#169), only dependency locks changed (#165, #167: Pillow 13.0.0 and the Python training lock); the pins below and every cited source line are unchanged.
- The pinned bridge runtime is CPython 3.12.13, Stone Soup 1.9.1, NumPy 2.5.3, SciPy 1.18.1 and PyO3 0.29.0, hash-locked in `crates/thresh-bridge/tests/requirements.txt`.
- Planning uses OpenSpec 1.13.2.

**Evidence.**
- The maintainer's binding decisions of 2026-11-23 rest on research evidence gathered for P0-C.
- That evidence consists of four research probes and three adversarial verification passes. The scripts were ephemeral and are not part of the repository.
- Every number this change depends on is recorded below. The committed tests re-derive the numbers that become tests.
- Each path:line below was re-checked against `c2b9e4f`.

**Bridge.**
- `ImmFilter::new` imports `stonesoup.predictor.interacting` (`crates/thresh-bridge/src/imm.rs:59`). That module is absent in 1.9.1, so the call returns `UnsupportedApi`.
- CI asserts this failure: `initialized_imm_constructor_reports_unresolved_api` (`tests/runtime_contract.rs:374-388`) passes in job 137895907234 (run 45232621968).
- **Upstream:** upstream `main` has no `predictor/interacting`. Open upstream PR #1257 adds `stonesoup.predictors`, `stonesoup.augmentor` and `stonesoup.reducer`, but no `IMMPredictor`.
- `MhtTracker::new` builds `MultiTargetMixtureTracker(max_num_hypotheses=…, prune_threshold=…)` (`src/mht.rs:35-59`).
  - That call raises `TypeError: … missing a required argument: 'initiator'`.
  - The wrapper's step calls `track`, but the class only has `update_tracker`.
  - `MhtConfig` has no window parameter (`src/mht.rs:9-25`).
  - No manifest test reaches a real Stone Soup MHT.

**Native MHT** (`crates/thresh-association/src/mht.rs`).
- One shared likelihood matrix is applied to every parent (`:65`, `:83-103`). Expansion therefore ignores per-hypothesis history.
- `prune_n_scan` (`:138-175`) has no caller outside the module.
- The tracker's `step_mht` calls only `expand` and `prune_k_best`, and gates at `clutter.ln()` (`crates/thresh-tracker/src/tracker.rs:644-649`).
- With the Unknown head's birth covariance of 1e4 (`crates/thresh-tracker/src/heads.rs:158`), the MHT path never associates.
- Change 2 owns all of these (Decision 10, Follow-up changes).

**Native IMM** (`crates/thresh-filter/src/imm.rs`).
- The core arithmetic agrees with an independent reference, but several mode policies are defective (Evidence summary).
- `ImmConfig::validate` (`:486-539`) checks only |row sum − 1| ≤ 1e-6 (`:511-517`) and |Σμ − 1| ≤ 1e-6 (`:519-525`). NaN entries pass, because every comparison involving NaN is false. Negative entries pass whenever their row still sums to one.
- `with_leaf_kind` panics on an invalid configuration through `validate().expect` (`:737`). Neither it, `ImmFilter::new` nor `MultiObjectTracker::new_imm_position` (`tracker.rs:207-215`) documents the panic.
- The learned-IMM tracker constructor validates first and maps any error to `Err` (`crates/thresh-tracker/src/tracker.rs:272-274`). It then enforces a 4-model bank (`:275-285`).
- Two thresholds treat a small predicted probability as zero, and they disagree (part of F4):
  - mixing computes μ_i|j only when c_j > 1e-30 (`:842`), leaving a lower-probability mode's mixed state and covariance at zero;
  - the mode update clamps c_j at 1e-300 before taking its logarithm (`:996`).
  - Consequence: with floor 0, a mode with c_j = 0 still takes the posterior whenever its log-likelihood exceeds the others' by more than about 690.8 nats. A review probe (both leaves far from a measurement at the origin, with mode 2's zeroed mixed estimate at the origin; Π = [[1,0],[1,0]]; setup in Decision 7) gave μ⁺ = [0, 1] where the exact posterior is [1, 0].
- The learned classifier overwrites the bank's public mode probabilities and then calls `combine()` (`crates/thresh-filter/src/imm_adapter.rs:245-247`).

**Boundary gate.**
- The gate requires 109 tests across 8 targets (`tests/boundary-test-manifest.json`).
- Eight of those tests use the wrappers as test vehicles: manifest `:10`, `:13`, `:114`, `:115`, `:117`, `:118`, `:119` and `:123`.
- The wrappers have no other callers. `AssociationStrategy::Mht` is used only in `tracker.rs`.

**Specs.**
- `stonesoup-bridge` "Stone Soup algorithm access" (`openspec/specs/stonesoup-bridge/spec.md:18-27`) is unmet for both of its scenarios.
- In `mht-tracking`, the new-track option (`:15`), N-scan pruning (`:21-29`) and stable IDs (`:53`) are unmet natively.
- No live spec covers IMM mathematics.
- The archived `imm-estimation` delta (`archive/2026-04-12-imm-filter/specs/imm-estimation/spec.md:13-19`) was never synced. It set a bar of "shift probability mass to the CTRV model within 5 update cycles".
- Two active changes carry pending IMM text that this change must not contradict:
  - `flight-data-training-pipeline` (`specs/learned-tracker-components/spec.md:108-126`): with the feature off, classical IMM "behaves identically" and "SHALL pass the existing IMM test suite without modification". The transition matrix and mixing remain unchanged.
  - `learned-imm-tracker-integration` (`specs/learned-imm-tracker/spec.md:17-39`): a 4-model `cv_ca_ctrv_ct` bank, and an invalid `ImmConfig` returns an error.
- No native GM-PHD exists outside `thresh-bridge`. The bridge's "until Rust-native versions exist" therefore stays meaningful for GM-PHD.

## Goals / Non-Goals

**Goals:**
- Make the live bridge spec true (JPDA and GM-PHD only) through an explicit, approved amendment whose Reason and Migration state the interim gaps.
- Deliver the P0-C acceptance prerequisites in Rust:
  - an exhaustive three-scan MHT reference;
  - hand-derived IMM mixing and update cases.
  Both run in the native lane and are protected by exact names.
- Give IMM mathematics a live spec that the current implementation demonstrably satisfies, without codifying defective policies.
- Fix transition-matrix and mode-probability validation (V).
- Preserve the boundary lane's identity, lock and exact-manifest guarantees. The manifest count changes only deliberately.

**Non-Goals:**
- Changes that belong to Change 2:
  - rebuilding native MHT;
  - any change to existing `mht-tracking` text, its Purpose, or `track-management`;
  - asserting N-scan commit rules or track IDs.
- Changes that belong to Change 3:
  - fixing native IMM defects F1–F8;
  - adding the maneuvering requirement;
  - modifying `imm-pluggable-leaf-filter`;
  - re-baselining learned IMM or evaluation.
- Changes not planned in either follow-up:
  - editing existing IMM tests, the learned-IMM model bank, the ONNX contract, or the text of either active change;
  - editing `stonesoup-bridge` `:7-16`, `:193-195` or `:289`. Decision 3 explains why each stays.
- Backlog items outside P0-C: MFA/OR-Tools lanes, a newer-version watch lane, native/reference parity (P1-A), and sequence evaluation.

## Decisions

### 1. MHT: amend the bridge claim and build an exhaustive Rust reference (M1)

- Remove the bridge MHT claim and delete `thresh_bridge::mht`.
- `mht-tracking` becomes the single MHT contract. It gains an ADDED, contract-neutral exhaustive reference (Decision 6).
- The native rebuild that makes `mht-tracking` true is Change 2.

**Alternatives considered:**
- **M2, a bridge MFA reference** (`MFAHypothesiser` + `MFADataAssociator(slide_window=N+1)`, requiring `ortools==9.14.6206`). Rejected on oracle quality and cost.
  - MFA is inexact: it returned suboptimal window solutions in 3.5–10.1% of steps, with excess up to 7.84 nats.
  - Its solver has a stopping defect and a sign defect.
  - Only W = 1 steps exit with a certificate.
  - It has no births, no k-best and no exposed scores, and it assumes a fixed target set.
  - It needs seven new pins: absl-py, immutabledict, ortools, pandas, a narrow protobuf, typing-extensions and tzdata.
  - It still leaves the native rebuild undone.

  CI feasibility was not the obstacle: a cold install costs +9 s and +149 MB. The decline is recorded as approved deviation D1.
- **M3, amend only.** Rejected because it violates the acceptance prerequisite at review `:406`, "tiny exhaustive three-scan … tests".

### 2. IMM: a native capability with an independent Rust oracle (I-d)

- Remove the bridge IMM claim and delete `thresh_bridge::imm`.
- Add the native `imm-estimation` capability. It specifies only policy-independent mathematics and validation, and verifies them against hand-derived exact rationals and an oracle that shares no code with `thresh-filter`.

**Alternatives considered:**
- **I-a, a bridge-composed IMM: thresh-authored mixing around Stone Soup `KalmanPredictor`/`KalmanUpdater`.** This is the review's primary recommendation (`:194-196`). It is rejected for now:
  - Stone Soup would contribute only per-mode Kalman arithmetic. The parts under test (mixing, mode update, projection) would be a second thresh-authored implementation, in Python.
  - It needs new bridge APIs for construction, prediction, update and missed scans.
  - It cuts against the Rust-native preference.

  It may return as P1-A parity work. Approved deviation D2.
- **I-b, FilterPy `IMMEstimator`.** Rejected (line citations refer to the FilterPy 2.0.0 source):
  - It refuses heterogeneous state dimensions (`IMM.py:139-143`).
  - It shares native's μ-weighted predicted output and stale coasting (`:222-223`), so agreement would hide F5 and F6.
  - It crashes for Π = [[1,0],[1,0]] (`:245-248`).
  - On likelihood underflow it collapses μ to c: [0.62, 0.38] instead of [0, 1].
- **I-c, Stone Soup's particle `MultiModelPredictor`/RB.** Rejected: it is not an IMM, it uses the global RNG (`particle.py:181-183`), and it zero-fills unmapped components (`:205-207`).

### 3. Spec delta mechanics: REMOVED plus a renamed ADDED

**The delta.**
- REMOVE "Stone Soup algorithm access", with **Reason** and **Migration** in the delta. The `hifi-orbital` REMOVED precedent is `archive/2026-07-16-orbit-propagation-fidelity`.
- ADD "Stone Soup reference algorithm access", which keeps the JPDA/GM-PHD obligation and the fallback sentence verbatim.
- `proposal.md` declares the pair a rename.
- One delta covers both bridge claims, so no two deltas touch the same requirement.

**Alternatives considered:**
- **MODIFIED that drops the two scenarios.** Fails `openspec validate --strict`: "omits scenario(s) the current spec still has".
- **RENAMED plus MODIFIED.** Fails strict validation with the same scenario-retention error. This was probed in a scratch sandbox with OpenSpec 1.13.2 on 2026-11-25.
- **Tombstone MODIFIED.** Keeps both scenario headers, rewritten as non-support/migration statements. It validates, but it was rejected: the live spec would keep MHT/IMM scenario titles that describe no supported behaviour. Choosing between the two forms is a judgment about visibility. The archived delta and this design keep the full record.

**Archive consequences** (verified by the research sandbox):
- Archiving REMOVED plus ADDED appends the ADDED requirement at the end of the main spec.
- No Reason or Migration text survives in the main spec.
- OpenSpec ignores a delta `## Purpose` for an existing spec. The installed OpenSpec 1.13.2 CLI only warns, in `dist/core/specs-apply.js:232-247`. The Purpose at `stonesoup-bridge/spec.md:4`, which still names MHT and IMM, is therefore hand-edited in the archive PR, immediately before `openspec archive` (task 7.1), to read:

  > Optional Rust-to-Python bridge, built on PyO3, for instantiating and calling Stone Soup tracking components from Rust, including GIL management and conversion between Rust (nalgebra) types and Stone Soup's numpy/State objects. It exposes Stone Soup's JPDA and Gaussian Mixture PHD algorithms as reference implementations and operational fallbacks, and sits behind the `stonesoup` Cargo feature so that the Rust-native tracking algorithms compile and run without Python or Stone Soup installed.

**Rows intentionally unchanged:**
- **`:7-16` "(filters, data associators, hypothesisers, trackers)".** The list names component kinds the bridge can instantiate in general, not a guarantee for each kind. After the deletion no bridge entry builds a tracker, and the sentence remains accurate.
- **`:193-195` "unresolved JPDA, GM-PHD, MHT, IMM … requirements are not marked complete".** Still true. The MHT and IMM requirements it refers to are now native (`mht-tracking`, `imm-estimation`).
- **`:289` "… not … MHT/IMM correctness".** Still true.

### 4. Delete the bridge modules rather than keep placeholders

Delete `src/{mht,imm}.rs`, their declarations in `lib.rs`, and with them the `thresh::bridge::{mht,imm}` paths re-exported at `crates/thresh/src/lib.rs:6`. This is BREAKING.

**Alternative considered:** placeholders that return `UnsupportedApi`.
- Rejected because a placeholder is a dead public surface that still needs tests and invites misuse.
- If a future upstream release supplies a real contract, a new change can reintroduce the claim.

**P0-B commitments:**
- P0-B `design.md:115` is a rollback rule and does not require keeping the wrappers.
- This change explicitly supersedes P0-B `design.md:110`, which retained MHT fresh-process coverage. `validation_precedes_runtime_and_missing_packages` already provides that coverage by running the `detection_process_child` child in an uninitialized process (Decision 5).
- P0-B `design.md:111` ("preserve IMM's independent unsupported-API regression test") is kept in relabelled form.

There are no non-test Rust callers. External callers migrate as described in the Migration Plan.

### 5. Re-home the eight boundary-manifest vehicles

| Manifest entry (line) | Vehicle today | Disposition |
|---|---|---|
| lib `imm::tests::initialized_predict_preserves_dispatch_and_errors` (`:10`) | IMM wrapper dispatch double | Deleted with the module. The dispatch-double helper `step_test_support::assert_step_contract` keeps one user, JPDA (`src/jpda.rs:238`). |
| lib `mht::tests::initialized_run_preserves_dispatch_and_errors` (`:13`) | MHT wrapper dispatch double | Deleted with the module. |
| `runtime_contract::explicit_host_initialization_is_reused_without_mutation` (`:114`) | JPDA and MHT constructors (`runtime_contract.rs:99-108`) | Keep the name. The vehicle becomes JPDA only, which is already exercised at `:100-103`. |
| `runtime_contract::initialized_imm_constructor_reports_unresolved_api` (`:115`) | IMM constructor (`:374-388`) | Replaced by the lib unit test `runtime::tests::pinned_stonesoup_lacks_interacting_predictor_api`. It calls the crate-private `import_module` (`src/runtime.rs:23`) on `stonesoup.predictor.interacting` and asserts `UnsupportedApi { api: "stonesoup.predictor.interacting" }`. It is documented as a pinned-absence classification regression for 1.9.1, with no upstream-canary claim. |
| `runtime_contract::missing_numpy_is_a_dependency_not_an_interpreter_error` (`:117`) | IMM constructor's NumPy import (`:231-249`) | Keep the name. The vehicle becomes `measurement_to_detection` with a valid context, which reaches the NumPy import at `src/detection.rs:55`. Importing Stone Soup 1.9.1 does not load NumPy (verified), so blocking NumPy fails there with `ModuleNotFoundError(name='numpy')`. |
| `runtime_contract::missing_stonesoup_submodule_is_an_api_error` (`:118`) | IMM `stonesoup.predictor.interacting` import (`:267-287`) | Keep the name. Block `stonesoup.hypothesiser.probability`, which the JPDA constructor imports first (`src/jpda.rs:83`), and assert `UnsupportedApi` naming that module. Placeholder components suffice because the import fails first. |
| `runtime_contract::missing_symbol_is_an_api_error` (`:119`) | MHT `MultiTargetMixtureTracker` lookup (`:289-322`) | Keep the name. Serve an empty `stonesoup.dataassociator.probability` module so that the `JPDA` lookup fails (`src/jpda.rs:116`). The test supplies real components (`reference_context`, `:124-144`) and asserts `UnsupportedApi { api: "JPDA" }`, `AttributeError`, and the operation "JPDA associator lookup". |
| `runtime_contract::token_free_constructors_require_host_initialization` (`:123`) | MHT constructor in a fresh process (`:72-86`) | Deleted. No token-free algorithm constructor remains. Fresh-process coverage is retained by `validation_precedes_runtime_and_missing_packages` (manifest `:74`), which runs the child test `detection_process_child` with case "uninitialized" (`tests/detection_contract.rs:762-766`, `:848-852`). The child entry itself (manifest `:64`) returns immediately when run directly (`:821-824`). `runtime::tests::uninitialized_shared_attachment_is_fallible` (manifest `:36`) is the second anchor. |

**Resulting counts.**
- The expected totals are 37 library tests and 8 `runtime_contract` tests, 106 across 8 targets.
- The exact number is set deliberately during implementation. It is recorded in the manifest, in `docs/bridge-testing.md`, and in the runner's PASSED line.
- A JPDA dependency-error vehicle already exists (`runtime_contract.rs:115-121`).

### 6. MHT reference: fixtures, independence, contract-neutral assertions

**Location.**
- The enumerator lives in `crates/thresh-association/tests/support/mht_oracle.rs`.
- The test target is `crates/thresh-association/tests/mht_reference.rs`, named after the bridge's `jpda_reference` and `phd_reference`.
- The oracle uses `std` only.

**Fixture model (declared by the reference).**
- Each track is a scalar random-walk Kalman filter:
  - predict: P⁻ = P + q;
  - innovation variance: S = P⁻ + r;
  - gain: K = P⁻/S;
  - update: m⁺ = m + K(z − m) and P⁺ = (1 − K)P⁻.
- Each detection is assigned to at most one track, or labelled a false alarm, or labelled new when λ_B > 0. Each track takes at most one detection. Tracks do not die inside the window.
- Gate: (z − m)²/S ≤ g.
- Per-scan scores. These are normalised by λ_FA^M, which is common to every hypothesis in a scan, so rankings and score differences are unchanged:

  | Event | Score |
  |---|---|
  | Assigned | ln(P_D · N(z; m, S) / λ_FA) |
  | Missed | ln(1 − P_D) |
  | False alarm | 0 |
  | New track | ln(λ_B/λ_FA); the new track starts at (z, r) |

- This is the standard likelihood-ratio score for a hypothesis-oriented MHT. Two caveats:
  - It is not native behaviour today. Native scores a miss as 1, a false alarm as 1, and gates at ln λ.
  - It does not bind native behaviour. The spec says so explicitly: a fixture's declared score, birth and gate models define only that fixture's expected values. The model is the proposed input for Change 2's MODIFIED "Hypothesis tree maintenance", where the score terms become normative.

**Fixtures.**
- Common parameters: P_D 0.9, q 0.5 and r 0.5. C1, C2, C3 and C5 have priors A (0, 1) and B (4, 1) and λ_FA 0.05. C4 has a single prior A (0, 0.25).
- All values are at full float64 precision.
- **Size bound.** The spec caps each committed fixture at 100,000 global hypotheses. That keeps births out of long fixtures: the 4-scan 2-D crossing shape has 15,379 leaves without births and 32,232,892 with them. It replaces a three-scan cap. Change 2's planned fixtures (a 5-scan N = 3 case with 243 leaves, and the crossing fixture with λ_B = 0) fit under the bound.

| Fixture | Differences from common | Scans | Leaves | Best cumulative score | Runner-up score (gap) |
|---|---|---|---|---|---|
| C1 deferred resolution | no births, ungated | {2.1}, {3.0}, {5.0} | 27 | −4.056137962656715 | −4.36634826630236 (0.31021030364564517) |
| C2 two-target confirmation | gate 9 | {2.5}, {0.5}, {2.5, 5.0} | 51 | −1.1827855182382336 | −1.3983633581276353 (0.2155778398894017) |
| C3 recent agreement | gate 9 | {2.5}, {5.0}, {−2.0} | 13 | −4.919774326293078 | −7.141592099531 (2.2218177732379196) |
| C4 birth across a miss | λ_FA 0.01, λ_B 0.02, gate 9 | {0.1, 5.0}, {−0.1}, {0.0, 5.2} | 185 | 11.96617537989422 | 11.044462926042648 (0.9217124538515726) |
| C5 empty final scan | no births, ungated | {2.1}, {3.0}, {} | 9 | −6.741456196614258 | −7.37469287881025 (0.6332366821959923) |

**Best and runner-up histories.**
- Notation: z0 and z1 index the scan's detections in the order listed above, and "FA" marks a false alarm. Every runner-up is unique: the gap from rank 2 to rank 3 is at least 0.0597 in every fixture.

| Fixture | Best history (scan 1 / scan 2 / scan 3) | Runner-up history (scan 1 / scan 2 / scan 3) |
|---|---|---|
| C1 | A z0, B miss / A z0, B miss / A miss, B z0 | A miss, B z0 / A miss, B z0 / A miss, B z0 |
| C2 | A z0, B miss / A z0, B miss / A z0, B z1 | A miss, B z0 / A z0, B miss / A z0, B z1 |
| C3 | A miss, B z0 / A miss, B z0 / A z0, B miss | A miss, B miss, z0 FA / A miss, B z0 / A z0, B miss |
| C4 | A z0, new track N from z1 / A z0, N miss / A z0, N z1 | A z0, z1 FA / A z0 / A z0, new track from z1 |
| C5 | A miss, B z0 / A miss, B z0 / A miss, B miss | A z0, B miss / A miss, B z0 / A miss, B miss |

The C4 best history is the birth-across-a-miss case: N is born from the scan-1 detection at 5.0, missed at scan 2, and re-detected at 5.2 in scan 3, while A takes 0.1, −0.1 and 0.0.

**C1 hand anchors.**
- These were derived in closed form and committed as constants:
  - scan 1: −1.7802254585825268 if A takes the detection, −1.5802254585825266 if B takes it;
  - scan 2: −1.228787824770912;
  - scan 3: −1.0471246793032758;
  - total: −4.056137962656715.
- Summation order moves the last digits by up to about 2e-15. For example, the C3 gap is 2.2218177732379196 in one enumeration order and 2.2218177732379214 in another. The 1e-10/1e-9 tolerances absorb this.

**Ties.** Ties are grouped with |Δscore| ≤ 1e-9.

| Fixture | Distinct score groups | Tied groups | Notes |
|---|---|---|---|
| C2 | 50 | 1 | Ranks 10 and 11 tie at −4.74083102492305. |
| C4 | 148 | 31 | The largest group has 4 members. There are 37 adjacent tied pairs. |
| C1, C3, C5 | — | none | — |

The smallest separation between distinct groups is 0.0046 (C4), far above the grouping tolerance.

**Assertions** (contract-neutral):
- **Counts:** leaf counts are exactly as tabled, and each is at most the spec's 100,000 bound.
- **Two enumeration orders:** the reference enumerates by per-track recursion. A second, per-detection label product with a feasibility filter, written inside the test target, must yield the same set of histories, keyed by per-scan events, and the same scores within 1e-12.
- **Exclusivity:** no detection is used twice within any scan, and every existing track has exactly one event per scan.
- **Scores:** each cumulative score equals the sum of its per-scan scores.
- **Best and runner-up:** the best and runner-up histories and scores match the tables (atol 1e-10, rtol 1e-9).
- **Ranking:** ranking is compared as ordered tie groups, including the C2 and C4 group structure. No asserted prefix ends inside a group.
- **MAP:** the maximum-score hypothesis without pruning matches.
- **Not asserted:** any N-scan rule (R1, R2 or R3), retained sets, committed prefixes, IDs, tie-breaks, or native output.

**Independence rules.**
- The oracle is written from the textbook formulation above, not from `mht.rs`. Its enumeration order differs from native `EnumCtx`.
- It names no `thresh_` crate. This is checked mechanically: a test reads the oracle source with `include_str!` and asserts that it contains no `thresh_` path.

**Provenance (recorded here; the scratch scripts are not committed).**
- During the P0-C review, two independently written Python enumerators produced these values: one labels each detection, the other recurses per track.
- A third per-detection enumeration on 2026-11-25 matched every leaf count, best score, best history, gap and tie structure.
- A fourth per-detection enumeration, written on 2026-11-25 during artifact review, matched every leaf count, best score and best history, and produced the runner-up histories and scores above.
- The committed Rust tests are the reproduction. A deliberate perturbation of one constant must fail an assertion; the perturbation is not retained.

**Complexity.** The oracle is decomposed into phases with focused unit tests, keeping cognitive complexity ≤ 15 (CLAUDE.md):

`predict_tracks → enumerate_track_events → label_remaining_detections → score_event → extend_history`

The second enumeration in the test target follows the same rule: `label_product → filter_feasible → score_labels → key_history`.

### 7. IMM verification: oracle independence, specified versus characterisation tests

**Location.**
- The oracle lives in `crates/thresh-filter/tests/support/imm_oracle.rs`.
- Specified behaviour is tested in `crates/thresh-filter/tests/imm_reference.rs`.
- Current-policy characterisation is tested in `crates/thresh-filter/tests/imm_current_policy.rs`.
- The shared fixtures (the asymmetric-mixing estimates, the worked measurement and the zero-destination configuration) live once, in `crates/thresh-filter/tests/support/imm_fixtures.rs`, so the two targets do not duplicate setup code for SonarCloud's duplication check.
- All four are new files.
- **Shared-module rules:**
  - Each target declares `mod support;` with `#[allow(dead_code)]`, because each target uses only part of the shared code and CI denies warnings (`RUSTFLAGS=-Dwarnings`, Clippy `-D warnings`).
  - The support modules contain no `#[test]` functions, so no test is compiled into both targets. The oracle's phase unit tests live in `imm_reference.rs`.

**Oracle independence rules.**
- It imports only `nalgebra` and `std`: no `thresh_filter` or `thresh_core` items. An `include_str!` source check enforces this.
- It declares the model matrices as closed-form fixture models instead of importing them:
  - CV uses discrete white-noise acceleration, σ_a²[[dt⁴/4, dt³/2],[dt³/2, dt²]] per axis.
  - CA uses continuous white-noise jerk, σ_j²[[dt⁵/20, dt⁴/8, dt³/6],[dt⁴/8, dt³/3, dt²/2],[dt³/6, dt²/2, dt]] per axis.
  - Review §3.1 records the CV convention trap against Stone Soup's continuous-time form.
- It implements the textbook update: the marginal (selector [0,1,3,4,6,7]) of a CA state, Joseph update, Cholesky log-determinant, log-sum-exp posterior and moment-matched combination. Its specified-mathematics path treats a zero predicted probability exactly (the mode's posterior weight is zero, with no clamp).
- Every unspecified policy is an explicit oracle parameter, used only by `imm_current_policy`, so each characterisation test states the policy it emulates:
  - CA acceleration reconstruction (mean and variance);
  - the floor and its renormalisation;
  - zero-destination handling: the mixed state and covariance set to zero below the 1e-30 mixing threshold, and the 1e-300 clamp in the mode update;
  - coast propagation;
  - predicted-output weighting.

**What counts as specified.**
- The worked CV+CA cycle passes through CA's re-creation of acceleration (currently mean 0, variance 100 (m/s²)²). Its CA log-likelihood (−8.519306179549595), posterior and combined moments therefore depend on an unspecified reconstruction policy (F1).
- This refines the research brief, which labelled its Cases 1–3 "spec-backed". Three things are specified:
  - the policy-independent phases of those cases;
  - the CA mode's contribution to mixing, which the spec fixes as the marginal of its common components;
  - an equal-dimension bank, which involves no projection.
- **Small predicted probabilities.** The spec states the mixing and posterior formulas for c_j > 1e-12, a bound both native thresholds (1e-30 and 1e-300) satisfy. Treatment at or below 1e-12 is listed as unspecified, with one exception: the zero-probability case, when the zero-probability mode is no more likely than the best of the others. Native meets that exception because its 1e-300 clamp leaves such a mode at most 1e-300/1e-12 of the dominant weight. The general treatment is not claimed: the review probe recorded in Context gives [0, 1] where the exact posterior is [1, 0].
- **Elapsed time.** The spec fixes Π for a 1 s cycle as the configured matrix and leaves other intervals unspecified (F7). An identity Π stays the identity under any elapsed-time policy, so Π = I runs at any dt are specified behaviour. Non-identity Π runs at dt ≠ 1 are characterisation.

**Specified-behaviour tests (`imm_reference`):**
1. **Hand-derived exact rationals.** The oracle reproduces each of these values:
   - asymmetric, identity and symmetric mixing (c, μ_i|j, x0_j, P0_j, spread);
   - the worked CV prediction, innovation [15/31, 39/31, 23/62], det S = 80701246125/59105344, and ℓ_CV = −6.461143273565262;
   - the exact zero-destination combined mean [1689797/152447, 1612044/152447, 207755578/37959303, 63716985/12653101, 2541809541/25306202, 494019/12653101].
2. **Posterior from given log-likelihoods.**
   - The worked case gives [0.9274195958363679, 0.07258040416363172].
   - The underflow case, ℓ = [−163428.31075105123, −44784.431767829774] (z_x = 2011), gives exactly [0, 1].
   - The worked combination gives x = 10.84142489054062 and P[x,x] = 2.7725529136406477.
   - The elapsed-time grid: from one fixed mixed estimate, each CV mode's log-likelihood at dt 0.1, 1 and 2.3 matches a closed-form value derived in the test from the DWNA matrices.
   - **Negative control:** perturbing one committed constant by more than the tolerance (for example ℓ_CV by 1e-8; a last-digit change stays inside atol 1e-10 + rtol 1e-9) must fail an assertion. The perturbation is not retained.
3. **Native conformance** (only configurations whose outputs depend on no listed unspecified policy):
   - **`interaction_step()` for all three Π:**
     - on a CV+CV bank (identity mappings), both leaves equal (x0_j, P0_j);
     - on a CV+CA bank, the CV leaf equals (x0_1, P0_1). The CA leaf after back-projection depends on the unspecified reconstruction (F1), so it is characterisation.
   - **A CV+CV bank (σ_a = 1 and 3) run end-to-end with floor 0:**
     - Π ∈ {I, symmetric, asymmetric} at dt = 1;
     - Π = I at dt 0.1 and 2.3;
     - an underflow case.
     - μ⁺, the leaf posteriors and the combined moments must match the oracle. With Π = I, the posterior ratio also checks each mode's elapsed-time log-likelihood difference natively.
   - **The CV+CA zero-destination case**, Π = [[1,0],[1,0]] with floor 0: μ⁺ = [1, 0] and the exact combined mean, within tolerance. This meets the spec's zero-probability clause:
     - Under the current zero-destination assignment (mixed state and covariance zero; F4), native gives ℓ_2 = −184.35509375485347, below ℓ_1 = −6.859426052271374.
     - The 1e-300 clamp puts mode 2's log-weight about 868 nats below mode 1's, which underflows to exactly 0.
     - The test claims nothing about larger ℓ_2.
   - **Combination with supplied probabilities:** overwrite the bank's public mode probabilities with a given distribution, call `combine()`, and compare with the oracle's moment match. The learned classifier uses exactly this path (`imm_adapter.rs:245-247`), so this verifies the spec's learned-classifier clause without ONNX.
   - **The invariants**, with every c_j > 1e-12: mixing columns sum to one within 1e-12, Σ_j c_j x0_j = Σ_i (Σ_j π_ij) μ_i x_i, spread terms are PSD, and posteriors sum to one within 1e-12.
   - **Probability guard:** in every end-to-end run above, each predicted probability exceeds 1e-12 at every asserted cycle, except in the declared zero-destination case. Runs are limited to a stated number of cycles, and the test asserts the bound, so that no cited test depends on the unspecified treatment at or below 1e-12.
4. **Validation tests** (Decision 8).

**Complexity.** The oracle and both test targets follow the phase-helper rule (cognitive complexity ≤ 15):

`predict_modes → mix_common → likelihood → posterior → combine`

Policy emulation is a separate phase layer that the specified path never calls.

**Characterisation tests (`imm_current_policy`).**
- **Characterisation setup.** These inputs are needed to re-derive the anchors below:
  - The worked CV+CA bank is `ImmConfig::cv_ca(1, 1)`, that is σ_a = 1 (CV) and σ_j = 1 (CA), with dt = 1 s, R = 4I and z = [11, 6, 100.5].
  - **Coast:** two dt = 1 s cycles from the worked leaves, with the asymmetric Π, μ = [0.6, 0.4] and floor 0. The first cycle predicts without an update; z is applied only in the second.
  - **F4 far-measurement probe:** `cv_ca(1, 1)`, both leaves at [10000, 0, 0, 0, 1000, 0] with P = I, Π = [[1,0],[1,0]], μ = [0.6, 0.4], floor 0, one 1 s prediction, and z at the origin with R = 4I. Mode 2's zeroed mixed estimate lies at the origin, next to the measurement, so its log-likelihood exceeds mode 1's by more than about 690.8 nats and it captures the posterior.
  - **F4 small-probability mixing probe:** `cv_ca(1, 1)` with Π = I, both leaves at [50, 1, 20, 2, 300, 0] with P = I, and the mode probabilities overwritten with [1, 1e-40] before `interaction_step()`.
- Labelling rules:
  - The module docs open with "CURRENT-POLICY CHARACTERISATION — not specified behaviour; changes deliberately in `correct-native-imm-mode-policies`".
  - Every test name starts with `current_policy_`.
  - Each test cites its defect ID and records the exact or textbook value next to the pinned value.
- Cases pinned:
  - **Worked CV+CA cycle under the current CA expansion (F1):** ℓ_CA −8.519306179549595; μ⁺ [0.9274195958363679, 0.07258040416363172]; combined mean [10.84142489054062, 10.234515136297619, 5.5293027990598915, 5.285603042308766, 100.36744617154977, 0.03519257163692508] (agreement within 1.4e-14 between re-derivations); P[x,x] 2.7725529136; P[x,vx] 1.1895654128. Also the identity and symmetric posteriors, [0.9421643833, 0.0578356167] and [0.9334538576, 0.0665461424].
  - **CA reconstruction (F1):** acceleration mean 0, variance 100 (m/s²)², zero cross-covariance, selector [0,1,3,4,6,7]. After `interaction_step()` on the CV+CA bank, the CA leaf's common marginal equals (x0_2, P0_2) only because this reconstruction round-trips the common components.
  - **Default floor 0.01 at a zero-probability destination (F3, F4):**
    - μ = [0.9900990099, 0.0099009901];
    - vz 1.7555881216 against the exact 0.0390433144;
    - var(vz) 297.3863109839 against 2.6226006573.
  - **Small-probability thresholds (F4), floor 0:**
    - With Π = I and μ = [1, 1e-40], `interaction_step()` leaves mode 2's mixed state and covariance at zero, where the formula gives that mode's own estimate.
    - With Π = [[1,0],[1,0]], both leaves far from a measurement at the origin and mode 2's zeroed mixed estimate at the origin, native gives μ⁺ = [0, 1] where the exact posterior is [1, 0].
  - **Coast (F5):** one prediction without an update, then an update, gives μ⁺ [0.8900140198, 0.1099859802]. Propagating μ ← Πᵀμ would give [0.8963774786, 0.1036225214].
  - **Predicted output (F6):** the μ-weighted predicted x is 11.65704584040747. The c-weighted output would be exactly [58/5, 54/5, 21/5, 23/5, 502/5, 0].
  - **Transition matrix and dt (F7):** c is identical at dt 0.1, 1 and 2.3. The CV+CV end-to-end runs with the symmetric and asymmetric Π at dt 0.1 and 2.3 match the oracle with Π held at its configured value. The research brief listed this as spec-backed; it is characterisation here because the time dependence of the transition matrix is unspecified except for a 1 s cycle.
- Not pinned:
  - F2 (altitude), which the preregistered criterion covers;
  - F8 (non-positive-definite S), which is listed as unspecified.

**Existing tests.**
- No existing test that exercises IMM is edited:
  - the `#[cfg(test)] mod tests` in `imm.rs` and in `imm_adapter.rs`;
  - `crates/thresh-filter/tests/learned_imm.rs`;
  - the `#[cfg(test)] mod tests` in `crates/thresh-tracker/src/tracker.rs` (from `:1137`, including the IMM, learned-IMM and MHT tests);
  - the tests in `crates/thresh/src/eval_harness.rs` and `crates/thresh/src/training/mod.rs`, and `crates/thresh/tests/eval_tracker_cli.rs`, which run IMM-preset trackers.
- That adding new files satisfies "existing IMM test suite without modification" is an inference, recorded as such.

### 8. Transition-matrix and mode-probability validation (V)

**The change.**
- Change only `ImmConfig::validate`, plus doc-only `# Panics` sections on the infallible constructors.
- The new checks run after the existing dimension checks (`imm.rs:487-509`) and before the row-sum check (`:511`), so they never index a matrix of the wrong shape. They reject any transition-matrix entry, or any initial mode probability, that is non-finite or negative.
- Each error message names the offending row and column, or the offending index.
- The dimension checks and the 1e-6 sum tolerance are unchanged.

**Construction paths.**
- **Fallible:** `new_imm_position_learned` already returns `Err` from `validate` before loading any model (`tracker.rs:272-274`). The new rejections therefore surface as errors there, not panics. Before this change, a NaN or negative matrix passed validation, and any failure came later, through NaN propagation or the `partial_cmp().unwrap()` in dominant-mode selection. A new test file (`crates/thresh-tracker/tests/imm_config_validation.rs`) holds one `learned-imm`-gated test that asserts the error. That test runs in the existing CPU-only ONNX job.
- **Infallible:** `ImmFilter::new`, `ImmFilter::with_leaf_kind` and `MultiObjectTracker::new_imm_position` keep their refusal through `expect`. V widens the set of inputs that panic, so each gains a `# Panics` rustdoc section; today there is none (Context). New `#[should_panic]` tests pin that they never build a bank from a rejected configuration. The `new_imm_position` test is ungated, in the same new tracker test file.

**Alternative considered:** a new fallible `ImmFilter::try_new`. Rejected for this change: it is an API addition beyond V, and `imm-pluggable-leaf-filter` keeps the constructors unchanged. Change 3 may add one.

**Compatibility.**
- The two-mode and four-mode presets are unaffected (verified).
- This is a deliberate behaviour change for invalid input only. Every configuration valid today behaves identically, and the existing suite is unmodified.
- `imm-pluggable-leaf-filter` `:24-32` remains satisfied. V leaves the `ImmConfig` struct, its `cv_ca`/`cv_ctrv`/`cv_ca_ctrv_ct` constructors and the preset results unchanged. Only `validate()` now rejects inputs that were previously accepted but invalid.
- It is consistent with `learned-imm-tracker-integration` "invalid ImmConfig … returns an error".
- The spec states the timeless property ("every library preset passes validation"). The "filtering results unchanged by V" check is a verification step in the tasks, not spec text, because Change 3 deliberately changes preset results.

### 9. Exact-name protection for native reference targets

**The mechanism.**
- A checked-in plain-text manifest, `scripts/native-reference-manifest.txt`, lists one `<package> <target> <test>` entry per line.
- A dependency-free checker, `scripts/check-native-reference-tests.sh` (bash and cargo only; no Python, no jq), does the following for every manifest target. It must run under bash 3.2, the maintainer's macOS shell, and use only utilities portable between BSD and GNU: no associative arrays, `mapfile`/`readarray` or `${x,,}`, and no GNU-only flags such as `grep -P`, suffixless `sed -i` or `sort -V`.
  1. Rejects an empty, duplicated or malformed manifest.
  2. Compares the exact output of `cargo test --locked -p <package> --no-default-features --test <target> -- --list --format terse` with the manifest.
  3. Runs the target and requires each listed test to report `ok`, plus exactly one summary of `N passed; 0 failed; 0 ignored; 0 measured; 0 filtered out`, where N is the manifest count. A `#[should_panic]` test reports as `test <name> - should panic ... ok`, so the checker accepts that suffix.
- A `--self-test` mode feeds canned libtest output. It must fail on each of: missing, unlisted, ignored, failed, filtered and zero-test output, and on an empty, duplicated or malformed manifest. It must pass on a matching inventory that includes a should-panic line.
- **Protected targets:**
  - `thresh-association/mht_reference`;
  - `thresh-filter/imm_reference`;
  - `thresh-filter/imm_current_policy`, whose renaming in Change 3 is then visible in the manifest diff.
- The `native-no-python` job runs the self-tests and then the check, after its existing workspace tests.

**Alternatives considered:**
- **Extend `run_boundary_tests.py`.** Rejected: it is bound to CPython and Stone Soup, and the native lane must not need Python.
- **A `harness = false` target.** Rejected: it loses libtest listing and filtering, and diverges from the manifest convention.
- **Plain `cargo test --workspace`.** Rejected: the tree already has 25 `#[ignore]` attributes, and an ignored or renamed test passes silently.
- **A Rust `xtask` crate.** Rejected because it adds a crate.

### 10. Interim disclosure

**Rustdoc warning on `AssociationStrategy::Mht`** (`crates/thresh-tracker/src/tracker.rs:82-92`).
- The warning states that the native path is not yet a working MHT, for three reasons:
  - expansion ignores per-hypothesis history;
  - N-scan pruning is not invoked;
  - with default heads, the path does not associate.
- It names `rebuild-native-mht-hypothesis-history`.
- There is no behaviour change.

**CHANGELOG `[Unreleased]`.**
- **Removed (BREAKING):** the bridge modules.
- **Changed:** IMM validation.
- **Corrections:**
  - `:28`: N-scan pruning is not wired into the tracker.
  - `:30`: `mht_dense_clutter_maintains_track` is not MHT robustness evidence.
  - `:168`: the 0.1.0 bridge MHT/IMM claim never worked on the pinned runtime.
- The unreleased bullets at `:28` and `:30` are amended in place to point at the correction. The released `[0.1.0]` text stays as history.
- The correction paraphrases the `:168` claim rather than quoting it, so the repository search in task 2.4 still matches only the released line.

**Repository docs.**
- `CLAUDE.md:41` and `AGENTS.md:41` describe `thresh-bridge` as JPDA and GM-PHD references.
- `docs/bridge-runtime.md:43-46` is tightened: after the deletion, the only standalone bridge entry point is `measurement_to_detection`.
- **Analysis documents.** `docs/analysis/stonesoup-reference-review.md` is the historical review and stays unedited. `docs/analysis/advanced-methods-integration.md` claims at `:235` that MHT ships "plus Stone Soup bridges", and at `:26` that native MHT ships today. It gets a dated correction note at `:235` rather than a rewrite: the bridge MHT and IMM never worked on the pinned runtime and are removed, and native MHT is not yet a working MHT.

## Preregistered IMM maneuvering criterion

The criterion is preregistered now. It becomes an ADDED `imm-estimation` requirement in `correct-native-imm-mode-policies`. It carries forward the archived `imm-estimation` bar of 5 update cycles.

**Fixed values.**
- These values are fixed before any native run.
- Changing one requires a recorded reason and maintainer approval in that change's design.
- Tuning to native results is prohibited.

**Configuration.**
- **Banks:**
  - B1 = `ImmConfig::cv_ctrv(1, 5, 0.5)`, whose turn mode is CTRV;
  - B2 = `ImmConfig::cv_ca_ctrv_ct(5, 1, 2, 0.1)`, whose turn modes are CTRV and CT.
  - Turn-mode probability means the sum over the turn modes. Each bank uses its preset transition matrix and initial μ.
- **Truth:**
  - constant altitude h ∈ {0, 1000, 10000} m;
  - 100 m/s along +x;
  - straight, level flight for t ∈ (0, 20] s;
  - from t = 20 s, an exact constant-rate horizontal turn at ω ∈ {0.05, 0.15} rad/s until t = 40 s.
- **Measurements:** 3-D Cartesian position at t = 1, …, 40 s (dt = 1 s), with R = 10² I m².
- **Initialisation:** at t = 0, the true state and P0 = diag(100, 25, 100, 25, 100, 25) on [x, vx, y, vy, z, vz]. The floor is the library default in force when Change 3 lands.
- **Runs:** 12 configurations (2 banks × 3 altitudes × 2 rates). Each has one noise-free run and ten seeded runs, seeds 0–9, with Gaussian noise from a deterministic seeded RNG.

**Pass conditions** (every configuration must pass):

| ID | Condition | Applies to |
|---|---|---|
| P1 | No false turn: for scans 6–20, turn-mode probability < 0.5, and CV has the largest single-mode probability. | The noise-free run and at least 9 of 10 seeded runs. |
| P2 | Detection: turn-mode probability > 0.5 at some scan 21–25, within 5 update cycles of onset. | The noise-free run and at least 9 of 10 seeded runs. |
| P3 | Consistency: at least 90% of scans 6–40 have NIS ≤ χ²₀.₉₅(3) = 7.8147, where NIS uses the combined predicted measurement and the innovation covariance used for gating. | Each seeded run. |
| P4 | Accuracy: 3-D position RMSE over scans 6–40 ≤ 17.32 m (10√3 m, the RMS raw-measurement error). The noise-free run additionally requires \|combined z − true z\| ≤ 1 m at every scan 6–40. | Each seeded run, plus the noise-free run for the altitude bound. |

**Current native status** (evidence only, not calibration):
- **Altitude:** at 1 km and 10 km, both banks fail P2 and P4. Turn-mode μ stays at the floor (0.0099 for B1, 0.0196 for B2), and combined z is biased by −99 m and −196 m at 10 km.
- **Planar:** at 0 m, B2 fails P1: its turn mass is 0.86–0.92 before onset (0.921 and 0.899 in the retained probe output). B1 at ω 0.05 fails P2: in the retained noise-free probe the maximum turn-mode μ is about 0.12, and another probe, whose output was not retained, reported 0.131. One seeded run crossed 0.5 briefly and decayed to 0.04 by scan 20.
- **Only near-pass:** B1 at 0 m with ω 0.15 would meet P2.
- The probes used other R values and onsets, so they demonstrate failure rather than set the thresholds. The REMOVED Migration states this gap in the same terms.

## Evidence summary and reproduction

The probe scripts were ephemeral research artefacts and are not committed. Numbers that become tests are reproduced by the committed Rust targets (Decisions 6–7). The procedures for the remaining numbers are described here.

**Bridge wrappers.** Probed under CPython 3.12.13 and Stone Soup 1.9.1:
- **IMM:** importing IMM gives `ModuleNotFoundError(name='stonesoup.predictor.interacting')`. The predictor package's submodules are `_utils, asd, base, categorical, composite, ensemble, information, kalman, kernel, particle, pointmass`.
- **MHT tracker:** `MultiTargetMixtureTracker` takes the parameters `[initiator, deleter, detector, data_associator, updater]`. Construction with the wrapper's keywords raises a `TypeError` for the missing `initiator`. The class has `update_tracker` and no `track`.
- **MFA:** importing MFA requires `ortools`.
- **CI:** CI job 137895907234 reported "Boundary gate PASSED: 109 required tests across 8 targets; zero ignored."
- **Reproduce:** import those symbols in the pinned virtual environment.

**MFA** (approved deviation D1).
- **Setup:** an isolated venv with `ortools==9.14.6206`.
- **Associator:** `MFAHypothesiser` around `PDAHypothesiser(prob_gate=1, include_all=True)`, and `MFADataAssociator(slide_window=W)` with W = 1–4 and `normalise` ∈ {False, True}.
- **Scenarios:** 300 random 2-target, 4-scan 2-D CV scenarios with P_D 0.9, clutter 0.01, R 0.25·I and diffusion 0.05.
- **Comparison:** each step was compared with an exhaustive enumeration of the same window problem.

Results:
- **Suboptimal window solutions** (of 1200 steps each):

  | `normalise` | W = 2 | W = 3 | W = 4 |
  |---|---|---|---|
  | False | 56 | 83 | 85 |
  | True | 42 | 85 | 121 |

  That range is 3.5–10.1% of steps.
- **Maximum excess:** 7.8405565993965425 nats (seed 30, `normalise=False`, W = 4, scan 4), which exited on "no primal improvement".
- **Effect of `normalise`:** it changes the full-horizon optimum in 156 of 300 scenarios.
- **Effect of κ = 1e-6:** suboptimal steps fell from 27 to 17 (W = 2) and from 42 to 32 (W = 3) on a 150-scenario subset.
- **Solver defects in the source:** it stops on no primal improvement (`_step.py:241-248`), and its gap test has the wrong sign for negative costs (`:251-252`).
- **Crossing fixture:** it matched only by a margin of 0.0035 nats against a 2% gap of about 0.066 nats. Only the four W = 1 steps exit certified; all 12 steps with W ≥ 2 exit via `gap<0.02`.

**Native MHT** (Change 2).
- **History-blind expansion:** the best leaf equals the best parent plus the per-scan argmax. No pruning rule can change the MAP current decision; this held in 6000 of 6000 randomized trials.
- **No association:** one clean target over 20 steps receives 20 IDs under MHT and 1 under Hungarian. The cause is ln(P_D·N) ≈ −16.68, below the gate ln(1e-6) = −13.82.
- **Tracker scoring against the reference:** the reference's best hypothesis ranks 26/27 (C1), 50/51 (C2), 13/13 (C3) and 9/9 (C5). C4 cannot be represented.
- **Normalised scoring:** still wrong, at rank 2 in C1 (−4.3663) and C2 (−1.3984).
- **Reproduce:** Change 2's conformance tests run native code against the committed reference.

**Native IMM agreement.**
- **Against an independent NumPy reference:** native agrees within ≤ 7.1e-14 over 10 cases × 17 checks, with a tolerance of 1e-10 + 1e-9·|ref|. The cases were worked, worked_floor, identity, symmetric, zero_dest_floor0, zero_dest_default, underflow, coast2, dt_0p1 and dt_2p3.
- **Exact rationals:** two independent exact-rational re-derivations, plus a third on 2026-11-25, reproduce the worked case.
- **FilterPy 1.4.5 (equal dimensions):** a scratch numeric check gives |Δμ| 3.3e-16 and |Δx| 2.8e-14. The `IMM.py` line citations in Decision 2 refer to the FilterPy 2.0.0 source.
- **Reproduce:** `imm_reference` and `imm_current_policy`.

**F2 probes** (Change 3).
- **Cause:** CTRV and CT write z = vz = 0, with variance 100, into common space (`imm.rs:129-158`, `:215-230`), and the likelihood is computed there (`:945-963`).
- **Probe setup:** truth at 100 m/s, straight and then turning, with 3-D position measurements.

| Bank | Altitude | Turn-mode μ | Other observations |
|---|---|---|---|
| B1 | 1 km, 10 km | stays at 0.0099 | At 10 km, one probe reported approximately mean NIS 84.9, 39 of 40 scans above χ²₀.₉₅(3), and RMSE 34.3 m. Its output was not retained, so these three figures cannot be cross-checked. |
| B2 | 1 km, 10 km | stays at 0.0196 (exactly 0 with floor 0) | 4-mode z variance 1.92e6 m². |

The fail conclusions rest on the retained outputs: the 0.0099 and 0.0196 floors, the 1.92e6 m² z variance, and the −99 m and −196 m biases.

- **Combined z bias:** −99 m (B1) and −196 m (B2) at 10 km; −10 m and −20 m at 1 km.
- **Why the existing tests pass:** the existing maneuver tests feed z = 0 (`imm.rs:1338`, `:1372`, `:1431`, `:1453`).
- **Reproduce:** run the preregistered criterion in Change 3.

## Risks / Trade-offs

- **[No working MHT between Change 1 and Change 2]** → The REMOVED Migration states the gap. The rustdoc warning and the CHANGELOG correction make it visible. Change 2 depends only on this change's archive.
- **[Native IMM fails the maneuvering scenario until Change 3: at altitude, and in every probed planar configuration except B1 at ω 0.15]** → The Migration states the gap in those terms. The criterion is preregistered. Change 3 follows immediately and fixes F2 first.
- **[The `imm-estimation` spec claims more than native does]** → Each claim was checked against `imm.rs`. The formulas are scoped to predicted probabilities above 1e-12 and to floor zero. The zero-probability clause is restricted to the case native meets. The thresholds and the far-likelihood behaviour are listed as unspecified and pinned as F4 characterisation.
- **[Characterisation tests mistaken for contract]** → Mitigated by a separate target, the `current_policy_` prefix and module docs. The spec's "Explicitly unspecified mode policies" forbids citing them as evidence.
- **[The ADDED `imm-estimation` codifies current behaviour, although review `:406` permits IMM spec edits "only if behavior is deliberately changed"]** → Disclosed (D3). The ADD carries only independently verified mathematics, plus one deliberate behaviour change (V). It is the Migration target that keeps the bridge IMM obligation from being deleted.
- **[V rejects previously accepted inputs]** → Only invalid input is affected. The presets and the existing suite are unchanged. The learned path returns an error, and the infallible paths keep their existing refusal.
- **[A re-homed vehicle is weaker than the original]** → Each test keeps its name, its classification assertion and its import-detail assertions. The count changes deliberately, and the runner self-tests still cover missing, ignored and zero tests.
- **[Floating-point summation order varies across enumeration orders]** → Mitigated by the 1e-10/1e-9 tolerances and tie grouping at 1e-9. The minimum separation between groups is 0.0046.
- **[The rename becomes less visible after archive]** → The ADDED requirement moves to the end of the spec, and its Reason and Migration live only in the archived delta and `proposal.md`. The Purpose is hand-edited in the archive PR (task 7.1).
- **[Upstream later ships IMM or a sliding-window MHT]** → A new change can reintroduce the claim. Observation belongs to the review's newer-version lane (§3.3 item 5), because the hash-locked lane cannot observe upstream.
- **[An oracle shares its author's misconceptions]** → Mitigated by exact enumeration, closed-form hand anchors, two enumeration orders, multiple recorded independent re-derivations, the source-independence check, and a perturbation negative control on each reference (MHT and IMM).

## Migration Plan

**Implementation order.** Each step keeps the tree green:
1. **Re-home** the four retained `runtime_contract` vehicles and add the lib pinned-absence regression, while the wrappers still exist.
2. **Delete** the bridge modules, their dispatch tests and the token-free test. Update the manifest, `docs/bridge-testing.md` counts and the runner's status text, and confirm the exact executed count.
3. **Update** the remaining docs, gap strings, the CHANGELOG, `CLAUDE.md` and `AGENTS.md`.
4. **Add** the MHT reference target and the `AssociationStrategy::Mht` warning.
5. **Add** the IMM oracle, the specified-behaviour tests, the V fix with its tests, and the characterisation tests.
6. **Add** the native exact-name checker to the `native-no-python` job.
7. **Run** integrated verification and record versions, counts and maximum residuals in `docs/bridge-testing.md` and `docs/native-reference-tests.md`.
8. **Archive** (a separate archive/sync PR): hand-edit the `stonesoup-bridge` Purpose, run `openspec archive`, then validate the live and archived specs (task 7.1).

**External callers.** There is no compatibility shim.
- `thresh_bridge::imm` → `thresh_filter::imm::ImmFilter`, which takes native Rust state, not Stone Soup objects.
- `thresh_bridge::mht` → no working replacement until Change 2. The native MHT strategy exists but is documented as unfit for use.

**Rollback.**
- Revert the implementation PR as a unit.
- A rollback must not reintroduce a claim that the bridge MHT or IMM works.
- Restoring the removed requirement also requires restating its unmet status.

## Follow-up changes

**Change 3: `correct-native-imm-mode-policies` (M/L; immediately after this change).**
- Fix F2 first, then F1 and F3–F8.
- ADD the maneuvering requirement to `imm-estimation`, using the preregistered criterion.
- MODIFY "Explicitly unspecified mode policies" as each policy becomes specified.
- MODIFY the `imm-estimation` requirements whose wording a chosen fix touches:
  - "Analytic mode-probability posterior": its zero-probability clause and the 1e-12 bound, when the exclusion policy is specified;
  - "Common-state mixing": the 1e-12 bound and the 1 s transition-matrix anchor, if rate semantics or mixing beyond the common state (carrying mode-specific components) change them;
  - "Model-conditioned log-likelihood": if F2 is fixed by evaluating each mode's likelihood in its own state.
  These requirements are already worded to admit the recommended defaults, so a MODIFIED is needed only when a fix departs from that wording.
- MODIFY `imm-pluggable-leaf-filter` `:26-32`.
- Update `imm_current_policy` deliberately.
- Re-baseline learned IMM and evaluation.
- Coordinate with the two active learned-IMM changes: archive them first, or MODIFY their pending text.
- Recommended defaults:
  - floor 0, with zero-destination exclusion;
  - μ ← Πᵀμ on coast;
  - c-weighted predicted output;
  - preserve z and vz through CTRV and CT;
  - Π declared per scan, with rate semantics deferred to P2-B;
  - carry mode-specific components.

**Change 2: `rebuild-native-mht-hypothesis-history` (XL; split into 2a and 2b; depends on this change being archived).**
- **Spec MODIFIEDs to `mht-tracking`:**
  - a hand-edited Purpose;
  - "Hypothesis tree maintenance", naming the score terms (the Decision 6 model is the proposed input);
  - "N-scan pruning" `:21`, `:27` and `:29`, for the chosen commit rule (R1 recommended, with its departures from R2 and R3 disclosed);
  - "Best-hypothesis track extraction" `:53`, the ID policy;
  - "K-best pruning": the text is unchanged, but reclamation is implemented.
  - "Exhaustive reference verification" (added by this change): MODIFIED only if Change 2 needs to change its fixture bound or its assertion categories. As worded, it admits rule-specific, tie-break and conformance assertions once requirements of `mht-tracking` define them. Change 2's planned fixtures (243 and 15,379 leaves) also fit its 100,000-hypothesis bound.
- **`track-management` MODIFIED, scoped to MHT:** `:22-27` (births), `:52-57` (IDs), and M-of-N counting from best-hypothesis ancestry.
- **BREAKING:** the `HypothesisTree` API, and `AssociationStrategy::Mht` gains P_D, λ_FA and λ_B.
- **2a (generic tree):**
  - a tree generic over state, likelihood and birth initialiser, with an arena and orphan reclamation;
  - NaN rejection and a canonical tie-break;
  - rule-specific oracle assertions: the retained set and committed prefix after every scan at N = 1 and N = 2, plus a 5-scan N = 3 case;
  - conformance on C1–C5 and on the 2-D crossing fixture with λ_B = 0 (15,379 leaves).
- **2b (tracker integration):**
  - tracker integration, covering IMM tracks or an initial restriction to non-IMM tracks;
  - a clean-target test: 20 steps, 1 ID and 20 ancestry hits;
  - a rework of `mht_dense_clutter_maintains_track`.

**Backlog (outside P0-C):**
- **`add-stonesoup-mfa-crosscheck`:** only if OR-Tools is approved. It would be a deterministic regression against enumerator-verified fixtures, with tzdata pinned unconditionally and `__cause__`-walking error classification.
- **A newer-version watch lane:** it would probe the #1257 modules.

## Approved deviations and disclosures

**Explicitly approved deviations (maintainer, 2026-11-23):**
- **D1: MFA/OR-Tools evaluated and declined.**
  - This deviates from review §3.1 `:291`, "MFA's approximate solver … must each be checked against that oracle", a P1-A item. It also deviates from §3.3 `:375-376`, which provisions `ortools==9.14.6206` for an explicitly selected MFA lane.
  - The MFA evidence above is P0-C's "independently evaluate MFA history" deliverable.
  - Native flat history is checked against the oracle in Change 2.
- **D2: I-d instead of review §2.3's primary IMM recommendation** (`:194-196`, an independent NumPy reference using supported single-model predictors and updaters). This uses that sentence's "or another explicitly evaluated reference" clause and the §5 amendment route (`:406`).

**Disclosures that follow from the binding decisions:**
- **D3:** the ADDED `imm-estimation` codifies current, policy-independent mathematics as the Migration target, although review `:406` permits IMM spec edits only for a deliberate behaviour change. V is the deliberate behaviour change it contains. It supersedes the archived, never-synced `imm-estimation` delta.
- **D4:** P0-B `design.md:110` is superseded (Decision 4).
- **D5:** the bridge maneuvering scenario is preregistered and deferred with an explicit gap instead of being re-homed as a passing requirement. Adding it now would fail at altitude and in most planar configurations, or would have to be narrowed to the one planar configuration that passes.
