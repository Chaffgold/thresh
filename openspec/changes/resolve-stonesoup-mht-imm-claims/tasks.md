# Tasks

## 1. Re-home bridge diagnostics off the MHT/IMM wrappers

- [x] 1.1 Re-vehicle `missing_numpy_is_a_dependency_not_an_interpreter_error` onto `measurement_to_detection`, using a valid synthetic context so that the NumPy import at `crates/thresh-bridge/src/detection.rs:55` is the first blocked import.
  - **Verify:** the error is `MissingDependency { name: "numpy" }`, with the synthetic `ModuleNotFoundError` type and message, operation `import numpy`, and a `find_spec` traceback.
- [x] 1.2 Re-vehicle `missing_stonesoup_submodule_is_an_api_error` onto the JPDA constructor's first import, `stonesoup.hypothesiser.probability` (`src/jpda.rs:83`), with placeholder components.
  - **Verify:** the error is `UnsupportedApi` naming that module, and the same import-detail assertions hold.
- [x] 1.3 Re-vehicle `missing_symbol_is_an_api_error` onto the `JPDA` lookup (`src/jpda.rs:116`): serve an empty `stonesoup.dataassociator.probability` module and supply real predictor and updater components.
  - **Verify:**
    - the error is `UnsupportedApi { api: "JPDA" }`;
    - the cause is an `AttributeError` whose message names `JPDA`;
    - the operation is "JPDA associator lookup".
- [x] 1.4 Reduce `explicit_host_initialization_is_reused_without_mutation` to its JPDA vehicle.
  - **Verify:** three repeated calls each return `StoneSoupNotInstalled`, and the `sys.path`, environment and `sys` module snapshot is unchanged.
- [x] 1.5 Add the library unit test `runtime::tests::pinned_stonesoup_lacks_interacting_predictor_api` and register it in `tests/boundary-test-manifest.json`.
  - **Behaviour:** under an initialized host, it calls the crate-private `import_module` (`src/runtime.rs:23`) on `stonesoup.predictor.interacting`.
  - **Documentation:** a pinned-absence classification regression for Stone Soup 1.9.1, with no claim to be an upstream canary.
  - **Verify:**
    - it asserts `UnsupportedApi { api: "stonesoup.predictor.interacting" }`;
    - the runner self-tests and the full boundary runner pass under the hash-locked CPython 3.12.13 environment while the wrappers still exist.

## 2. Remove the bridge MHT and IMM wrappers (BREAKING)

- [x] 2.1 Delete `crates/thresh-bridge/src/{mht,imm}.rs`, their `lib.rs` declarations and their dispatch-double unit tests. Remove `initialized_imm_constructor_reports_unresolved_api`, `token_free_constructors_require_host_initialization` and the now-unused imports from `tests/runtime_contract.rs`.
  - **Verify builds:**
    - `cargo build -p thresh-bridge` succeeds with and without `--features stonesoup`;
    - `cargo build -p thresh --features stonesoup` succeeds;
    - `cargo test -p thresh-bridge --features stonesoup --no-run` compiles every edited test target.
  - **Verify references:** a repository search for `thresh_bridge::mht`, `thresh_bridge::imm`, `bridge::mht`, `bridge::imm`, `MhtTracker` and `MhtConfig` finds nothing. The search excludes `openspec/changes/archive/`, this change's directory `openspec/changes/resolve-stonesoup-mht-imm-claims/`, `docs/analysis/`, and the CHANGELOG removal entry added in 2.4.
- [x] 2.2 Remove the four deleted names (two library, two `runtime_contract`) from `tests/boundary-test-manifest.json`. Then set the per-target table and total in `docs/bridge-testing.md` (`:66-77`) to the exact resulting counts. The expected counts are 37 library tests, 8 `runtime_contract` tests and 106 in total; record the actual numbers.
  - **Verify:**
    - the runner self-tests pass;
    - the full runner prints "Boundary gate PASSED: <exact count> required tests across 8 targets; zero ignored." under CPython 3.12.13.
- [x] 2.3 Rewrite the current-gap and status strings so they state two things:
  - the bridge provides JPDA and GM-PHD references only;
  - the MHT and IMM contracts are native (`mht-tracking`, `imm-estimation`), with interim gaps.

  Files to update:
  - `.github/workflows/ci.yml:171`;
  - `crates/thresh-bridge/tests/run_boundary_tests.py:249-250`;
  - `docs/bridge-testing.md:7`, `:90`, `docs/bridge-detections.md:94-95` and `docs/bridge-references.md:109`;
  - `docs/bridge-runtime.md:37`, and `:43-46` so it names `measurement_to_detection` as the only standalone entry point;
  - `crates/thresh-bridge/src/detection.rs:25`;
  - `crates/thresh-bridge/Cargo.toml:3` (the package description only);
  - `crates/thresh-bridge/src/lib.rs:1`;
  - `docs/analysis/advanced-methods-integration.md:235`: add the dated correction note from design.md Decision 10 without rewriting the analysis.
  - **Verify:**
    - a search for `MHT` and `IMM` in the bridge, CI and repository files above shows only the corrected wording. For `docs/analysis/advanced-methods-integration.md`, verify only that the dated correction note is present at `:235` and that the rest of the file is unchanged;
    - a prose search for "Stone Soup bridges", "Stone Soup MHT" and "Stone Soup IMM" finds no claim that a bridge MHT or IMM exists. It excludes `docs/analysis/stonesoup-reference-review.md`, `openspec/changes/` and the released CHANGELOG sections;
    - the job name "Stone Soup boundary" is unchanged;
    - `RUSTDOCFLAGS=-Dwarnings cargo doc -p thresh-bridge --no-deps --features stonesoup` and the feature-enabled bridge doctests pass.
- [x] 2.4 Update the repository docs and changelog for the bridge change:
  - `CLAUDE.md:41` and `AGENTS.md:41` describe `thresh-bridge` as Stone Soup JPDA/GM-PHD references.
  - Add the CHANGELOG `[Unreleased]` entries from design.md Decision 10:
    - the BREAKING removal of `thresh_bridge::{mht,imm}` and `thresh::bridge::{mht,imm}`, with its migration targets;
    - corrections of `:28`, `:30` and the released `:168` claim. The `:168` correction paraphrases the claim rather than quoting it.
  - Amend the unreleased `:28` and `:30` bullets in place.
  - **Verify:** released sections are unchanged. A search for `JPDA/MHT/IMM` or `MHT, and IMM` over `CHANGELOG.md`, `CLAUDE.md`, `AGENTS.md`, `crates/` and `docs/` (excluding `docs/analysis/`) matches only the released `[0.1.0]` line.

## 3. Exhaustive MHT reference

- [x] 3.1 Implement the std-only fixture model and per-track-recursion enumerator from design.md Decision 6 in `crates/thresh-association/tests/support/mht_oracle.rs`.
  - **Structure:** phase helpers (predict, enumerate track events, label remaining detections, score, extend history), each with a focused unit test and cognitive complexity ≤ 15.
  - **Verify:**
    - `cargo test -p thresh-association --test mht_reference` passes;
    - Clippy reports no warnings with `-D warnings`;
    - an `include_str!` test asserts that the oracle source names no `thresh_` crate.
- [x] 3.2 Add a second enumeration inside `mht_reference.rs`: a per-detection label product with a feasibility filter, decomposed into phase helpers (label product, feasibility filter, scoring, history key) with cognitive complexity ≤ 15.
  - **Verify:** for C1–C5 the two enumeration orders produce identical sets of histories, keyed by per-scan events, with scores equal within 1e-12. The leaf counts are exactly 27, 51, 13, 185 and 9, and a test asserts each is at most the spec's 100,000-hypothesis bound.
- [x] 3.3 Commit the C1–C5 fixtures and their full-precision constants from design.md Decision 6:
  - the best history and score;
  - the runner-up history and score, and the gap;
  - the C1 per-scan hand anchors.

  Add a provenance comment that names the independent enumerations and their agreement.
  - **Verify:** assertions pass at atol 1e-10 and rtol 1e-9, and a deliberate perturbation of one constant makes an assertion fail. Revert the perturbation.
- [x] 3.4 Add the contract-neutral structural assertions:
  - no detection is reused within a scan;
  - each existing track has exactly one event per scan;
  - each cumulative score equals the sum of its per-scan scores;
  - ranking is compared as tie groups at 1e-9: C2 has 50 groups, including the rank 10–11 tie at −4.74083102492305; C4 has 148 groups, 31 of them tied, the largest with 4 members;
  - no asserted prefix ends inside a group;
  - the MAP matches without pruning.
  - **Verify:** no assertion encodes an N-scan rule, retained set, ID policy, tie-break or native output.
- [x] 3.5 Add the rustdoc warning from design.md Decision 10 to `AssociationStrategy::Mht` (`crates/thresh-tracker/src/tracker.rs:82-92`). It names `rebuild-native-mht-hypothesis-history` and changes no behaviour.
  - **Verify:** `RUSTDOCFLAGS=-Dwarnings cargo doc -p thresh-tracker --no-deps` passes, and `cargo test -p thresh-tracker` results are unchanged.

## 4. IMM estimation verification and validation fix

- [x] 4.1 Implement the specified-mathematics oracle in `crates/thresh-filter/tests/support/imm_oracle.rs`, using only `nalgebra` and `std`, and the shared fixtures in `crates/thresh-filter/tests/support/imm_fixtures.rs`.
  - **Oracle contents:**
    - closed-form CV and CA models;
    - the CA common-state marginal;
    - mixing with spread;
    - Joseph update;
    - Cholesky log-likelihood;
    - log-sum-exp posterior with exact zero-probability handling;
    - moment-matched combination.
  - **Structure:** phase helpers with cognitive complexity ≤ 15. Neither support module contains a `#[test]`. Each test target declares `mod support;` with `#[allow(dead_code)]` (design.md Decision 7).
  - **Verify:**
    - an `include_str!` test asserts that the source names no `thresh_` crate;
    - the oracle reproduces every hand-derived value listed in design.md Decision 7, item 1;
    - the oracle's phase unit tests pass in `imm_reference.rs`.
- [x] 4.2 Add the current-policy emulation parameters to the oracle as a separate phase layer that the specified path never calls. The parameters are: CA acceleration reconstruction, the floor and its renormalisation, the zero-destination thresholds (1e-30 in mixing, 1e-300 in the mode update), coast propagation, and predicted-output weighting.
  - **Verify:**
    - the 4.1 reproduction still passes unchanged;
    - with the current settings, the oracle reproduces the characterisation values listed in design.md Decision 7.
- [x] 4.3 Add the exact-rational and given-likelihood tests in `crates/thresh-filter/tests/imm_reference.rs`:
  - asymmetric, identity and symmetric mixing (c, μ_i|j, x0_j, P0_j, spread);
  - the worked CV prediction, innovation, det S and ℓ;
  - the elapsed-time grid: per-mode log-likelihoods from one fixed mixed estimate at 0.1, 1 and 2.3 s;
  - the worked posterior;
  - log-space underflow giving [0, 1];
  - the exact zero-destination combined mean;
  - the worked combination.
  - **Verify:**
    - comparisons pass at atol 1e-10 and rtol 1e-9, and probabilities sum to one within 1e-12;
    - perturbing one committed IMM constant makes an assertion fail. Revert the perturbation.
- [x] 4.4 Add native conformance tests to the same target, limited to configurations that depend on no listed unspecified policy (design.md Decision 7, item 3):
  - `interaction_step()` for all three transition matrices: on a CV+CV bank both leaves equal the mixed moments; on a CV+CA bank the CV leaf does;
  - a CV+CV bank (σ_a 1 and 3) run end-to-end with floor 0, with Π ∈ {I, symmetric, asymmetric} at dt = 1, Π = I at dt 0.1 and 2.3, plus an underflow case;
  - the CV+CA zero-destination case with floor 0;
  - combination under supplied mode probabilities: overwrite the public mode probabilities and call `combine()`;
  - the mixing and posterior invariants, with every predicted probability above 1e-12.
  - **Structure:** phase helpers with cognitive complexity ≤ 15, sharing setup through the support fixtures.
  - **Verify:**
    - native outputs agree with the oracle at the stated tolerances;
    - every predicted probability exceeds 1e-12 at every asserted cycle, except in the declared zero-destination case, and each end-to-end run asserts that bound over its stated number of cycles;
    - no test in this target depends on CA reconstruction, a nonzero floor, coasting or non-identity Π at dt ≠ 1.
- [x] 4.5 Fix V in `ImmConfig::validate`. After the unchanged dimension checks (`imm.rs:487-509`) and before the row-sum check (`:511`), reject non-finite or negative transition-matrix entries and initial mode probabilities, naming the offending entry. Add `# Panics` rustdoc sections to `ImmFilter::new`, `ImmFilter::with_leaf_kind` and `MultiObjectTracker::new_imm_position`. These are doc-only, with no API change.
  - **Verify validation:** new `imm_reference.rs` tests reject the following, and every preset still validates:
    - NaN and ±∞ transition entries;
    - a [1.2, −0.2] row;
    - NaN, ∞ and [1.5, −0.5] initial probabilities.
  - **Verify construction paths:**
    - **Infallible constructors refuse:** `#[should_panic]` tests show that rejected configurations are refused by:
      - `ImmFilter::new`, tested in `imm_reference.rs`;
      - `MultiObjectTracker::new_imm_position`, tested ungated in the new `crates/thresh-tracker/tests/imm_config_validation.rs`.
    - **Fallible constructor returns an error:** one `learned-imm`-gated test in that new file shows that `MultiObjectTracker::new_imm_position_learned` returns an `Err` mentioning the invalid configuration, without panicking. Run it with `cargo test -p thresh-tracker --features learned-imm`.
  - **Verify presets:** preset filtering results are unchanged by V. The existing `imm::tests` and tracker IMM tests pass unmodified (4.7).
- [x] 4.6 Add `crates/thresh-filter/tests/imm_current_policy.rs`, using the module header and `current_policy_` test-name prefix from design.md Decision 7. It pins:
  - **F1:** the worked CV+CA cycle, including the identity and symmetric posteriors; the CA acceleration reconstruction; and the CA leaf's round-trip after `interaction_step()`;
  - **F3/F4:** the default floor at a zero-probability destination, and the small-probability thresholds at floor 0 (the Π = I, μ = [1, 1e-40] mixing case and the far-measurement [0, 1] case);
  - **F5:** coasting;
  - **F6:** predicted-output weighting;
  - **F7:** c across dt, and the CV+CV end-to-end runs with symmetric and asymmetric Π at dt 0.1 and 2.3.

  Each test records the exact or textbook value beside the pinned value. Tests follow the phase-helper rule and share setup through the support fixtures.
  - **Verify:** every test passes against the oracle configured with the declared current policy, and `imm_reference.rs` references none of them.
- [x] 4.7 Confirm the existing IMM tests are untouched.
  - **Verify:**
    - `git diff` against the base shows no change to any existing test that exercises IMM (design.md Decision 7, "Existing tests"): the `#[cfg(test)] mod tests` in `crates/thresh-filter/src/imm.rs`, `crates/thresh-filter/src/imm_adapter.rs` and `crates/thresh-tracker/src/tracker.rs`; `crates/thresh-filter/tests/learned_imm.rs`; the tests in `crates/thresh/src/eval_harness.rs` and `crates/thresh/src/training/mod.rs`; and `crates/thresh/tests/eval_tracker_cli.rs`;
    - the only hunks in `crates/thresh-tracker/src/tracker.rs` are rustdoc: the `AssociationStrategy::Mht` warning and the `new_imm_position` `# Panics` section;
    - `cargo test -p thresh-filter` passes;
    - `cargo test -p thresh-filter --features learned-imm` passes.
- [x] 4.8 Document the IMM scope and the validation change:
  - Rustdoc on `ImmConfig::validate` lists the rejected inputs, with a self-contained example that rejects a negative transition-matrix row.
  - The IMM section of `crates/thresh-filter/README.md` covers the specified mathematics, the list of unspecified policies with a pointer to `correct-native-imm-mode-policies`, and the stricter validation.
  - Add a CHANGELOG `[Unreleased]` entry for the IMM validation change.
  - **Verify:**
    - `cargo test -p thresh-filter --doc` runs the new `validate` example;
    - `RUSTDOCFLAGS=-Dwarnings cargo doc -p thresh-filter --no-deps` passes;
    - the README section is checked by review. The README is not compiled as rustdoc, so no doctest claim is made for it.

## 5. Exact-name protection for native reference targets

- [x] 5.1 Add `scripts/native-reference-manifest.txt`, listing every test in `thresh-association/mht_reference`, `thresh-filter/imm_reference` and `thresh-filter/imm_current_policy`. Add `scripts/check-native-reference-tests.sh`, using only bash and cargo, to implement design.md Decision 9. The script must run under bash 3.2 with utilities portable between BSD and GNU (Decision 9). The checker accepts libtest's ` - should panic` suffix on an `ok` line.
  - **Verify:** with `--no-default-features`, the checker passes locally on macOS under `/bin/bash` 3.2, counts the `#[should_panic]` tests in `imm_reference` as passed, and reports exact per-target and total counts.
- [x] 5.2 Add the checker's `--self-test` mode, which runs against canned libtest output.
  - **Verify:** it fails for each of:
    - missing, unlisted, ignored, failed, filtered-out and zero-test inputs;
    - an empty, duplicated or malformed manifest.

    It succeeds for a matching inventory that includes a `- should panic ... ok` line. The self-test passes both locally on macOS under `/bin/bash` 3.2 and in the ubuntu CI job.
- [x] 5.3 Wire the self-test and then the check into the `native-no-python` job, after its workspace tests, with no Python and no path filtering. Document the command, the manifest and the expected counts in `docs/native-reference-tests.md`.
  - **Verify:**
    - a local run equivalent to CI, with Python discovery disabled (`PYO3_NO_PYTHON=1`), passes;
    - temporarily marking one reference test `#[ignore]` makes the check fail. Revert the change.

## 6. Integrated verification

- [x] 6.1 Run the boundary runner self-tests, the complete exact manifest under the hash-locked CPython 3.12.13 environment, and the feature-enabled bridge doctests.
  - **Record:** the package versions and the exact executed count, in a dated "Recorded reference verification" section of `docs/bridge-testing.md`, following the P0-B section.
  - **Verify:** the SonarCloud coverage step still runs that manifest, with workspace and fresh-process profiles combined and the coverage thresholds unchanged.
- [x] 6.2 Run the native lane:
  - `cargo tree --workspace --exclude thresh-py --no-default-features` contains no PyO3;
  - `cargo build --locked --workspace --exclude thresh-py --no-default-features` succeeds with Python discovery disabled;
  - `cargo test --locked --workspace --exclude thresh-py --no-default-features` passes with Python discovery disabled;
  - the native reference checker passes.

  Record the counts and the maximum oracle residuals in a dated section of `docs/native-reference-tests.md`.
- [x] 6.3 Run the workspace Rust checks.
  - **Verify each passes:**
    - `cargo test --workspace`, in the CI Test job. Locally, run `cargo test --workspace --exclude thresh-py`, because the `thresh-py` test binary does not load on the maintainer's macOS host; this change does not touch `thresh-py`;
    - `cargo test -p thresh-filter --features learned-imm`;
    - `cargo test -p thresh-tracker --features learned-imm`;
    - `cargo clippy --workspace --all-targets -- -D warnings`;
    - `cargo clippy -p thresh-bridge --all-targets --features stonesoup -- -D warnings`;
    - `cargo clippy -p thresh-filter -p thresh-tracker --all-targets --features learned-imm -- -D warnings`;
    - `cargo fmt --all -- --check`;
    - `RUSTDOCFLAGS=-Dwarnings cargo doc --workspace --no-deps`.
- [ ] 6.4 Confirm the SonarCloud pull-request quality gate.
  - **Verify:** the gate passes on new code for cognitive complexity (rule `rust:S3776`, ≤ 15 per function), duplication between `imm_reference.rs` and `imm_current_policy.rs`, and coverage.
- [x] 6.5 Run the specification and diff checks:
  - `openspec validate --all --strict --no-interactive`;
  - `openspec validate --archived --strict --no-interactive`;
  - `openspec status` for every active change;
  - `git diff --check`.

  Then review the final diff against every scenario in the three deltas.
  - **Verify the delta set:**
    - `stonesoup-bridge`: REMOVED "Stone Soup algorithm access" plus ADDED "Stone Soup reference algorithm access", declared as a rename;
    - `mht-tracking`: ADDED "Exhaustive reference verification" only;
    - `imm-estimation`: a new capability;
    - no delta touches `track-management`, `imm-pluggable-leaf-filter` or the other `mht-tracking` requirements.
  - **Verify the implementation diff changes none of:**
    - `openspec/specs/`, or the active `learned-imm-tracker-integration` and `flight-data-training-pipeline` change directories;
    - dependency, feature or version entries in any `Cargo.toml` or requirements file. `Cargo.lock` and `requirements.{in,txt}` are unchanged; only the `thresh-bridge` package description changes;
    - existing IMM tests (4.7);
    - committed scratch-evidence files.
  - **Verify still open:** the MHT and IMM interim gaps and both follow-up changes remain explicitly open.

## 7. Archive handoff

- [ ] 7.1 In the archive/sync PR, before running `openspec archive resolve-stonesoup-mht-imm-claims`, hand-edit the Purpose at `openspec/specs/stonesoup-bridge/spec.md:4` to the JPDA/GM-PHD text recorded in design.md Decision 3. This edit is needed because OpenSpec ignores a delta Purpose for an existing spec.
  - **Verify before archiving:** `openspec/specs/stonesoup-bridge/spec.md:4` names neither MHT nor IMM, and `openspec validate --all --strict` passes.

**Archive-PR acceptance (checked after `openspec archive`; not a tracked task):**
- `openspec show stonesoup-bridge --type spec` shows a Purpose naming neither MHT nor IMM.
- The spec contains "Stone Soup reference algorithm access" and no longer contains "Stone Soup algorithm access".
- `openspec/specs/imm-estimation/spec.md` carries the delta Purpose, not a `TBD` placeholder.
- `openspec validate --all --strict` and `openspec validate --archived --strict` both pass.
