# Proposal

## Why

The `stonesoup-bridge` requirement "Stone Soup algorithm access" promises Stone Soup MHT and IMM references that cannot exist on the pinned Stone Soup 1.9.1 runtime, which has no Gaussian IMM predictor and no sliding-window MHT that the wrapper can construct. The reference review (`docs/analysis/stonesoup-reference-review.md` §5, P0-C) requires either that contract or an approved amendment backed by exhaustive three-scan MHT tests and hand-derived IMM tests, and the maintainer approved the amendment route on 2026-11-23.

## What Changes

- **CAPABILITY AMENDMENT (approved 2026-11-23).** `stonesoup-bridge` REMOVES "Stone Soup algorithm access", with an in-delta Reason and Migration that state the interim gaps. It ADDS "Stone Soup reference algorithm access" for JPDA and GM-PHD only, keeping the "operational fallbacks until Rust-native versions exist" sentence verbatim. **This REMOVED/ADDED pair is a rename** that drops the two unmet MHT and IMM scenarios; design.md Decision 3 explains why the delta takes this form.
- **BREAKING:** Delete `thresh_bridge::mht` (`MhtTracker`, `MhtConfig`) and `thresh_bridge::imm` (`ImmFilter`, `ImmConfig`), including the `thresh::bridge::{mht,imm}` re-export paths. No placeholders remain.
  - The MHT migration target is native `thresh_association::mht` and the tracker's MHT strategy, which remain unfit for use until the native rebuild.
  - The IMM migration target is native `thresh_filter::imm`.
- Re-home the eight boundary-manifest tests that used the wrappers as test vehicles. The IMM unsupported-API check survives as a library-unit "pinned-absence classification regression" with no upstream-canary claim. The manifest count changes deliberately.
- **New native `imm-estimation` capability:** policy-independent IMM mathematics and configuration validation, an explicit list of mode policies that are not yet specified, and verification against hand-derived exact rationals and an independent nalgebra-only Rust oracle. Labelled characterisation tests in new files pin the current policies; existing IMM tests are not modified.
- **BREAKING (invalid input only):** IMM configuration validation rejects non-finite or negative transition-matrix entries and initial mode probabilities. Fallible paths, including the learned-IMM tracker constructor, return an error instead of constructing a bank from the invalid configuration. Valid configurations and the presets are unaffected.
- **ADD** the `mht-tracking` requirement "Exhaustive reference verification": a test-only exhaustive hypothesis enumerator, independent of the native hypothesis tree, with contract-neutral assertions over invented fixtures at full precision.
- Add exact-name protection for the new native reference test targets in the no-Python CI lane.
- Disclose the interim gaps: a rustdoc warning on the tracker's MHT association strategy, a CHANGELOG `[Unreleased]` correction of the false MHT and bridge claims, and an IMM maneuvering criterion preregistered in `design.md`. The follow-up changes are `rebuild-native-mht-hypothesis-history` and `correct-native-imm-mode-policies`.
- No dependency changes. OR-Tools/MFA was evaluated and declined, an approved deviation recorded in `design.md`.

## Capabilities

### New Capabilities

- `imm-estimation`: The policy-independent IMM mathematics of the native filter bank (mixing with spread, model-conditioned log-likelihood, the exact floor-zero posterior, combined moments), configuration validation, independent verification, and the list of mode policies not yet specified. It supersedes the never-synced archived `imm-estimation` delta (`archive/2026-04-12-imm-filter`), whose maneuver bar moves into the preregistered criterion.

### Modified Capabilities

- `stonesoup-bridge`: REMOVED "Stone Soup algorithm access" and ADDED "Stone Soup reference algorithm access", JPDA and GM-PHD only (rename). The Purpose line naming MHT and IMM is hand-edited at archive time, because OpenSpec ignores a delta Purpose for an existing spec.
- `mht-tracking`: ADDED "Exhaustive reference verification". Its other requirements are unchanged.

## Impact

- **Bridge (`crates/thresh-bridge`):** `src/{mht,imm}.rs` deleted; `src/lib.rs`, the `src/detection.rs` doc text, the `Cargo.toml` description, `src/runtime.rs` (new unit test), `tests/runtime_contract.rs`, the boundary manifest and the runner's status text change.
- **Native code:**
  - `crates/thresh-filter/src/imm.rs`: `ImmConfig::validate`, plus `# Panics` rustdoc on the infallible constructors.
  - New test targets under `crates/thresh-filter/tests/` and `crates/thresh-association/tests/`.
  - `crates/thresh-tracker`: rustdoc only, plus a new validation test file (one test gated on `learned-imm`).
- **CI:** a dependency-free exact-name checker joins the `native-no-python` job. The Stone Soup boundary job keeps its identity and dependency lock.
- **Docs:** the bridge docs, `CHANGELOG.md`, `CLAUDE.md`, `AGENTS.md`, the `crates/thresh-filter` README, a correction note in one analysis document, and a new `docs/native-reference-tests.md`.
- **Unchanged:**
  - `track-management`, `imm-pluggable-leaf-filter`, the other `mht-tracking` requirements, and the active `learned-imm-tracker-integration` and `flight-data-training-pipeline` changes;
  - dependencies, and the hash-locked baseline of CPython 3.12.13, Stone Soup 1.9.1, NumPy 2.5.3, SciPy 1.18.1 and PyO3 0.29.0;
  - no GPU, no crates.io, no provider data; all fixtures are invented and synthetic.
- **Not addressed here:** native MHT conformance and the native IMM mode-policy defects, including the maneuvering requirement (the two follow-ups); MFA cross-checks, newer-version watch lanes, native/reference parity, and release or deployment gates.
