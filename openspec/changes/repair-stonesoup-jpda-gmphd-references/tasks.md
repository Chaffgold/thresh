# Tasks

## 1. Executable JPDA composition and migration

- [x] 1.1 Add explicit predictor/updater constructor inputs and detection probability with documented defaults; build the supported PDA hypothesiser/JPDA graph and verify real pinned-runtime construction plus component identity and configuration mapping.
- [x] 1.2 Add finite/domain configuration validation and preserve operation-specific Python errors; verify probability endpoints, invalid/nonfinite values, missing-package/API classification, and no import attempt for invalid Rust configuration.
- [x] 1.3 Migrate JPDA runtime callers and the compiled `docs/bridge-runtime.md` example; verify initialized-host reuse and successful construction, retain MHT/shared-runtime fresh-process checks, and run feature-enabled doctests without fabricating pre-initialization Python handles.
- [x] 1.4 Document probability-versus-squared-distance gating, clutter units, one compatible measurement space, and timestamp passthrough; verify preserved object identity, distinct observation/call times, and structured errors for invalid Python inputs with focused dispatch and real-runtime tests.
- [x] 1.5 Register all new/renamed JPDA tests in the existing exact manifest and update current JPDA gap wording; verify the runner's inventory checks and all existing boundary tests pass with the new constructor API.

## 2. Independent JPDA mathematical verification

- [x] 2.1 Implement a test-only scalar-Gaussian likelihood and injective joint-event enumeration oracle independent of Stone Soup/native JPDA helpers; verify the seven-event 2x2 and 34-event 3x3 counts and a hand-calculated one-track case.
- [x] 2.2 Add real reference tests for 0-3 tracks/detections, asymmetric/shared observations, ties, no tracks, no detections, all gated out, and zero detection probability; verify finite nonnegative marginals, numeric tolerances, missed entries, row sums and cross-track detection occupancy from design decision 2.
- [x] 2.3 Add independent 1D gate probability checks with interior/exterior and deliberately controlled equality cases; document expected squared threshold and tolerances in the fixtures, register tests, and verify the exact manifest executes them without iteration-order assumptions.

## 3. Executable GM-PHD composition and explicit scan inputs

- [x] 3.1 Build the supported distance/mixture hypothesiser, PHD updater and reducer graph from supplied predictor/updater objects; add and document optional squared measurement gating and existing reducer/default mappings, verifying real construction and invalid configuration tests.
- [x] 3.2 Implement explicit prior/birth/detection/timestamp scan inputs and tagged component, weight, shape, finite-value and temporal validation; verify malformed inputs, reserved birth tags, mismatched/missing/late timestamps and independently empty collections fail or succeed as specified, and reject one-shot iterators in every input position before consuming any collection. Reject repeated prior/birth component identities before copying or graph dispatch while preserving distinct components with equal values/tags or shared arrays/ancestry; verify input preservation and independently expected total intensity.
- [x] 3.3 Deep-copy the combined prior/birth component graph before hypothesis generation and execute hypothesis/update/reduction once each; verify input containers, tags, weights, times, arrays and ancestry remain unchanged on successful scans and injected mutating failures, including a structured copy-failure error.
- [x] 3.4 Return the reduced Gaussian mixture update with provenance and document absent timestamps for empty output; verify real one-step and empty-mixture behavior plus operation-specific error tests replacing the obsolete two-argument dispatch double.
- [x] 3.5 Replace only the PHD known-failure runtime assertion with successful composition while retaining IMM's unsupported-API assertion; update current PHD gap/API migration docs and exact test manifest, and verify the runtime suite and feature-enabled doctests.

## 4. GM-PHD mathematical and reduction verification

- [x] 4.1 Add the independent scalar and nonzero-interval CV prediction/update fixtures from design decision 5; verify detected/missed means, covariances, timestamps, weights and total intensity using neutral reduction and the stated tolerances.
- [x] 4.2 Add birth-only, existing-plus-birth, empty-detection, empty-component and survival/detection endpoint fixtures with neutral reduction; verify pinned detected-birth survival, absent missed births, fresh non-birth tags, non-normalized intensity and updater-stage epsilon substitution, documenting these compatibility limits and subsequent reduction behavior next to the tests.
- [x] 4.3 Test the same reducer configuration/helper used by the wrapper against independent prune-all/redistribution, moment-merge, strict squared threshold, asymmetric-covariance distance, merge clipping and truncation fixtures; verify exact expected weights/moments and no universal intensity-preservation claim.
- [x] 4.4 Add bounded measurement-gating tests distinct from merge-threshold tests and generated-tag/order-insensitive comparisons; register all numeric tests and verify manifest execution plus a deliberate expected-value perturbation fails an oracle assertion without retaining the perturbation.

## 5. Integrated verification and bounded handoff

- [x] 5.1 Extend runner/CI descriptions to include bounded JPDA/GM-PHD reference checks without changing the required job identity or dependency locks; run runner negative self-tests and the complete exact manifest, recording versions, executed counts, and maximum numeric residuals. Include instrumented exact-manifest execution in the existing Sonar coverage report while preserving workspace profiles, subprocess coverage, and production quality thresholds.
- [x] 5.2 Run feature-enabled bridge tests and doctests, `cargo fmt --all -- --check`, bridge Clippy with warnings denied, and bridge docs with `RUSTDOCFLAGS=-Dwarnings`; verify no stale constructor/step examples or unsupported working-reference claims remain in current docs.
- [x] 5.3 Run the native dependency-tree, build and test checks with Python discovery disabled (`--workspace --exclude thresh-py --no-default-features`), plus normal workspace tests; verify the native dependency tree contains no PyO3 and no optional-runtime regression is hidden by a skip.
- [x] 5.4 Run `openspec validate --all --strict --no-interactive`, `openspec validate --archived --strict --no-interactive`, active-change metadata checks and `git diff --check`; review the final diff against every scenario, keep MHT/IMM, native parity, sequence evaluation, rights and release gates explicitly unresolved, and record verification evidence before marking implementation complete.

## Review follow-up

Tasks 3.2 and 5.4 were reopened after PR #168 exposed silent intensity loss for repeated component identities. The user approved rejecting repeated component objects while preserving distinct objects with shared arrays or ancestry. Identity-only validation, duplicate-prior/birth rejection tests, and a real shared-content intensity regression implement that decision. The complete 109-test manifest, 14 runner self-tests, seven doctests, warning-strict bridge lint/docs, independent source review, workspace tests/lint/check, strict live/archive OpenSpec validation, active metadata checks, and whitespace checks pass. Verification evidence is recorded in `docs/bridge-testing.md`; unresolved algorithm, rights, and release scopes are unchanged.
