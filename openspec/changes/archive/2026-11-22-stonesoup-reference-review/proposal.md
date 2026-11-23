# Stone Soup Reference Review

## Why

Thresh relies on Stone Soup as an independent algorithm reference, but its
bridge currently provides no algorithm-level integration tests and a preliminary
source review found incompatible upstream API assumptions. A bounded review is
needed to establish which references are usable, which tests need strengthening,
and which additional algorithms address demonstrated gaps before expanding the
implementation.

## What Changes

- Produce a versioned compatibility assessment of the JPDA, MHT, IMM, and GM-PHD
  bridge claims, including conversion, timestamp, dependency, and CI behavior.
- Design reproducible synthetic reference tests covering numerical agreement,
  association ambiguity, clutter, missed observations, and observation timing.
- Evaluate a short candidate list: exact/approximate JPDA references
  (EHM/EHM2/LBP), Singer maneuver models, RTS smoothing/delayed measurements,
  and covariance-intersection parity. Record adopt-for-testing, investigate,
  or defer decisions rather than automatically adding algorithms.
- Deliver a prioritized follow-up plan that separates compatibility repair,
  reference-test coverage, and optional production algorithm work.

This change tracks the **review**, not implementation. Its deliverable will be
`docs/analysis/stonesoup-reference-review.md`, with review progress in `tasks.md`.
The report is not created by this planning step. Runtime findings must be labeled
unverified until an isolated, reproducible check has actually run.

## Capabilities

### New Capabilities

None. The review adds no runtime behavior.

### Modified Capabilities

None. `stonesoup-bridge` is the existing capability under assessment, not a
requirement silently amended by this review. This documentation-only change uses
`skip_specs: true`; any proposed behavior or capability-claim change belongs in
a separately approved implementation change with appropriate spec deltas.

## Impact

- **Review targets:** `crates/thresh-bridge/`, relevant native Rust tracking and
  fusion components, Python dependency definitions, `.github/workflows/ci.yml`,
  and the existing `stonesoup-bridge` specification.
- **Reference baseline:** Stone Soup v1.9.1 (released 2026-06-24), pinned to its
  source commit in `design.md`; unreleased upstream changes remain separate.
- **Existing context:** the July 2026 orbital/dependency reviews and
  `docs/analysis/advanced-methods-integration.md`; do not re-propose existing
  native CKF, JPDA, MHT, IMM, or covariance intersection as missing algorithms.
- **No production/dependency/CI changes in this review.** Isolated compatibility
  probes may be run during execution of the review, without altering project
  environments or publishing anything.
- **No external flight data.** Use invented trajectories and measurements only;
  do not acquire or store OpenSky or ADS-B Exchange captures.
- **No release work.** Crates.io remains deferred, and the existing strict
  no-regression and documented-rights gates for trained models remain unchanged.
