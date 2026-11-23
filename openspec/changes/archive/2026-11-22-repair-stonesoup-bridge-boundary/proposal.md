# Repair Stone Soup Bridge Boundary

## Why

The completed Stone Soup reference review reproduced interpreter-startup failures,
incomplete measurement conversions, and misleading dependency errors. Reliable
reference comparisons require a tested Rust/Python boundary before repairing
algorithm wrappers or adding algorithms.

## What Changes

- Establish explicit host-owned Python initialization and fallible bridge entry,
  with fresh-process tests independent of other crates initializing Python.
- **BREAKING:** require observation context for measurement conversion: relative
  time origin, common frame, sensor pose, noise, and ADS-B datum/velocity metadata.
  No implicit Unix epoch, degree-as-metre values, or absent-field zero filling.
- Construct correctly ordered float64 detections with actual UTC timestamps and
  attached measurement models carrying the transformed observation covariance.
  Cover radar with/without Doppler, EO/IR, and ADS-B with/without velocity.
- Validate array shape, numeric type, finite values, covariance, and context;
  preserve explicit rejection of OTHR, LiDAR, and Camera.
- Distinguish unavailable interpreter, absent packages, unavailable upstream API,
  invalid conversion, and other Python errors, with deterministic precedence.
- Add synthetic boundary tests and a pinned optional-runtime CI lane that fails
  on missing dependencies, ignored required tests, or zero-test execution.
  Preserve a separately checked native/no-Python build path.

This change covers **P0-A** of `docs/analysis/stonesoup-reference-review.md`.
Edits to algorithm modules are limited to shared runtime/error entry plumbing.
JPDA/GM-PHD graph repairs, MHT/IMM reference selection, numerical algorithm parity,
new tracking algorithms, model training, and checkpoint acceptance are not included.
Existing algorithm-access requirements remain outstanding, not removed or renamed.

## Capabilities

### New Capabilities

None. This extends the existing bridge capability.

### Modified Capabilities

- `stonesoup-bridge`: add explicit lifecycle, observation-context, conversion,
  error-classification, and executed-test contracts. Preserve existing algorithm
  and graceful-degradation requirements; boundary completion does not certify
  those broader promises or repair missing-Python shared-library deployment.

## Impact

- `crates/thresh-bridge/src/{lib,convert,detection,error}.rs`; shared entry/error
  handling in `src/{jpda,mht,imm,phd}.rs`; optional private runtime/context helpers.
- Bridge integration tests, isolated Python dependency constraints, test-manifest
  verification, bridge usage documentation, and `.github/workflows/ci.yml`.
- Public conversion/error APIs change; no Rust callers of the existing detection
  converter were found in the bounded source inspection. External callers need
  migration guidance. Core measurement vectors and native tracker behavior stay
  unchanged.
- Use the reviewed Stone Soup 1.9.1 / PyO3 0.29 baseline, not an unverified version
  upgrade. Reuse core coordinate/time helpers without adding geoid downloads or
  generalized inertial-frame conversion.
- Synthetic inputs only. No OpenSky access or storage, production model release,
  crates.io publication, completed-review archiving, or dependency-PR merging.
- Planning uses installed OpenSpec 1.13.2. Existing CI installs `openspec@latest`
  and currently fails on dependency PRs #162/#163; toolchain reconciliation is a
  separate follow-up, not a reason to weaken this change's validation.
