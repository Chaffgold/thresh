# Proposal

## Why

The Stone Soup review identified unsupported constructor and call contracts in the JPDA and GM-PHD wrappers. With the bridge boundary repaired in PR #166, the next prerequisite for trustworthy reference comparisons is to make these two graphs executable and verify their mathematics on independent, synthetic fixtures.

## What Changes

- **BREAKING**: Require caller-supplied predictor and updater objects for JPDA construction; make detection probability explicit alongside gate probability and clutter spatial density.
- **BREAKING**: Replace GM-PHD's unsupported direct updater call with an explicit hypothesiser, intensity updater, and reducer composition; require an update timestamp and explicit caller-owned birth components.
- Define configuration domains, measurement-space units, missed-detection behavior, birth/survival semantics, and reduction behavior for the pinned Stone Soup 1.9.1 runtime.
- Require reusable GM-PHD input collections and unique prior/birth component objects, while retaining support for distinct components that share values, tags, arrays, or ancestry.
- Add executable Rust-to-Python constructor and one-step tests, independent small JPDA event enumeration, and hand-derived GM-PHD intensity/reduction fixtures.
- Preserve the host-owned interpreter, observation conversion, structured-error, feature-isolation, and exact executed-test-manifest contracts established by the boundary repair.

## Capabilities

### New Capabilities

None.

### Modified Capabilities

- `stonesoup-bridge`: Add executable JPDA and GM-PHD reference contracts and bounded mathematical verification without claiming full tracking accuracy or relaxing the existing MHT/IMM requirements.

## Impact

- Primary implementation: `crates/thresh-bridge/src/jpda.rs` and `phd.rs`, their call sites and tests, optional bridge test runner/manifest, and bridge documentation.
- Planning prerequisite: completed `repair-stonesoup-bridge-boundary` (merged in PR #166). Evidence and sequencing: `docs/analysis/stonesoup-reference-review.md`, sections 2.1, 2.4, and 5 (P0-B).
- Reuse the existing pinned Python/Stone Soup environment and optional CI lane; no dependency upgrades or new solvers.
- MHT/IMM repairs, native/reference numerical parity, sequence-level evaluations, algorithm adoption, model releases, and crates.io publishing remain outside this change.
- All fixtures are invented synthetic data. No OpenSky or other provider data is acquired, retained, or redistributed. The strict no-regression release gate and data/model-rights requirements remain unchanged.
