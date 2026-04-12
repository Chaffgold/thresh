## Why

SonarCloud's quality gate is now active on the repository, and its community Rust plugin has flagged six functions that exceed the cognitive complexity threshold of 15 (rule `rust:S3776`). The worst offender is the Hungarian assignment implementation in `thresh-association` at a score of 113 — effectively unreviewable in its current form. The other five functions sit between 16 and 40, which is closer to the threshold but still hurts day-to-day maintainability.

These aren't bugs. The code works and is covered by tests. But high cognitive complexity has real costs: reviewers have to hold more state in their heads, diffs are harder to reason about, and small tweaks to loop/branch-heavy functions carry a disproportionate regression risk. Left unaddressed, the debt compounds — every follow-on change to these functions will either skirt around the complex section or add more complexity on top. Addressing the flagged functions now, while the list is small and bounded, is significantly cheaper than letting the list grow alongside the codebase.

We have also already validated the decomposition approach for similar cases: the great-circle and recentered-ENU tracker `step()` functions were refactored into named phase helpers (`predict_phase`, `build_cost_matrix_phase`, `update_matched`, `update_lifecycle`, etc.) with no behavioral change and significantly improved readability. The same pattern applies cleanly to most of the flagged functions.

## What Changes

- Decompose each of the six flagged functions into smaller, focused helpers named after the phase or sub-task they implement.
- For the Hungarian algorithm (complexity 113), perform real structural work: split the Jonker-Volgenant implementation into cost-matrix reduction, augmenting-path search, and label/slack update phases, each independently testable. Back the refactor with a comparison harness that runs the old and new implementations against random cost matrices until we are confident they agree.
- For the `extract_ground_truth` ADS-B function (complexity 40), extract per-ICAO24 grouping, 1-second grid interpolation, and short-trajectory handling into separate helpers.
- For the mid-complexity cases (orbital dataset frame generation, `Trajectory::generate`, RK4 propagator step, and the stereographic long-traverse test), extract phase helpers following the tracker refactor precedent.
- Preserve exact functional behavior — every existing test in every affected crate must continue to pass without modification of its assertions.

## Capabilities

### New Capabilities

- `cognitive-complexity`: Enforce that the six SonarCloud-flagged functions in `thresh-association`, `thresh-data`, `thresh-tracker`, and `thresh-synth` remain at or below cognitive complexity 15 while preserving their existing behavior and test coverage.

### Modified Capabilities

None. This change is purely a refactor and does not alter any public API, data contract, or observable behavior.

## Impact

**Affected crates:**
- `thresh-association` — Hungarian algorithm (`src/hungarian.rs`)
- `thresh-data` — ADS-B dataset (`src/adsb.rs`), orbital dataset (`src/orbital.rs`)
- `thresh-tracker` — stereographic tracker test (`tests/stereographic_tracker_tests.rs`)
- `thresh-synth` — trajectory generator (`src/trajectory.rs`), orbital RK4 propagator (`src/orbital.rs`)

**Dependencies:** No new crates, no new external dependencies, no feature gating.

**Deployment:** None — internal refactor only.

**Behavior:** Identical. All existing tests must continue to pass without assertion changes. Where the refactor exposes edge cases worth pinning down, we will add regression tests rather than weaken existing ones.

**Verification:**
- `cargo clippy --workspace --all-targets -- -D warnings` clean
- `cargo test --workspace` green
- SonarCloud reports cognitive complexity ≤ 15 for each of the six functions after the refactor
