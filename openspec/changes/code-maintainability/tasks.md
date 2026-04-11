## 1. Hungarian Algorithm (complexity 113 → 15)

- [x] 1.1 Read existing `crates/thresh-association/src/hungarian.rs` and identify distinct phases (row/column reduction, augmenting-path search, label/slack update, assignment extraction)
- [x] 1.2 Extract row/column reduction into `reduce_cost_matrix()` with its own unit tests
- [x] 1.3 Implement bipartite augmenting-path matching (`augment_matching` / `try_augment_from` / `find_augmenting_target` / `apply_augmenting_path`) so `greedy_zero_assignment` can be grown into a maximum matching; without this the König vertex-cover check is invalid and the algorithm returns non-optimal matchings on non-trivial inputs
- [x] 1.4 Extract label/slack update step into `update_labels()` with its own unit tests
- [x] 1.5 Add `#[cfg(test)]` comparison harness that ran old and new implementations against ≥10,000 random cost matrices (asserted equal total cost and match count, relaxed from identical assignment vectors because tied optima are non-unique); deleted once equivalence was established
- [x] 1.6 Retire the old implementation after the comparison harness validated equivalence and SonarCloud flagged its complexity
- [x] 1.7 Verify SonarCloud reports cognitive complexity ≤ 15 for the top-level `hungarian_assignment` function
- [x] 1.8 Verify all existing `thresh-association` tests still pass

## 2. ADS-B `extract_ground_truth` (complexity 40 → 15)

- [x] 2.1 Read `crates/thresh-data/src/adsb.rs` around line 599 and identify the distinct phases in `extract_ground_truth`
- [x] 2.2 Extract per-ICAO24 grouping into a helper (e.g., `group_states_by_icao24()`)
- [x] 2.3 Extract 1-second grid interpolation into `interpolate_trajectory()`
- [x] 2.4 Extract single-sample / short-trajectory handling into its own helper so the main function doesn't branch on trajectory length
- [x] 2.5 Verify all existing ADS-B tests still pass
- [x] 2.6 Verify SonarCloud reports cognitive complexity ≤ 15 for `extract_ground_truth`

## 3. Stereographic Tracker Long-Traverse Test (complexity 21 → 15)

- [x] 3.1 Read `crates/thresh-tracker/tests/stereographic_tracker_tests.rs` around line 178 and map the test to the pattern we used for the recentered-ENU long-traverse test
- [x] 3.2 Extract measurement-generation loop into a helper
- [x] 3.3 Extract per-step tracker update into a helper
- [x] 3.4 Extract final-error computation into a helper
- [x] 3.5 Confirm the test still exercises the same coverage (same measurements, same assertions)
- [x] 3.6 Verify SonarCloud reports cognitive complexity ≤ 15 for the test function

## 4. Orbital Dataset Frame Generation (complexity 18 → 15)

- [x] 4.1 Read `crates/thresh-data/src/orbital.rs` around `OrbitalDataset::frames` / `ground_truth` (≈ line 672+) and identify phases in frame construction
- [x] 4.2 Extract a frame-building helper for each propagated position
- [x] 4.3 Extract ground-truth entry construction into its own helper
- [x] 4.4 Verify all orbital dataset tests still pass
- [x] 4.5 Verify SonarCloud reports cognitive complexity ≤ 15 for the targeted frame-construction function(s)

## 5. `Trajectory::generate` (complexity 16 → 15)

- [x] 5.1 Read `crates/thresh-synth/src/trajectory.rs` around line 48 and identify per-segment phases
- [x] 5.2 Extract per-segment waypoint generation into a helper
- [x] 5.3 Verify all trajectory tests still pass
- [x] 5.4 Verify SonarCloud reports cognitive complexity ≤ 15

## 6. Orbital RK4 Propagator Step (complexity 16 → 15)

- [x] 6.1 Read `crates/thresh-synth/src/orbital.rs` and identify the four RK4 stages in `rk4_step(...)` (≈ line 368)
- [x] 6.2 Extract RK4 stage computation helper (k1/k2/k3/k4 share structure)
- [x] 6.3 Verify all orbital propagation tests still pass
- [x] 6.4 Verify SonarCloud reports cognitive complexity ≤ 15

## 7. CI Guardrail and Documentation

- [x] 7.1 Confirmed: the default SonarCloud `rust:S3776` rule (threshold 15) is active in the Sonar Way quality profile on `Chaffgold_thresh`; PRs #32 and #33 both cleared Quality Gate with 0 new issues after their refactors, demonstrating the threshold is enforced at PR time.
- [x] 7.2 Added a "Style Guide: Phase-Helper Decomposition" section to `CLAUDE.md` with a worked-example reference to `hungarian.rs`, `adsb.rs::extract_ground_truth`, and `rk4_stage`.
- [x] 7.3 Ran `cargo clippy --workspace --all-targets -- -D warnings` and `cargo test --workspace`; both clean on this branch.
