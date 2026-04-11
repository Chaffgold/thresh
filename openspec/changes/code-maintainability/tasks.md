## 1. Hungarian Algorithm (complexity 113 → 15)

- [x] 1.1 Read existing `crates/thresh-association/src/hungarian.rs` and identify distinct phases (row/column reduction, augmenting-path search, label/slack update, assignment extraction)
- [x] 1.2 Extract row/column reduction into `reduce_cost_matrix()` with its own unit tests
- [x] 1.3 Extract augmenting-path search into `find_augmenting_path()` with its own unit tests
- [x] 1.4 Extract label/slack update step into `update_labels()` with its own unit tests
- [x] 1.5 Add `#[cfg(test)]` comparison harness that runs old and new implementations against ≥10,000 random cost matrices and asserts identical assignments
- [x] 1.6 Retire the old implementation once the comparison harness has established equivalence
- [x] 1.7 Verify SonarCloud reports cognitive complexity ≤ 15 for the top-level `hungarian_assignment` function
- [x] 1.8 Verify all existing `thresh-association` tests still pass

## 2. ADS-B `extract_ground_truth` (complexity 40 → 15)

- [ ] 2.1 Read `crates/thresh-data/src/adsb.rs` around line 599 and identify the distinct phases in `extract_ground_truth`
- [ ] 2.2 Extract per-ICAO24 grouping into a helper (e.g., `group_states_by_icao24()`)
- [ ] 2.3 Extract 1-second grid interpolation into `interpolate_trajectory()`
- [ ] 2.4 Extract single-sample / short-trajectory handling into its own helper so the main function doesn't branch on trajectory length
- [ ] 2.5 Verify all existing ADS-B tests still pass
- [ ] 2.6 Verify SonarCloud reports cognitive complexity ≤ 15 for `extract_ground_truth`

## 3. Stereographic Tracker Long-Traverse Test (complexity 21 → 15)

- [ ] 3.1 Read `crates/thresh-tracker/tests/stereographic_tracker_tests.rs` around line 178 and map the test to the pattern we used for the recentered-ENU long-traverse test
- [ ] 3.2 Extract measurement-generation loop into a helper
- [ ] 3.3 Extract per-step tracker update into a helper
- [ ] 3.4 Extract final-error computation into a helper
- [ ] 3.5 Confirm the test still exercises the same coverage (same measurements, same assertions)
- [ ] 3.6 Verify SonarCloud reports cognitive complexity ≤ 15 for the test function

## 4. Orbital Dataset Frame Generation (complexity 18 → 15)

- [ ] 4.1 Read `crates/thresh-data/src/orbital.rs` around `OrbitalDataset::frames` / `ground_truth` (≈ line 672+) and identify phases in frame construction
- [ ] 4.2 Extract a frame-building helper for each propagated position
- [ ] 4.3 Extract ground-truth entry construction into its own helper
- [ ] 4.4 Verify all orbital dataset tests still pass
- [ ] 4.5 Verify SonarCloud reports cognitive complexity ≤ 15 for the targeted frame-construction function(s)

## 5. `Trajectory::generate` (complexity 16 → 15)

- [ ] 5.1 Read `crates/thresh-synth/src/trajectory.rs` around line 48 and identify per-segment phases
- [ ] 5.2 Extract per-segment waypoint generation into a helper
- [ ] 5.3 Verify all trajectory tests still pass
- [ ] 5.4 Verify SonarCloud reports cognitive complexity ≤ 15

## 6. Orbital RK4 Propagator Step (complexity 16 → 15)

- [ ] 6.1 Read `crates/thresh-synth/src/orbital.rs` and identify the four RK4 stages in `rk4_step(...)` (≈ line 368)
- [ ] 6.2 Extract RK4 stage computation helper (k1/k2/k3/k4 share structure)
- [ ] 6.3 Verify all orbital propagation tests still pass
- [ ] 6.4 Verify SonarCloud reports cognitive complexity ≤ 15

## 7. CI Guardrail and Documentation

- [ ] 7.1 Confirm the SonarCloud quality gate enforces the ≤ 15 cognitive complexity threshold going forward so new violations are caught at PR time
- [ ] 7.2 Document the phase-helper decomposition pattern in `CLAUDE.md` or a short style guide so new contributors apply it consistently
- [ ] 7.3 Run `cargo clippy --workspace --all-targets -- -D warnings` and `cargo test --workspace` and confirm both are clean before closing the change
