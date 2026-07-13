# consistency-benchmark-gates Specification (Delta)

## ADDED Requirements

### Requirement: Optional consistency bounds in scenario baselines
Benchmark scenario TOMLs in `thresh-data` SHALL accept optional consistency-bound fields in their baselines section alongside the existing `mota` / `hota` / `idf1` fields: at minimum a two-sided ANEES interval (lower and upper bound) and a two-sided average-NIS interval. All new fields SHALL be optional with serde defaults so that every existing scenario TOML continues to parse unchanged and no existing scenario's baseline values are altered by this change.

#### Scenario: Existing TOMLs parse unchanged
- **WHEN** every scenario TOML already in the repository is loaded after the new fields are introduced
- **THEN** each SHALL deserialize successfully with all consistency bounds absent, and running its benchmark SHALL produce the same regression-check outcome as before this change

#### Scenario: TOML with consistency bounds parses
- **WHEN** a scenario TOML declares ANEES and NIS bound fields in its baselines section
- **THEN** the manifest SHALL deserialize with those bounds populated and available to the regression check

### Requirement: Benchmark results carry consistency statistics
The benchmark result produced by the shared runner path SHALL optionally carry the consistency statistics needed to enforce the bounds — at minimum the scenario-level ANEES (computed against the scenario's synthetic ground truth) and the average NIS, each with its sample count (from which, together with the per-sample dimension, the chi-squared degrees of freedom follow). Runners that do not (yet) collect filter diagnostics SHALL leave the statistics absent rather than reporting fabricated values, and an absent statistic SHALL be distinguishable from a computed one.

#### Scenario: Runner populates consistency statistics
- **WHEN** a benchmark runner that collects per-update filter diagnostics and ground truth completes a scenario
- **THEN** the benchmark result SHALL contain the scenario-level ANEES and average NIS computed by the `filter-consistency-metrics` machinery

#### Scenario: Absent statistics are explicit
- **WHEN** a runner completes without collecting the diagnostics needed for a statistic
- **THEN** the result SHALL represent that statistic as absent (not zero, not NaN), and any bound asserted against an absent statistic SHALL be reported as a failure rather than silently passing

### Requirement: Regression check enforces consistency bounds
The existing regression-check mechanism (`check_regression` precedent) SHALL enforce declared consistency bounds two-sidedly: a scenario fails when its ANEES or average NIS falls outside the declared interval, in either direction, and each violation SHALL produce a human-readable failure message naming the statistic, its value, and the violated bound. Scenarios that declare no consistency bounds SHALL be checked exactly as today.

#### Scenario: Dishonest covariance fails the gate
- **WHEN** a scenario TOML declares an ANEES interval calibrated for an honest filter, and the benchmark is run with the filter's process noise scaled down so its covariance is overconfident
- **THEN** the regression check SHALL return a failure identifying ANEES as above the declared upper bound, even if MOTA/HOTA/IDF1 still meet their baselines

#### Scenario: Honest covariance passes the gate
- **WHEN** the same scenario is run with the correctly tuned filter
- **THEN** the regression check SHALL return no consistency failures, and the MOT baseline checks SHALL behave exactly as before

#### Scenario: Underconfident covariance also fails
- **WHEN** the benchmark is run with process noise scaled up so ANEES falls below the declared lower bound
- **THEN** the regression check SHALL return a failure identifying ANEES as below the lower bound (bounds are two-sided, not a one-sided ceiling)

#### Scenario: Deterministic gate outcome
- **WHEN** the same scenario with consistency bounds is run twice with the same seed
- **THEN** the computed consistency statistics and the pass/fail outcome SHALL be identical between runs
