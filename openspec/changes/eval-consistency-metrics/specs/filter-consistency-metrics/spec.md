# filter-consistency-metrics Specification (Delta)

## ADDED Requirements

### Requirement: NEES computation
`thresh-eval` SHALL compute the normalized estimation error squared (NEES) per Bar-Shalom's formulation: given a ground-truth state `x`, an estimate `x_hat`, and an estimate covariance `P`, NEES = `(x - x_hat)^T P^{-1} (x - x_hat)`. The computation SHALL accept dynamically sized nalgebra vectors/matrices, SHALL support evaluating a subset of the state (e.g., position-only NEES against position-only truth) by taking the error vector and covariance block as supplied by the caller, and SHALL report the degrees of freedom (the error-vector dimension) alongside every score. Inputs with a non-invertible covariance SHALL produce an explicit error, not a silent NaN or panic.

#### Scenario: NEES matches the closed-form value
- **WHEN** NEES is computed for a hand-constructed error vector `e = [1, 2]` and covariance `P = diag(1, 4)`
- **THEN** the result SHALL equal `1 + 1 = 2` (i.e., `e^T P^{-1} e`) within floating-point tolerance, with reported degrees of freedom 2

#### Scenario: Singular covariance is rejected
- **WHEN** NEES is requested with a covariance matrix that is singular (or numerically non-invertible)
- **THEN** the computation SHALL return an explicit error value rather than NaN, infinity, or a panic

### Requirement: NIS computation without ground truth
`thresh-eval` SHALL compute the normalized innovation squared (NIS): given an innovation `nu` and its innovation covariance `S` from a filter update, NIS = `nu^T S^{-1} nu`. NIS SHALL be computable from filter update diagnostics alone — the API SHALL NOT require ground-truth states, so NIS-based consistency checking is available on live/real data where no truth exists.

#### Scenario: NIS from filter diagnostics only
- **WHEN** a filter processes a measurement sequence of `N` updates with measurement dimension `m` and only its per-update innovation and innovation covariance are handed to the NIS computation (no ground-truth argument exists in the call)
- **THEN** the per-update NIS sequence SHALL be produced, and for a consistent linear-Gaussian filter its average SHALL fall inside the two-sided 95% bounds of the time-average test (chi-squared with `N·m` degrees of freedom, divided by `N`)

#### Scenario: NIS matches the closed-form value
- **WHEN** NIS is computed for innovation `nu = [2]` and innovation covariance `S = [[4]]`
- **THEN** the result SHALL equal `1.0` within floating-point tolerance

### Requirement: Time-averaged and per-track consistency variants
The system SHALL provide both time-averaged and per-track aggregation of NEES and NIS. The time-averaged NEES (ANEES) over `K` samples of dimension `n` SHALL be the sample mean of the per-step NEES values, judged against two-sided chi-squared bounds on `K·n` degrees of freedom divided by `K` (Bar-Shalom's time-average test); the analogous average applies to NIS with the measurement dimension. Per-track aggregation SHALL produce one verdict per track identifier so an inconsistent track can be singled out from an otherwise consistent set.

#### Scenario: ANEES aggregation over a track lifetime
- **WHEN** per-step NEES values are aggregated over a track of `K = 100` steps with state dimension `n = 4`
- **THEN** the reported ANEES SHALL equal the sample mean of the 100 per-step values, and its consistency interval SHALL be the two-sided chi-squared bounds for `400` degrees of freedom divided by `100`

#### Scenario: Per-track verdicts isolate the inconsistent track
- **WHEN** two tracks are evaluated where track A's estimates come from an honest filter and track B's covariances have been scaled down by a factor of 100
- **THEN** the per-track report SHALL mark track A consistent and track B inconsistent (ANEES above the upper bound)

### Requirement: Two-sided chi-squared consistency bounds
The system SHALL provide two-sided chi-squared acceptance bounds at a configurable confidence level (default 95%) for a given number of degrees of freedom, and SHALL emit a three-way consistency verdict for each averaged statistic: consistent (inside the bounds), overconfident/optimistic (above the upper bound), or underconfident/pessimistic (below the lower bound). Bound values SHALL agree with published chi-squared quantiles.

#### Scenario: Bounds match reference chi-squared quantiles
- **WHEN** the two-sided 95% bounds are requested for representative degrees of freedom (e.g., 2, 4, 100, 400)
- **THEN** the returned lower and upper bounds SHALL match reference chi-squared quantile values (e.g., from published tables or scipy.stats.chi2 computed offline) within 1% relative tolerance

#### Scenario: Configurable confidence level
- **WHEN** bounds are requested at a 99% confidence level instead of the default 95%
- **THEN** the returned interval SHALL widen accordingly and match the corresponding reference quantiles within the same tolerance

### Requirement: Consistency verdict on a linear-Gaussian reference problem
The NEES/NIS machinery SHALL be validated end-to-end on a linear-Gaussian reference problem where a Kalman filter with the true `Q` and `R` is provably consistent: the honest filter SHALL pass, and dishonest covariance tunings SHALL fail in the correct direction. This is the falsifiability contract — a filter whose covariance is wrong must be detectable even when its point estimates score identically on MOT metrics.

#### Scenario: Consistent filter passes
- **WHEN** a linear KF is run with the exact `Q` and `R` used to generate a seeded linear-Gaussian trajectory and its ANEES (against truth) and average NIS are evaluated at 95% confidence over a sufficient sample (e.g., 50+ Monte Carlo runs or an equivalently long single run)
- **THEN** both statistics SHALL fall inside their two-sided bounds and the verdict SHALL be consistent

#### Scenario: Overconfident filter fails high
- **WHEN** the same reference problem is run with the filter's process noise scaled down (e.g., `Q × 0.01`) so its covariance understates the true error
- **THEN** ANEES SHALL exceed the upper chi-squared bound and the verdict SHALL be overconfident

#### Scenario: Underconfident filter fails low
- **WHEN** the same reference problem is run with the filter's process noise scaled up (e.g., `Q × 100`) so its covariance overstates the true error
- **THEN** ANEES SHALL fall below the lower chi-squared bound and the verdict SHALL be underconfident

### Requirement: Deterministic consistency evaluation
Given identical inputs (error/innovation sequences, covariances, confidence level), all NEES/NIS computations, aggregations, bounds, and verdicts SHALL be deterministic — repeated evaluation produces bitwise-identical results, with no randomness, hash-order dependence, or time dependence in the evaluation path.

#### Scenario: Repeated evaluation is bitwise identical
- **WHEN** the same recorded filter-run inputs are evaluated twice in the same process and across separate `cargo test` invocations
- **THEN** every reported statistic, bound, and verdict SHALL be identical between runs
