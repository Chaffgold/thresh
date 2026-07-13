# gospa-metric Specification

## Purpose
GOSPA (alpha=2) set-to-set metric in `thresh-eval` with exact localization/missed/false decomposition, metric-axiom guarantees, and per-frame evaluation alongside the existing MOT metrics, reusing the `thresh-association` Hungarian solver for optimal assignment.

## Requirements

### Requirement: GOSPA metric computation
`thresh-eval` SHALL compute the GOSPA metric of Rahmathullah, García-Fernández, and Svensson (2017) between a ground-truth set and an estimate set of point states, with the `alpha = 2` convention fixed, cutoff distance `c > 0` and order `p >= 1` configurable (default `p = 2`). Base distances SHALL be Euclidean, per-pair costs SHALL be `min(d, c)^p`, and the assignment between the two sets SHALL be optimal (minimum total cost), reusing the existing Hungarian solver in `thresh-association` rather than introducing a new assignment implementation or external dependency. Unassigned elements in either set SHALL incur the `c^p / 2` penalty of the alpha=2 convention.

#### Scenario: Known worked example
- **WHEN** GOSPA is computed with `c = 10`, `p = 2` between truth `{(0,0), (10,0)}` and estimate `{(1,0)}`
- **THEN** the optimal assignment SHALL pair `(0,0)` with `(1,0)` at localization cost `1^2 = 1`, leave `(10,0)` missed at cost `10^2 / 2 = 50`, and the total SHALL equal `(1 + 50)^{1/2} = sqrt(51)` within floating-point tolerance

#### Scenario: Empty-set edge cases
- **WHEN** GOSPA is computed with one or both input sets empty
- **THEN** two empty sets SHALL yield distance 0; an empty estimate against `k` truths SHALL yield `(k · c^p / 2)^{1/p}` (all missed); and the symmetric case (empty truth, `k` estimates) SHALL yield the same value attributed to false tracks

### Requirement: GOSPA decomposition
Every GOSPA evaluation SHALL report the exact decomposition into localization, missed-target, and false-track contributions: the localization component is the sum of assigned-pair `min(d, c)^p` costs, the missed component is `c^p / 2` times the number of unassigned ground truths, and the false component is `c^p / 2` times the number of unassigned estimates. The three components SHALL sum exactly to the p-th power of the reported total, so the decomposition is auditable rather than approximate.

#### Scenario: Decomposition sums to the total
- **WHEN** GOSPA is computed on arbitrary non-degenerate truth and estimate sets
- **THEN** `localization + missed + false` SHALL equal `total^p` within floating-point tolerance, and each component SHALL be individually non-negative

#### Scenario: Pure-miss configuration isolates the missed component
- **WHEN** the estimate set is empty and the truth set is non-empty
- **THEN** the localization and false components SHALL be exactly zero and the missed component SHALL account for the entire total

### Requirement: GOSPA metric axioms
The implemented GOSPA SHALL satisfy the metric axioms on sets: identity (distance is zero if and only if the two sets are identical), symmetry (`d(X, Y) = d(Y, X)`), and the triangle inequality (`d(X, Z) <= d(X, Y) + d(Y, Z)`), for all valid `c > 0` and `p >= 1`.

#### Scenario: Identity and symmetry
- **WHEN** GOSPA is evaluated between a set and itself, and between two distinct sets in both argument orders
- **THEN** the self-distance SHALL be exactly zero, and the two argument orders SHALL yield equal values

#### Scenario: Triangle inequality on randomized sets
- **WHEN** GOSPA is evaluated pairwise over triples of deterministically seeded random point sets (varying cardinalities, at least 100 triples)
- **THEN** `d(X, Z) <= d(X, Y) + d(Y, Z)` SHALL hold (within floating-point tolerance) for every triple

### Requirement: Per-frame GOSPA over evaluation sequences
GOSPA SHALL be computable per frame over the same ground-truth-vs-track frame sequence that the existing MOT metrics consume in `thresh-eval` (matching truth and track positions frame by frame), and SHALL be reported alongside the existing MOT metrics: per-frame values plus a sequence-level summary (the order-p average of per-frame GOSPA and the summed decomposition components). Adding GOSPA SHALL NOT change any existing MOT metric value or report field.

#### Scenario: Sequence evaluation alongside MOT metrics
- **WHEN** an evaluation sequence already scored for MOTA/HOTA/IDF1 is additionally scored with GOSPA
- **THEN** the report SHALL contain per-frame GOSPA totals and decompositions plus the sequence summary, and all pre-existing MOT metric values SHALL be unchanged from an evaluation without GOSPA

#### Scenario: Deterministic evaluation
- **WHEN** the same frame sequence is scored twice with identical GOSPA parameters
- **THEN** every per-frame value, decomposition component, and summary SHALL be bitwise identical between runs, including when the optimal assignment is non-unique (tie-breaking SHALL be deterministic)
