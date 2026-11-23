# Design — Stone Soup Reference Review

## Context

See `proposal.md` for motivation and scope. A design artifact is warranted
because the assessment crosses Rust/PyO3/Python APIs, numerical conventions,
test isolation, and algorithm-equivalence claims.

The preliminary source review was performed on 2026-09-20 against thresh
`develop` at `53622656ea3f9281ed10b4922949e51f342ef8b0`, with unrelated local
OpenSpec housekeeping left intact. Graph tools were unavailable; evidence came
from bounded source reads, Git history, and tagged upstream source. Stone Soup
was not installed in the inspected system Python or project virtual environment;
no runtime algorithm results are claimed.

There has been partial recent review, not a verified full bridge refresh:

- `5d42912` (2026-07-02) migrated PyO3 and rejected unsupported measurement
  variants, but did not add algorithm-level bridge tests.
- `d33f243` (2026-07-12) recorded Stone Soup's orbital deprecation in the
  orbital/ballistic design; orbital validation deliberately uses other golden
  references.
- `f395ab3` (2026-07-12) added the broad advanced-methods integration analysis,
  including association, maneuver-model, smoothing, and delayed-data priorities.
- `crates/thresh-bridge/tests/stonesoup_integration.rs` contains two ignored
  tests: import and nalgebra/NumPy round-trip. Neither executes an algorithm.
  The inspected workflows do not run a Stone Soup runtime lane, and the Python
  training manifest/lock does not pin Stone Soup.

### Preliminary compatibility evidence

These are source findings to reproduce or bound in the review report, not
completed runtime tests. Existing `stonesoup-bridge` requirements remain the
acceptance reference; a broken wrapper does not justify silently deleting its
promised capability.

| Bridge surface | Observed mismatch against v1.9.1 | Review question |
|---|---|---|
| `src/jpda.rs` | `PDAHypothesiser` lacks required predictor/updater inputs; `gate_probability` differs from upstream `prob_gate`. | What explicit component graph and parameter mapping makes association executable? |
| `src/mht.rs` | Unsupported constructor parameters and `track(...)` call on `MultiTargetMixtureTracker`; that class is not itself the promised multi-frame hypothesis tree. | Is there a genuine supported MHT reference satisfying the spec, or a capability gap requiring a separate decision? |
| `src/imm.rs` | Imports `stonesoup.predictor.interacting.IMMPredictor`, absent from the tagged predictor module directory. | Is a supported IMM composition available, or must an independent reference approach be proposed? |
| `src/phd.rs` | Predictor/reducer settings are passed to `PHDUpdater`; parameter names and `update(prior, detections)` differ from its hypothesis-based contract. | Which predictor, hypothesiser, updater, and reducer components are actually required? |
| `src/detection.rs` | Time is stored in metadata rather than `Detection.timestamp`; measurement-model/covariance conventions need an explicit contract. | Can observation time, units, dimensions, and noise semantics survive conversion correctly? |

The `src/` paths above are under `crates/thresh-bridge/`. Tagged upstream evidence:
[PDA hypothesiser](https://github.com/dstl/Stone-Soup/blob/v1.9.1/stonesoup/hypothesiser/probability.py),
[mixture tracker](https://github.com/dstl/Stone-Soup/blob/v1.9.1/stonesoup/tracker/simple.py),
[tracker stepping](https://github.com/dstl/Stone-Soup/blob/v1.9.1/stonesoup/tracker/base.py),
[predictor modules](https://github.com/dstl/Stone-Soup/tree/v1.9.1/stonesoup/predictor),
[point-process updater](https://github.com/dstl/Stone-Soup/blob/v1.9.1/stonesoup/updater/pointprocess.py).

## Goals / Non-Goals

**Goals:**

- Separate source-confirmed incompatibility, runtime-reproduced failure, and
  untested assumptions in a reproducible compatibility matrix.
- Define concrete reference-test assertions and an optional-dependency CI plan.
- Rank additional references against existing native functionality and actual
  synthetic benchmark gaps, with adoption requiring evidence rather than novelty.
- Produce bounded follow-up recommendations without weakening existing specs.

**Non-Goals:**

- Repairing wrappers, installing packages into project/system environments,
  changing lockfiles/CI, implementing algorithms, or altering live requirements.
- Training or distributing model checkpoints, acquiring external flight data,
  or publishing to crates.io.
- Replacing native orbital verification with deprecated Stone Soup orbital APIs.
- Treating an algorithm import, ignored test, or successful build as numerical
  validation.

## Decisions

### 1. Freeze a stable reference; inspect development changes separately

Use Stone Soup **v1.9.1**, released **2026-06-24**, as the initial baseline.
Its annotated tag object is `d9e6fb16f5ae176817aeb6a6fc3a39f544694408`, which
resolves to source commit `a4336b920a799cfe0a77ecb05867c5deeb371c7a`.
Record Python, NumPy, SciPy, Stone Soup, Rust, PyO3, platform, exact commands,
and any optional packages for each future runtime probe. Recheck the latest
stable release when executing the review; document a baseline update explicitly.

Review post-release changes only as a separately dated watchlist. Relevant
recent signals include NumPy compatibility, covariance/Cholesky robustness,
measurement-model changes, and orbital deprecation; their applicability must
be assessed rather than assumed. Sources:
[v1.9.1 release](https://github.com/dstl/Stone-Soup/releases/tag/v1.9.1),
[v1.9 release](https://github.com/dstl/Stone-Soup/releases/tag/v1.9).

**Alternative rejected:** testing floating `main` as the only reference makes
failures difficult to reproduce and conflates released APIs with future work.

### 2. Review contracts before performance or algorithm expansion

For each advertised wrapper, record the claimed capability, exact upstream
symbols/signatures, required components, state/measurement ordering, timestamps,
and error behavior. Reproduce constructor and one-step behavior in an isolated
temporary environment where feasible. Keep reproduction snippets and compact
results in the report; do not scatter ad hoc fixtures into `openspec/`.

A missing IMM module or semantically incorrect MHT substitute is a first-class
finding, not a reason to rename a different algorithm as an equivalent. The
report must propose either a supported reference implementation/composition or
an explicit follow-up capability decision. No such change happens in this review.

**Alternative rejected:** simply bumping a package version cannot repair
incorrect API or algorithm assumptions.

### 3. Separate numerical parity, statistical comparisons, and lifecycle tests

The report's test plan will cover these levels, without pretending they already
exist:

- **Boundary contracts:** interpreter initialization and missing-dependency
  errors, array shape/order/dtype, covariance validity, sensor frame and angular
  units/wrapping, UTC observation timestamps versus arrival time, and explicit
  rejection of unsupported measurement variants.
- **Deterministic numerical parity:** KF/EKF/UKF/CKF prediction and update on
  matched models, with finite symmetric covariance and PSD checks; association
  probabilities normalized including missed-detection hypotheses; analytical
  hand-computable cases alongside Stone Soup. Pin tolerances per quantity and
  precision rather than demanding byte equality across numerical libraries.
- **Synthetic sequences:** crossing targets, clutter sweeps, births/deaths,
  missed-detection bursts, maneuvers, 1 Hz/10 Hz/irregular intervals, and delayed
  measurements. Both implementations receive the exact same materialized
  observations and independent truth, not merely equal RNG seeds.
- **Metrics:** position error, identity switches, GOSPA, MOTA/IDF1, and NEES/NIS
  where meaningful, with repeat-seed uncertainty and runtime/hypothesis-count
  measurements for scaling. Do not equate different association approximations
  or unlike lifecycle policies through unjustified exact-equality assertions.

For the eventual CI proposal, separate the no-Python default-feature lane from
a pinned optional Stone Soup runtime lane. Specify explicit execution of relevant
tests, fail on missing dependencies in that lane, and detect zero-test/ignored-test
false positives. A separate scheduled newer-version compatibility lane is a
proposal to assess, not a replacement for the pinned required lane.

**Alternative rejected:** import-only smoke tests cannot detect the observed
constructor, stepping, and semantic mismatches.

### 4. Evaluate a small, justified algorithm shortlist

| Candidate | Intended review value | Important boundary |
|---|---|---|
| `JPDAwithEHM` / `JPDAwithEHM2` / `JPDAwithLBP` | Exact versus approximate marginal association and scaling under ambiguity/clutter. | Compare small-case marginals, then quality/runtime; these are established references, not all new in v1.9. |
| `Singer` | Correlated-acceleration reference for maneuver-model and process-noise work already identified in July. | Compare transition/process covariance across intervals; distinguish `SingerApproximate` from the full model. |
| `KalmanSmoother` (RTS) and OOSM examples | Offline/fixed-lag smoothing and delayed-observation test design. | Offline future-informed estimates are not causal online baselines; the cited OOSM example assumes known constant delay. |
| Covariance-intersection/Chernoff reference | Validate existing native fusion under unknown cross-correlation. | Reference parity first; not a proposal to add a duplicate production fusion algorithm. |

Sources:
[association](https://github.com/dstl/Stone-Soup/blob/v1.9.1/stonesoup/dataassociator/probability.py),
[Singer](https://github.com/dstl/Stone-Soup/blob/v1.9.1/stonesoup/models/transition/linear.py),
[RTS](https://github.com/dstl/Stone-Soup/blob/v1.9.1/stonesoup/smoother/kalman.py),
[OOSM example](https://github.com/dstl/Stone-Soup/blob/v1.9.1/docs/examples/oosm/KalmanFilterOOSMExample.py),
[Chernoff updater](https://github.com/dstl/Stone-Soup/blob/v1.9.1/stonesoup/updater/chernoff.py).

Record GM-LCC, visibility-aware Bernoulli, and broader RFS trackers as secondary
candidates. Promote one only if a concrete birth/death, cardinality, or occlusion
test gap justifies it. A single-target Bernoulli method is not a drop-in
multi-target tracker. Reconcile choices with
`docs/analysis/advanced-methods-integration.md` rather than opening competing
JIPDA/PMBM/GLMB implementations by default.

**Alternative rejected:** a broad upstream feature import duplicates existing
capabilities and adds maintenance cost without a measured tracking benefit.

### 5. Close the review with evidence and recommendations, not code changes

Create one final report at `docs/analysis/stonesoup-reference-review.md` containing:
the version/environment ledger; capability/API matrix; reproduction outcomes and
limitations; proposed test/CI matrix with assertions; candidate dispositions;
and ordered follow-up scopes with affected files/specs, dependencies, verification
criteria, and rationale.

Mark a review task complete only when its specified report section/evidence exists.
Record blocked probes honestly; do not close a required verification as passing
because installation or API setup failed. A reproduced API failure is a valid
review result, not a working algorithm. Require maintainer agreement before
creating implementation changes or modifying capability claims.

**Alternative rejected:** checking implementation tasks off after a design-only
review would repeat the gap between declared support and actual coverage.

## Risks / Trade-offs

- **Upstream is not an infallible oracle** → use analytic cases and mathematical
  invariants; record upstream bug/deprecation evidence and version boundaries.
- **A compatible constructor hides semantic mismatch** → require genuine
  algorithm/lifecycle behavior, especially multi-frame MHT and IMM mixing.
- **Optional Python makes CI appear green without testing** → propose separate
  no-feature and runtime lanes with explicit executed-test counts.
- **The current environment lacks Stone Soup** → source findings are preliminary;
  runtime checks use an isolated environment and never mutate the training venv.
- **Review grows into an implementation project** → keep one report deliverable;
  seek approval for follow-up implementation and any extra algorithm scope.
- **Fixture provenance or release scope expands accidentally** → use wholly
  synthetic observations, no provider downloads, and no publication steps.

## Migration Plan

No runtime, dependency, or schema migration is performed by this review. Its
report will propose any required migrations and rollback strategy for separately
approved implementation work. Existing feature gates, model-release gates, and
the deferred crates.io plan are unchanged.
