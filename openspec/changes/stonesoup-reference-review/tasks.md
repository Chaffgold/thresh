# Tasks — Stone Soup Reference Review

> Planning is complete when the OpenSpec artifacts validate; the review itself
> is complete only when the evidence and report below exist. All tasks remain
> open at planning time. `report` means
> `docs/analysis/stonesoup-reference-review.md`. Do not repair production code,
> change dependencies/CI, or add algorithms while executing this review.

Execution update (2026-09-22): the report and isolated runtime probes are complete.
All 16 tasks have report evidence; an independent root-agent review checked the
report and prompted the no-Python lane exclusion and explicit CV noise fixture
corrections. Strict change, live-spec and archive validation passed. The report
is presented for maintainer agreement before any follow-up changes. Reproduced
bridge failures are findings, not passing implementation acceptance tests.

## 1. Reproducible baseline

- [x] 1.1 Create the report's version/evidence ledger, recheck the latest stable release, and record the selected tag plus peeled source commit, thresh revision, review date, and distinction between released and unreleased changes. Verify the ledger includes primary links and reconciles the July PyO3/orbital/advanced-methods reviews with the preliminary findings in `design.md`.
- [x] 1.2 Prepare an isolated temporary review environment for the selected Stone Soup version, without modifying project/system environments or lockfiles. Verify import/version output and record Python/NumPy/SciPy/Stone Soup/Rust/PyO3/platform versions, optional dependencies, reproducible setup commands, and existing bridge-test execution/ignored counts in the report; report unavailable runtime checks as blocked, not passing.

## 2. Existing bridge compatibility

- [x] 2.1 Audit and reproduce the JPDA constructor/association mismatch using the selected upstream version. Verify the report maps predictor/updater requirements, `prob_gate`, clutter/detection probabilities, and missed-detection hypothesis semantics to the Rust wrapper, with a minimal reproduction and observed outcome.
- [x] 2.2 Audit the MHT wrapper against a genuine multi-frame hypothesis-tracking contract. Verify the report explains the `MultiTargetMixtureTracker` constructor/stepping mismatch and records a supported reference approach or an explicit unavailable-capability finding; do not relabel single-step mixture reduction as MHT.
- [x] 2.3 Reproduce the IMM import failure and assess supported upstream components or another independent reference approach. Verify the report identifies how mixing, model probabilities, prediction/update, and state-space mapping would be validated, or explicitly records the unsupported contract without claiming working IMM access.
- [x] 2.4 Audit and reproduce the GM-PHD constructor/update mismatch. Verify the report separates prediction, hypothesis generation, intensity update, birth/survival/detection/clutter parameters, and mixture reduction, with exact upstream symbols and reproduction outcomes.
- [x] 2.5 Audit measurement/state conversion and failure handling, including actual detection timestamps, frame/angle/velocity ordering, noise/covariance attachment, unsupported variants, interpreter initialization, and missing dependencies. Verify each supported/unsupported case has an expected outcome and source or probe evidence in the report.

## 3. Reference-test and CI recommendations

- [x] 3.1 Specify deterministic KF/EKF/UKF/CKF and association parity cases using matched model/noise conventions and independent analytical checks. Verify the report defines inputs, expected quantities, absolute/relative tolerances, covariance invariants, normalization checks, and which comparisons are exact versus approximate.
- [x] 3.2 Specify a wholly synthetic sequence matrix covering crossing targets, clutter, births/deaths, missed-observation bursts, maneuvers, 1 Hz/10 Hz/irregular timing, and delayed arrivals. Verify it requires identical observations and independent truth across implementations and defines applicable GOSPA/MOTA/IDF1/identity/NEES/NIS/runtime metrics, repeat-seed reporting, and causal versus smoothed evaluation boundaries.
- [x] 3.3 Recommend pinned optional-runtime and no-Python CI lanes, plus a separately labeled newer-version compatibility check if justified. Verify the report names commands, dependency/interpreter selection, feature flags, executed-test count checks, ignored-test handling, and failure conditions without changing workflow files.

## 4. Algorithm/reference shortlist

- [x] 4.1 Assess EHM/EHM2 exact JPDA and LBP approximate marginals as reference/scaling candidates. Verify the report identifies stable upstream symbols, small-case correctness tests, dense-scene quality/runtime experiments, optional dependencies, and an adopt-for-testing/investigate/defer disposition consistent with the existing native association code.
- [x] 4.2 Assess the Singer maneuver-model reference against July's maneuver/process-noise priorities. Verify the report distinguishes full and approximate Singer covariance, specifies transition/process-covariance comparisons across sampling intervals/damping parameters, and records a bounded disposition rather than committing to a new production model.
- [x] 4.3 Assess RTS smoothing and OOSM examples for delayed-data testing. Verify the report identifies the known-delay assumptions, separates offline/fixed-lag and causal-online metrics, and specifies what would require separate native tracker work before recording a disposition.
- [x] 4.4 Assess covariance-intersection parity and record why GM-LCC, visibility-aware Bernoulli, or broader RFS alternatives should be deferred or separately investigated. Verify each disposition cites a real benchmark gap, maintenance cost, and existing native coverage; do not equate single-target Bernoulli with multi-target tracking.

## 5. Findings and handoff

- [x] 5.1 Complete a prioritized follow-up backlog separating bridge repair, numerical/sequence tests, and optional algorithm additions. Verify each proposed scope names affected code/specs, dependencies, measurable acceptance criteria, and unresolved capability decisions, while preserving synthetic-only data handling, strict trained-model release gates, and deferred crates.io publishing.
- [x] 5.2 Review the report for reproducibility and unsupported claims; verify every compatibility verdict has source/probe evidence, required runtime probes have results rather than silent skips, all candidate dispositions are explicit, and `openspec validate stonesoup-reference-review --strict --no-interactive` passes. Present the review for maintainer agreement before opening or implementing any follow-up change; do not archive this change merely because its planning artifacts exist.
