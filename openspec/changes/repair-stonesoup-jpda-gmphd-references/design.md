# Design

## Context

See `proposal.md` for motivation and scope. P0-A landed in PR #166; the current wrappers still use unsupported upstream constructors/calls. The baseline is CPython 3.12.13, Stone Soup 1.9.1, NumPy 2.5.3, SciPy 1.18.1, and PyO3 0.29.0, with the existing hash-locked Python requirements and Cargo lockfile unchanged.

Evidence was checked against `src/jpda.rs`, `src/phd.rs`, `tests/runtime_contract.rs`, `docs/bridge-runtime.md`, and the installed pinned upstream source. Relevant upstream modules are `hypothesiser/{probability,distance,gaussianmixture}.py`, `dataassociator/probability.py`, `updater/pointprocess.py`, `mixturereducer/gaussianmixture.py`, and `types/{state,mixture}.py`. These checks supplement the historical review's sections 2.1/2.4; successful Python probes are not evidence that the Rust wrappers already work.

## Goals / Non-Goals

- Establish executable, small linear-Gaussian references with independent numerical expectations and documented upstream behavior.
- Keep caller-selected dynamics/noise, host-owned initialization, existing feature gates, and operation-specific error capture.
- Do not introduce a tracker, a native GM-PHD implementation, learned/adaptive births, CPHD/LCC, spawned targets, or simultaneous heterogeneous-sensor likelihoods.
- Do not compare native approximate JPDA directly to exact reference marginals: the native algorithm uses a different approximation and missed-weight convention. Native/reference parity is a separate scope.
- Nonlinear boundary conversion remains covered by P0-A; this change does not certify nonlinear algorithm results or resolve MHT/IMM/deployment gaps.

## Decisions

### 1. Caller-composed JPDA with bounded configuration

Change construction to `JpdaAssociator::new(config, predictor: &Py<PyAny>, updater: &Py<PyAny>)`. Retain `associate(tracks, detections, timestamp)` and Python object identity. Use the existing attachment helper and structured error mapping; do not initialize Python internally.

Construct `PDAHypothesiser(predictor=predictor, updater=updater, prob_gate=Pg, prob_detect=Pd, clutter_spatial_density=clutter_density, include_all=False, normalise=True)`, then `JPDA(hypothesiser)`. Use explicit keyword arguments; upstream's positional property order differs from this presentation. Map the existing `gate_probability` and density fields directly, and add `detection_probability` defaulting to 0.85 (the previously implicit upstream default). Retain existing gate 0.99 and density 1e-6 defaults.

Validate finite `0 < Pg < 1`, `0 <= Pd <= 1`, and density `lambda > 0` before attachment using `BridgeError::AlgorithmError`. The strict gate interval is a deliberate bridge policy, not an upstream guarantee: it avoids infinite/zero gates and the zero missed-weight case `Pg=Pd=1`. A hard-coded predictor/updater would invent dynamics/noise, so it is rejected. Exposing additional normalization modes is unnecessary for this bounded exact reference.

Gate membership uses `d² <= chi_squared_inverse_cdf(Pg, measurement_dimension)`. Density has inverse measurement-volume units; fixtures and callers must use one compatible measurement space. The default numeric density is not a universal angular/Cartesian conversion.

JPDA passes times through: missed predictions use the association timestamp; detected hypotheses use each detection's timestamp. There is no call-time fallback for `None`, and upstream can predict backward. Document this behavior; do not silently add an out-of-sequence policy. Use explicit timezone-aware UTC times in supported fixtures and assert a distinct-time case.

### 2. Independent JPDA oracle

Use a scalar observation of two-state constant velocity: initial covariance `diag(1,1)`, measurement variance 1, diffusion 0.1, elapsed time 1, giving predicted measurement variance `S = 3 + 0.1/3`. Compute Gaussian likelihoods independently with ordinary arithmetic.

For each track, use raw detected weights `Pd * N(z; predicted_mean, S) / lambda` and missed weight `1 - Pd * Pg`. Enumerate assignments in `{miss, d1, ..., dm}`, discard events reusing a detection, normalize products across events, and sum into per-track marginals. Per-track normalization factors cancel. A 2x2 fixture has seven events and a 3x3 fixture has 34.

Cover 0-3 tracks/detections, shared observations, asymmetric likelihoods, ties, empty inputs, all gated out, and `Pd=0`. Compare by fixture identity, not Python set order; absent gated hypotheses count as zero. Assert finite nonnegative marginals, use `atol=1e-12`, `rtol=1e-10`, row sums within `1e-12`, and detection occupancy at most `1+1e-12`. Independently check the 1D 0.99 gate against squared distance 6.634896601021211 with clearly interior/exterior observations; test equality deliberately without hiding rounding tolerance in general gate membership.

Pinned SciPy quantiles can differ by one float64 ULP across platforms. For the controlled equality fixture only, calibrate real residual/covariance inputs over a bounded set of positive measurement variances until the computed squared distance equals the runtime threshold exactly and the adjacent larger observation is strictly outside. Failure to find such inputs is a test failure. Keep the independent quantile and marginal-oracle checks separate and unchanged.

### 3. Explicit GM-PHD graph and scan API

Keep `PhdFilter::new(py, config, predictor, updater)`, with live Python token and supplied components. Own the following graph:

1. `DistanceHypothesiser` with `SquaredMahalanobis` and a configured `missed_distance`.
2. `GaussianMixtureHypothesiser(..., order_by_detection=True)`.
3. `PHDUpdater(updater=..., clutter_spatial_density=..., prob_detection=..., prob_survival=..., normalisation=True)`.
4. `GaussianMixtureReducer(prune_threshold=..., merge_threshold=..., max_number_components=...)`.

Replace the unsupported two-argument update with `filter(prior, birth_components, detections, timestamp)`, each Python input passed as `&Py<PyAny>`. Validate the scan, prepare independent working components, call `hypothesise(components, detections, timestamp)`, call `update(hypotheses)` once, reduce its components, and return the `GaussianMixtureUpdate` with reduced components and hypothesis provenance. An empty mixture's component-derived timestamp is `None`; do not fabricate a component to carry time.

Require tagged weighted Gaussian inputs: every explicit birth must carry tag `birth`, and no prior may carry that reserved tag. Require common prior timestamps no later than the scan; birth and detection times must equal the scan time. All required times must be timezone-aware UTC values. Validate finite nonnegative component weights (which may exceed one); do not normalize their sum. Validate supported state/covariance shape, finiteness and compatible dimensions without adding jitter or changing caller models. Contract violations use `AlgorithmError`; unexpected upstream failures retain their original structured Python cause. Numeric config validation precedes imports; Python-owned input checks run attached before hypothesis generation.

The user approved requiring reusable collections (lists, tuples, sets, or Gaussian mixtures) and rejecting one-shot iterators/generators. Check all three collection inputs before iterating any of them. Reading a generator would consume caller state, contradicting preservation; silently materializing it is therefore not a compatible fallback. Test rejection with observable unconsumed generators in every input position.

Apply `copy.deepcopy` once to the combined fresh prior-and-birth list before hypothesis generation, preserving relationships within that working graph without aliasing caller components. A fresh outer list is insufficient: state construction can share arrays, and prediction can mutate reachable component ancestry. Do not copy predictor/updater objects. A copy failure returns its structured Python exception with operation `PHD component copy`; never fall back to shallow copying. Test unchanged containers, tags, weights, timestamps, arrays and ancestry on success and on an injected mutating failure. Keep this a per-scan operation without hidden persistent birth or track state. An implicit tracker/birth heuristic would alter the reference problem and is excluded.

### 4. Pin semantics rather than silently correct upstream

Keep existing probability/density/reducer defaults. Map `clutter_intensity` to upstream `clutter_spatial_density`; document inverse measurement-volume units. Validate finite `Ps,Pd` in [0,1], positive finite density, nonnegative finite prune/merge thresholds, and `max_components > 0`. Add `gating_threshold_squared: Option<f64>` default `None` (unbounded); `Some(x)` requires finite `x > 0`. This is distinct from merge threshold. Existing merge default 4.0 already means squared Mahalanobis distance: do not square it again.

For an existing component, use missed weight `(1-Pd) Ps w` and detected weight `Pd Ps w q / (kappa + sum_j Pd Ps w_j q_j)`. The pinned mixture hypothesiser retags detected births before updating, so detected births also receive `Ps`; undetected births retain the birth tag and are omitted from the missed branch. Exact zero updater weights become float64 epsilon before reduction; pruning can subsequently remove or redistribute those components. Use neutral reduction for endpoint fixtures. These are compatibility decisions, not textbook claims; changing them requires a separately approved algorithm change.

Reducer behavior must remain explicit:

- Pruning redistributes removed weight equally over survivors; pruning everything yields an empty mixture.
- Merging uses squared Mahalanobis distance with the candidate component covariance and a strict `<` threshold. Pairwise merged weight is capped at one; total intensity is not always preserved.
- Truncation retains the largest weights and redistributes removed mass equally among them.
- Generated tags and output order are not stable identifiers. Assert numeric contents and birth/non-birth semantics, not UUID literals or fabricated track IDs.

Neutral reduction for update fixtures means prune=0, merge=0 and a sufficiently large component cap. The strict merge comparison makes zero disable numeric merging.

### 5. Independent GM-PHD fixtures

For a scalar random walk with zero process noise, `m=0, P=1, w=.5`, `H=1, R=1, z=0`, `Ps=.8, Pd=.5`, define `q=1/sqrt(4*pi)` and `kappa=.2*q`. With neutral reduction:

| Inputs | Expected components / total intensity |
|---|---|
| Existing component + detection | detected `(w=.5,m=0,P=.5)`, missed `(w=.2,m=0,P=1)`; total .7 |
| Existing component + empty scan | missed .2 |
| Birth .5 + detection | detected .5, no missed birth |
| Birth + empty scan | empty |
| Existing .5 + birth .25 + detection | detected .4 and .2, existing miss .2; total .8 |
| No components + any detections | empty |

Include a nontrivial one-second CV prediction: `m=[0,1], P=I`, process diffusion 0, `H=[1,0], R=1, z=2`, same weight/probabilities, and `kappa=.2*exp(-1/6)/sqrt(6*pi)`. Prediction is `m=[1,1]`, `P=[[2,1],[1,1]]`; detected output has weight .5, mean `[5/3,4/3]`, covariance `[[2/3,1/3],[1/3,2/3]]`; missed output has weight .2 and predicted moments. Both timestamps equal the scan time.

Test reducer operations independently from updates: prune [.05,.2,.3] at .1 to [.225,.325]; prune all; merge .2 at mean0 and .3 at mean1 (unit variances) to weight .5, mean .6, variance 1.24; clip a .7+.6 merge to1; keep means0/2 separate at squared threshold4; truncate [.1,.2,.3] to [.35,.25]. Include an asymmetric covariance fixture to verify which covariance defines merge distance. Use `atol=1e-12`, `rtol=1e-10` on small numeric fixtures and explicit epsilon assertions at Ps/Pd endpoints. Test bounded measurement gating separately from mixture merging.

### 6. Extend the existing verification lane

Keep the current locked environment and required `Stone Soup boundary` job; extend its wording to distinguish boundary checks from bounded JPDA/GM-PHD reference checks without changing branch-protection job identity. Register new reference targets/names in `tests/boundary-test-manifest.json`, matching the runner's complete Cargo inventory checks. Retain runner negative tests for missing/ignored/zero tests and dependency failures. Dispatch doubles test error/context handling only, never substitute for real numerical tests.

The existing Sonar lane also executes that exact manifest under coverage instrumentation and combines its profiles, including fresh-process children, with normal workspace profiles. Keep a shared coverage target/profile directory so neither run discards the other. Classify fixture files as tests; do not lower thresholds or exclude production bridge code to satisfy the coverage gate.

Preserve native/no-Python checks (excluding unconditional `thresh-py`). Do not relax required lanes, install OR-Tools, upgrade dependencies, or introduce provider captures. Historical review evidence remains historical; update current status text in runtime/detection docs and runner/CI comments after implementation.

## Risks / Trade-offs

- Pinned birth survival and merge clipping can surprise consumers → document and assert exact behavior; do not promise mass conservation or production tracking quality.
- Caller-supplied Python objects admit unsupported combinations → define the supported linear-Gaussian profile, reject explicit contract violations, retain upstream diagnostic details for failures outside it.
- Python component mutation and unstable ordering can invalidate fixtures → isolate working components, assert input preservation, compare outputs by numeric content/fixture identity.
- Tiny mathematical oracles are not sequence-level accuracy evidence → keep native parity, sequence evaluation, and MHT/IMM decisions as separate backlog scopes.

## Migration Plan

1. Update config structs and both wrapper APIs plus every Rust/doc caller in one implementation change; no compatibility shim may invent predictor, updater, births, or timestamps.
2. Update JPDA initialized-host/import tests with supplied Python objects. Valid object handles cannot exist before host initialization: remove only the impossible JPDA constructor assertion from the fresh-process test, retaining MHT and shared-attachment/standalone entry-point coverage. Verify JPDA constructor/call reuse of the initialized host.
3. Replace the PHD known-constructor-failure expectation with successful real composition; preserve IMM's independent unsupported-API regression test. Replace the obsolete PHD dispatch double with graph-step and operation-specific error tests.
4. Migrate the compiled `docs/bridge-runtime.md` example to explicit supported components, revise current-gap text in `docs/bridge-detections.md`, and run feature-enabled doctests.
5. Require strict OpenSpec validation, exact runtime manifests, native checks, lint, and documentation checks before merging. Record exact executed counts and residuals; only then complete tasks and request verification/archive.

Rollback, if needed, reverts the wrapper/API/test changes together. It must not reintroduce claims that the old unsupported constructors provide working references. No release or crates.io action is part of this plan.
