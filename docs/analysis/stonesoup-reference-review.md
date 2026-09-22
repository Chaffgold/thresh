# Stone Soup reference review — 2026-09-22

## Outcome and scope

The current optional bridge is **not a verified algorithm reference or operational
fallback**. Its four advertised algorithm constructors fail against Stone Soup
1.9.1; explicitly executing its two existing ignored tests also fails before any
algorithm runs. Fix interpreter/conversion contracts and repair the viable
component graphs before considering additional production algorithms.

This is the deliverable for `stonesoup-reference-review`, not an implementation
of those repairs. Only invented measurements were used. No provider data, trained
weights, project environments, lockfiles, workflow files, or production code were
changed. Crates.io remains deferred. A passing compatibility probe is not a
trained-model acceptance result; documented rights and strict no-regression
release gates remain in force. Follow-up scopes below require maintainer agreement.

## 1. Reproducible baseline

| Item | Observed baseline |
|---|---|
| Review date / thresh revision | 2026-09-22; `2bce9909e572808b69cf1dbf263ddb49d0282557` (OpenSpec cleanup on top of `53622656ea3f9281ed10b4922949e51f342ef8b0`; production bridge unchanged) |
| Latest stable, rechecked through GitHub release API | **v1.9.1**, published `2026-06-24T13:57:38Z` |
| Annotated tag object | `d9e6fb16f5ae176817aeb6a6fc3a39f544694408` — **not** the source commit |
| Peeled source commit | `a4336b920a799cfe0a77ecb05867c5deeb371c7a` |
| Runtime | CPython 3.12.13; Stone Soup 1.9.1; NumPy 2.5.3; SciPy 1.18.1 |
| Rust / bridge | rustc 1.97.0 (`2d8144b78`); Cargo 1.97.0 (`c980f4866`); locked PyO3 0.29.0; nalgebra 0.33.3 |
| Platform / installer | macOS 26.6, Darwin 25.6.0 arm64; uv 0.11.26 |
| Optional probe dependency | OR-Tools 9.14.6206, installed only after recording the missing-dependency result |
| Evidence method | Verify-tier bounded source inspection plus runtime probes. Graph MCP tools were unavailable: no project/generation/coverage claim is made. This is not an exhaustive codebase or upstream audit. |

Primary baseline evidence: [release](https://github.com/dstl/Stone-Soup/releases/tag/v1.9.1),
[tag object](https://api.github.com/repos/dstl/Stone-Soup/git/tags/d9e6fb16f5ae176817aeb6a6fc3a39f544694408),
[source commit](https://github.com/dstl/Stone-Soup/commit/a4336b920a799cfe0a77ecb05867c5deeb371c7a).
The v1.9.1 NumPy 2.5 fix matters to this environment; it does not repair thresh's
incorrect constructor assumptions.

The July reviews were partial, not nonexistent: `5d42912` migrated PyO3 and
rejected unsupported measurement variants; `d33f243` recorded orbital
deprecation; `f395ab3` added the
[advanced-methods analysis](advanced-methods-integration.md). Its association,
Singer/process-noise, RTS, and delayed-observation priorities remain useful.
Its historical absence claims must not be copied forward: current
`thresh-eval/src/{consistency,gospa}.rs` already implements NEES/NIS and GOSPA,
and `thresh-core/src/time.rs` now has absolute `Epoch` alongside relative
`Timestamp`. Native CKF, JPDA, MHT, IMM, and covariance intersection also exist.

### Released versus unreleased

On the review date, GitHub's comparison reported `main` **68 commits ahead**,
ending at `8d1edeb07ef8505ed065cbef435cfb5e517d9bdc`. No development build was
installed or tested. Relevant watchlist items, not claims about v1.9.1:

- [Kalman inverse to pseudoinverse](https://github.com/dstl/Stone-Soup/commit/ec120b8d96c31a73c407cc58a5f6e74bdaeb9763): singular-innovation behavior needs a separate test.
- [GNN infinite missed-distance fix](https://github.com/dstl/Stone-Soup/commit/9e83f33bbc15a6b274ba858784bf5a820dec7b24): do not use unbounded miss costs as a trusted stable-release oracle.
- [Vectorized measurement inverses](https://github.com/dstl/Stone-Soup/commit/215af2f38ee6c89139d1f2204c01288a7a1e3a23): test shape/order separately when upgrading.
- [Orbital removal](https://github.com/dstl/Stone-Soup/commit/7018f6dd28d7c01b2fd8b646b4f9b7d6414c0434): retain thresh's independent orbital references.
- [Weighted accumulated-state registration](https://github.com/dstl/Stone-Soup/commit/2cbe8527f0a72382fa9fe08808a27d8e9c930584): assess only if an accumulated-state delayed-data reference is selected.

### Isolation and commands

Actual temporary root: `/private/tmp/thresh-stonesoup-review.jxzf8Q`. All Python
probes ran with its interpreter. The project/system Python environments were not
modified. A fresh reproduction can use:

```sh
review_dir=$(mktemp -d /private/tmp/thresh-stonesoup-review.XXXXXX)
uv venv "$review_dir/venv" --python 3.12.13
uv pip install --python "$review_dir/venv/bin/python" \
  'stonesoup==1.9.1' 'numpy==2.5.3' 'scipy==1.18.1'
export PYO3_PYTHON="$review_dir/venv/bin/python"
export CARGO_TARGET_DIR="$review_dir/target"
cargo test --locked -p thresh-bridge --no-default-features
cargo test --locked -p thresh-bridge --features stonesoup --test stonesoup_integration
cargo test --locked -p thresh-bridge --features stonesoup \
  --test stonesoup_integration -- --ignored --test-threads=1 --nocapture

# Execute Appendix A directly; the Rust commands do not run these Python probes.
run_review_python() {
  awk -v blocks="$1" '
    /^## Appendix A\./ { appendix = 1 }
    appendix && /^```python$/ { block++; code = 1; next }
    appendix && /^```$/ { code = 0 }
    code && block <= blocks { print }
  ' docs/analysis/stonesoup-reference-review.md | "$review_dir/venv/bin/python" -
}
run_review_python 1  # Includes MFA import: capture missing OR-Tools before installation.
uv pip install --python "$review_dir/venv/bin/python" 'ortools==9.14.6206'
run_review_python 2  # Recreates base state, then executes the MFA continuation.
```

The first Python invocation records the missing-dependency result in §2.2. The
second starts a fresh interpreter, repeats the base block to recreate its state
(the MFA import now succeeds), and then runs the four-scan MFA continuation.

Installed transitive versions for reproducing the observed environment (pin
these too when creating a future CI constraints file): `contourpy=1.4.0`,
`cycler=0.12.1`, `fonttools=4.65.0`, `kiwisolver=1.5.1`, `matplotlib=3.11.2`,
`narwhals=2.26.0`, `ordered-set=4.1.0`, `packaging=26.3`, `pillow=12.3.0`,
`plotly=7.1.0`, `pymap3d=3.2.0`, `pyparsing=3.3.3`,
`python-dateutil=2.9.0.post0`, `rtree=1.4.1`, `ruamel.yaml=0.19.1`,
`six=1.17.0`, `utm=0.9.0`. Adding OR-Tools installed `absl-py=2.5.0`,
`immutabledict=4.3.1`, `pandas=3.0.6`, `protobuf=6.31.1`, and
`typing_extensions=4.16.0`. No `pyehm`, GPU, plotting execution, or provider-data
dependency was needed for the selected probes.

| Actual Rust command | Executed / ignored / result |
|---|---|
| Bridge `--no-default-features` | Compiles; **0 tests executed**, 0 ignored in each unit/integration/doc target. Build isolation only. |
| Bridge `--features stonesoup --test stonesoup_integration` | **0 executed, 2 ignored**; not runtime coverage. |
| Same target with `-- --ignored --test-threads=1 --nocapture` | **2 executed, 0 passed, 2 failed, 0 ignored**. Both panic because the interpreter is uninitialized and `auto-initialize` is disabled. |

Neither a successful build nor these ignored tests validates an algorithm.
An additional temporary Rust executable explicitly called `Python::initialize()`
before probing the **unchanged** bridge. It used PyO3 `=0.29.0`, nalgebra `0.33`,
and path dependencies on `thresh-core`/`thresh-bridge` with `stonesoup` enabled.
It had its own temporary Cargo lock, not the workspace lock (some transitive
patch versions consequently differ from the locked test run). On this embedded
Python setup, `PYO3_PYTHON` alone did not expose venv site-packages; the initialized
probe used `PYTHONPATH="$review_dir/venv/lib/python3.12/site-packages"`.

## 2. Compatibility findings

### 2.1 JPDA — executable upstream composition, broken wrapper

[`jpda.rs`](../../crates/thresh-bridge/src/jpda.rs) builds `PDAHypothesiser` without
predictor/updater and passes `gate_probability`. Actual Python and initialized
Rust calls fail with `TypeError` for missing `predictor`. Supplying predictor
and updater exposes the second failure: unexpected `gate_probability`.

The supported graph is Kalman-family predictor + updater/measurement model →
`PDAHypothesiser(..., prob_gate=..., prob_detect=..., clutter_spatial_density=...)`
→ `JPDA(hypothesiser).associate(tracks, detections, timestamp)`. `prob_gate` is a
probability, not a squared Mahalanobis threshold; convert via the measurement
dimension's chi-squared CDF/quantile when matching native gates. The wrapper
does not expose detection probability, leaving upstream's default 0.85 implicit.
Missed hypotheses carry mass proportional to `1 - P_D P_G`; detection hypotheses
use likelihood, detection probability, and spatial clutter density. Preserve
the miss entry when normalizing; do not discard it as a falsey object.
[Upstream contract](https://github.com/dstl/Stone-Soup/blob/v1.9.1/stonesoup/hypothesiser/probability.py).

The two-track/two-detection synthetic probe in Appendix A returned valid
normalized marginals. With no detections, the single track's missed probability
was exactly 1.0. This verifies **upstream composition**, not the broken Rust
wrapper or the native algorithm's equivalence.

### 2.2 MHT — incorrect substitute; MFA is a bounded reference candidate

[`mht.rs`](../../crates/thresh-bridge/src/mht.rs) fails for missing `initiator`.
`MultiTargetMixtureTracker` requires initiator, deleter, detector, data associator,
and updater; it has no `track(detections, timestamp)` method. Its supported
manual step is `update_tracker(time, detections)`. More importantly, it collapses
each scan's Gaussian mixture; fixing argument names does not create the promised
multi-frame hypothesis tree.
[Tracker source](https://github.com/dstl/Stone-Soup/blob/v1.9.1/stonesoup/tracker/simple.py),
[step contract](https://github.com/dstl/Stone-Soup/blob/v1.9.1/stonesoup/tracker/base.py).

There **is** a more appropriate upstream reference composition:
`MFAHypothesiser(PDAHypothesiser(...))` + `MFADataAssociator(..., slide_window=3)`
with tagged Gaussian-mixture track histories and Kalman updates. The first
import failed clearly for absent optional `ortools`. After installing the pinned
optional dependency, a four-scan invented crossing probe ran: per track,
component counts were `[3, 9, 9, 9]`, tag lengths `[1, 2, 3, 4]`; the number of
distinct first-scan tag choices dropped from 3 to 1 at scan 3. This is evidence
of retained history and pruning, **not** a complete MHT parity or optimality proof.
The upstream solver uses a finite iteration limit and dual/primal gap criterion;
require tiny exhaustive-history checks before treating it as a golden oracle.
[MFA associator](https://github.com/dstl/Stone-Soup/blob/v1.9.1/stonesoup/dataassociator/mfa/__init__.py),
[solver](https://github.com/dstl/Stone-Soup/blob/v1.9.1/stonesoup/dataassociator/mfa/_step.py),
[example](https://github.com/dstl/Stone-Soup/blob/v1.9.1/docs/examples/dataassociation/MFA_example.py).

Native `thresh-association/src/mht.rs` already exists. Its current
`prune_n_scan` compares surviving current assignments in a flat representation;
it is not an independent ancestry oracle. A follow-up must decide the intended
birth/clutter/miss and historical commitment contract, then test both
implementations against enumerated three-scan cases. Do not substitute PMHT,
single-step mixture reduction, or MFA's name alone for that decision.

### 2.3 IMM — advertised import unsupported

Both Python and initialized Rust reproduce
`ModuleNotFoundError: No module named 'stonesoup.predictor.interacting'`.
The Rust error is misleadingly classified `StoneSoupNotInstalled` even though
1.9.1 is installed. The inspected release's predictor/updater packages and tagged
file inventory do not supply the named `IMMPredictor` contract.
[Predictors](https://github.com/dstl/Stone-Soup/tree/v1.9.1/stonesoup/predictor).

`CompositePredictor` independently predicts sub-states; it does not implement
IMM interaction. Particle `MultiModelPredictor` and its Rao-Blackwellized variant
are not a drop-in Gaussian CV/CA/CT IMM oracle.
[Composite source](https://github.com/dstl/Stone-Soup/blob/v1.9.1/stonesoup/predictor/composite.py),
[particle predictors](https://github.com/dstl/Stone-Soup/blob/v1.9.1/stonesoup/predictor/particle.py).

Recommended decision: an independently authored, small NumPy IMM reference using
supported single-model predictors/updaters, or another explicitly evaluated
reference. This review implements neither and claims no working upstream IMM.
Require hand-derived two-model tests for `c_j = sum_i(mu_i p_ij)`, conditional
mixing weights, mixed covariance including mean-spread terms, each model's
prediction/update likelihood, normalized posterior probabilities, and final
mixture moments. Test identity/symmetric/asymmetric transition matrices,
zero-probability destinations, log-space underflow, and CV↔CA/common-state
projection with units and unobserved acceleration covariance explicit. The
bridge currently only advertises prediction; posterior adaptation needs an
explicit update contract before it can satisfy the live IMM scenario.

### 2.4 GM-PHD — split responsibilities, repairable composition

[`phd.rs`](../../crates/thresh-bridge/src/phd.rs) fails with unexpected `predictor`.
After constructing a valid updater, its two-argument `update(prior, detections)`
also fails: `update` accepts **one hypothesis collection**.

| Concern | Supported v1.9.1 owner / mapping |
|---|---|
| Dynamics / measurement likelihood | `KalmanPredictor` and `KalmanUpdater` (or a justified nonlinear variant) |
| Predict each component, gate candidate measurements | `DistanceHypothesiser` wrapped by `GaussianMixtureHypothesiser(order_by_detection=True)` |
| Birth intensity | Explicit `TaggedWeightedGaussianState` birth components, with timestamp, mean, covariance and weight; not a constructor probability shortcut |
| Intensity update | `PHDUpdater(updater, clutter_spatial_density=..., prob_detection=..., prob_survival=...)`; `update(hypotheses)` |
| Miss / survival | Hypothesis collection includes missed detections; surviving non-birth intensity and detection probability are separate |
| Prune / merge / truncate | `GaussianMixtureReducer(prune_threshold=..., merge_threshold=..., max_number_components=...)`, after intensity update |

The existing `prob_detect`/`clutter_intensity` names are wrong for `PHDUpdater`.
Its reducer's merge threshold is **squared** Mahalanobis distance, while the Rust
config comment merely says Mahalanobis: choose/document units, do not silently
square or preserve the number. Mixture weights estimate expected cardinality;
do not normalize their total to 1.
[Updater](https://github.com/dstl/Stone-Soup/blob/v1.9.1/stonesoup/updater/pointprocess.py),
[hypothesiser](https://github.com/dstl/Stone-Soup/blob/v1.9.1/stonesoup/hypothesiser/gaussianmixture.py),
[reducer](https://github.com/dstl/Stone-Soup/blob/v1.9.1/stonesoup/mixturereducer/gaussianmixture.py).

With one persistent component (weight 0.8), one birth (0.2), and the two invented
detections, the supported composition returned 3 hypothesis groups, 5 updated
components, intensity `2.079090797672432`, and 1 reduced component. This is a
one-step API check, not a validated cardinality/lifecycle benchmark.

### 2.5 Conversion, initialization, and errors

Actual initialized Rust probes plus
[`detection.rs`](../../crates/thresh-bridge/src/detection.rs),
[`convert.rs`](../../crates/thresh-bridge/src/convert.rs),
[`error.rs`](../../crates/thresh-bridge/src/error.rs), and
[`Measurement`](../../crates/thresh-core/src/measurement.rs) establish:

| Surface | Observed result | Required follow-up expectation |
|---|---|---|
| Radar | Column `[range, azimuth, elevation, optional range_rate]`; example `[100, .2, .1, -3]`; timestamp/model both `None` | Metres, radians, m/s; explicit sensor origin/frame and Doppler sign; reorder to the chosen upstream model, not a blind shared vector |
| EO/IR | Column `[azimuth, elevation]`; timestamp/model `None` | Match upstream bearing/elevation ordering, angle wrapping, extrinsics, and `R` |
| ADS-B | `[lat_deg, lon_deg, alt_m, optional vx, vy, vz]`; timestamp/model `None` | Explicit geodetic→Cartesian conversion and velocity-frame/altitude datum; rotate covariance consistently; no degree-as-metre identity model |
| Radar without Doppler / ADS-B without velocity | Source paths produce 3-D columns, versus 4-D / 6-D above | Distinct model shapes; optional fields must not become zero-valued measurements |
| Time | Stored only in metadata; Python `Detection(..., metadata={'time':1})` likewise has `timestamp=None` | Supply actual observation `datetime`; relative seconds need an explicit origin, not an assumed Unix epoch. Preserve separate arrival time. Reject invalid time/order as specified. |
| Noise / state covariance | Plain `Detection`, no model or covariance attachment; `Measurement::default_noise()` is not consulted | Detection measurement model carries `R`, or use `GaussianDetection` for state estimates; convert state `P` with declared ordering. Validate dimensions, finite values, symmetry/PSD. |
| OTHR / LiDAR / Camera | All three returned `ConversionError` with installed dependencies | Preserve explicit rejection until separately specified; do not silently accept a partially mapped sensor |
| Arrays | Matrix conversion uses explicit row lists; input matrix rejects 1-D. Vector conversion accepts and flattens 2-D; accepts NaN | Define vector shapes `(n,)`/`(n,1)` versus matrix `(m,n)`; reject accidental flattening/nonfinite data; test non-square and non-contiguous arrays, dtype/coercion, covariance symmetry |
| Fresh Rust process | Both ignored tests panic before initialization | Explicit interpreter ownership/initialization or deliberate feature configuration; library contract must not depend on another crate initializing Python |
| Package absent, initialized process | JPDA/MHT/PHD returned `StoneSoupNotInstalled`; IMM encountered missing NumPy first and returned `PythonNotAvailable` | Test errors independently and name the missing package accurately |
| Submodule absent in installed package | IMM reports `StoneSoupNotInstalled` | Distinguish unsupported API/version from missing top-level package; preserve original Python exception context |
| Unsupported sensor with packages absent | Import happens before enum rejection; observed missing-package error first | Decide and test error precedence; do not promise unconditional conversion errors today |
| Missing Python shared library | Not simulated | Build/link/loader failure can precede Rust error handling; specify supported embedding/deployment behavior and test it in an isolated process |

The missing-package control removed `PYTHONPATH` from the temporary initialized
Rust executable. Constructor errors above were observed before the scratch
probe's later NumPy `unwrap()` panicked; that final panic was in the **probe**,
not evidence of a bridge constructor panic after initialization. No claim is
made that missing-library startup degrades gracefully. The standard tests'
uninitialized-interpreter panic is independently reproduced.

## 3. Tests to implement in a separately approved change

Everything in this section is a **test specification**, not a claim those tests
were implemented or passed by this review. Reuse existing metrics and native
algorithms. Stable ordering, float64, explicit epochs, and fixed materialized
inputs are mandatory; identical RNG seeds in different libraries are insufficient.

### 3.1 Numerical matrix and independent checks

For scalar/vector comparisons use `abs(a-b) <= atol + rtol*abs(reference)`;
for matrices apply elementwise and report max residual. Primary conditioned
fixtures use `atol=1e-10, rtol=1e-9`; sigma-point/nonlinear fixtures start at
`atol=1e-8, rtol=1e-7`. Tolerances are prospective acceptance bounds, not measured
cross-language error bars. Fix causes before relaxing them, and record any
scale-dependent bound in the future spec.

| Case | Fully specified inputs / assertions |
|---|---|
| Hand-computable scalar KF | Prior `x=0,P=1`, `F=H=1`, `Q=.25,R=.5,z=1`: prediction `P=1.25`, innovation `1`, `S=1.75`, gain `5/7`, posterior `x=5/7,P=5/14`, NIS `4/7`; agree with closed form and Stone Soup |
| 1-D CV embedded in 3-D | `[x,vx,y,vy,z,vz]`, `x0=[0,1,10,-2,3,.5]`, `P0=diag(2,1,2,1,2,1)`, `dt={.1,1,2.3}`; `F` has three `[[1,dt],[0,1]]` blocks; common `Q` uses native discrete `sigma_a=.2`, i.e. three `.04*[[dt^4/4,dt^3/2],[dt^3/2,dt^2]]` blocks; `H` selects indices `[0,2,4]`, `R=diag(1,2,3)`, `z=H F x0+[.3,-.2,.1]`. Compare prediction/posterior mean, `P`, innovation, `S`, gain and likelihood |
| Linear EKF/UKF/CKF reduction | Same CV fixture must reduce to the analytic KF within tolerance. UKF fixes `alpha=.3,beta=2,kappa=0` initially, then separately tests thresh default `alpha=.001`; CKF uses spherical-radial `2n` points and matched Cholesky orientation |
| Nonlinear radar | Cartesian state `[1000,100,500,-50,300,5]`, diagonal `P=[25,4,25,4,25,4]`; explicit range/bearing/elevation/Doppler function, sensor at origin; `R=diag(100,1e-6,1e-6,1)`, perturbation `[3,.0002,-.0001,.1]`. Reorder for upstream model. Check analytic Jacobian against centered finite differences (`h_i=1e-5*max(1,abs(x_i))`, `atol=1e-6,rtol=1e-5`), wrapped angular residuals, and same-method parity; do not require EKF=UKF=CKF for nonlinear moments |
| Numerical robustness | `P=U diag(1e-8,1e-4,1,10,100,1e4) U^T`, with `U=diag(R(pi/6),R(pi/4),R(pi/3))` and `R(theta)=[[cos(theta),-sin(theta)],[sin(theta),cos(theta)]]`, plus deliberately singular/indefinite/nonfinite inputs. Require finite symmetric output, `max(abs(P-P^T)) <= 1e-10*max(1,norm(P,2))`, and minimum eigenvalue `>= -1e-10*max(1,norm(P,2))`; record jitter/repair and fail on unbounded repair. Match unsupported-input errors, not arbitrary repaired answers |
| Exact association | Enumerate all mutually exclusive joint events for 1–3 tracks and 0–3 detections; compare JPDA/EHM/EHM2 marginals to independent enumeration (`atol=1e-12,rtol=1e-10`). Include one shared measurement, all gated out, empty tracks, asymmetric likelihoods and ties. Every row including miss sums to 1 within `1e-12`; nonnegative probabilities and detection occupancy sums `<=1+1e-12` for exact exclusive marginals |
| Approximate association | Native `jpda_probabilities` normalizes per track, with miss `(1-P_D)*clutter`; it is not exact joint-event JPDA. First compare native to its independent formula/PDA under matched `P_G` convention. Compare against exact JPDA/LBP using marginal error, identity and runtime, **not an unjustified equality assertion** |
| Hungarian / MHT | Brute-force small assignment costs including misses, rectangular/all-gated matrices and ties; compare objective not arbitrary tie identity. For MHT, enumerate three-scan histories, prohibit detection reuse, verify cumulative weights and the exact scan being committed by pruning. MFA's approximate solver and native flat history must each be checked against that oracle |
| IMM | Independent two-model mixing/update cases in §2.3; model posterior and mixing columns sum to one within `1e-12`; verify between-model covariance and common-state mappings at .1 s, 1 s, irregular intervals and missed-update gaps. No Stone Soup `IMMPredictor` test is possible under this release |
| CI | Compare the information-form formula at matched weights and state/frame order; use equal covariances, complementary diagonals and highly correlated SPD inputs. Native CI searches a 0.01 grid clamped to [.01,.99], minimizing trace. Reproduce that grid before comparing outputs; Stone Soup's default omega=.5 or another objective is not the same algorithm setting |

Critical convention trap: native `models/cv.rs` uses
`sigma_a^2 * [[dt^4/4,dt^3/2],[dt^3/2,dt^2]]`; upstream `ConstantVelocity`
uses a continuous white-noise form proportional to
`[[dt^3/3,dt^2/2],[dt^2/2,dt]]`. Match explicit `Q` with
`LinearGaussianTimeInvariantTransitionModel`/a small reference model before
filter parity. Equal parameter names or equal dt do not make these covariances
equivalent. A production noise-model change is outside this review.

### 3.2 Synthetic sequence matrix

Materialize observation records `(observation_time, arrival_time, sensor_id,
measurement_id, vector, covariance, frame)` once per seed, with a separate
ground-truth table containing true IDs/states. Feed the **same record bytes** to
both implementations; the tracker never receives true associations. Persist only
invented fixtures. Use an explicit metric configuration (e.g. Cartesian metres,
GOSPA `p=2,alpha=2,c=25 m`, MOT gate `25 m` for this small-scene suite), not a
cross-regime global cutoff. Record confirmation/deletion and visibility policies.

| Sequence | Controls / required observations |
|---|---|
| Isolated CV / clean crossing | 1 then 2 targets, 120 s, separation crosses zero, zero clutter; assert stable IDs away from deliberately ambiguous crossing; inspect association marginals at crossing |
| Dense crossing / clutter | 2, 5, 10, 25 targets; clutter Poisson means 0, 2, 10, 30 per scan over fixed area, `P_D=.6,.9,1`; record actual density and gate occupancy; exact oracle only for tractable clusters |
| Births / deaths | Targets enter at 0/20/40 s and leave at 60/80/100 s; measure confirmation/deletion latency, false births, misses and cardinality bias, separately from localization |
| Missing-observation bursts | One target has deterministic 1/3/10 s gaps plus a separately seeded bursty dropout process; preserve prediction-only intervals and compare covariance growth/reacquisition/ID continuity |
| Maneuvers | Invented CV→turn→CA and correlated-acceleration truth, explicit transition times; record mode-probability lag and consistency without giving the tracker the true mode |
| Timing | Repeat physical 120 s at 1 Hz, 10 Hz, and repeating intervals [.05,.15,.3,.5,1] s; use the actual intervals for `F/Q`, history inputs, survival and lifecycle duration |
| Delayed arrival | Two sensors, known delay 5 s first; then varying 0/.2/2/5 s delays, reordering, duplicate and too-old observations. Keep observation time immutable; delayed-data policies reject, buffer or replay explicitly |

Report per-seed position RMSE, GOSPA with localization/miss/false decomposition,
MOTA, IDF1, ID switches, cardinality/latency, wall time per frame (median/p95),
peak hypothesis count and memory. Use at least seeds 0–29 for scheduled
comparisons, paired per-seed differences, and a 95% bootstrap interval over
**seeds**, not correlated frames. PR smoke uses fixed seeds 0–2 and small sizes;
performance budgets require a fixed runner/warmup and cannot be inferred here.

NEES uses the covariance marginal matching independently known truth; NIS uses
actual filter innovations and innovation covariance, not fabricated residuals.
Reuse `thresh-eval` consistency and GOSPA machinery, including deliberate
covariance-understatement/overstatement controls. For chi-squared consistency
tests use independent trajectories at fixed times (or documented subsampling);
do not assume every correlated frame is independent. Identify association-selected
populations and missing samples. RFS intensity-only outputs cannot receive IDF1
until an explicit identity-extraction policy exists. Offline RTS and fixed-lag
outputs get separate scoreboards with allowed look-ahead/latency; they cannot
beat a causal online release gate using future measurements unnoticed.

### 3.3 CI recommendation — no workflow edits in this review

1. **No-Python lane:** on an image without Python development/runtime packages,
   run `cargo test --locked --workspace --exclude thresh-py --no-default-features` and native
   filter/association/fusion/eval test targets with `RUSTFLAGS=-Dwarnings`.
   `thresh-py` unconditionally depends on PyO3, so a whole-workspace command
   without this exclusion is not a no-Python test; its binding tests belong in
   the existing separate Python lane. Keep optional Python features disabled.
   Verify `cargo tree -p thresh-bridge --no-default-features` has no PyO3 edge.
   Zero bridge tests is expected only here; assert native test counts are nonzero.
2. **Pinned optional-runtime lane, after repair:** pin Python 3.12.13 and an
   isolated venv using the ledger's direct/transitive versions and hashes;
   set `PYO3_PYTHON` before build and explicitly verify the embedded interpreter's
   `sys.path`, executable/prefix and package versions. Prefer a documented
   embedding initializer; never rely on accidental feature unification.
   Run `cargo test --locked -p thresh-bridge --features stonesoup --test
   stonesoup_integration -- --include-ignored --test-threads=1 --nocapture`.
   Until repaired, the existing two failures are expected review evidence,
   **not** a green merge gate or an `allow_failure` replacement for one.
3. **Count/coverage contract:** future tests should be non-ignored in this
   dedicated target. Maintain a reviewed list of required exact test names:
   interpreter/import, array roundtrip, JPDA step, PHD step, detection timing/model,
   unsupported sensors, shape/nonfinite rejection, missing-package subprocess,
   plus MHT-history and IMM-contract tests once their reference decisions are
   approved. Compare `-- --list` to that manifest; run each named case with
   `--exact --include-ignored`, require exactly 1 executed and 0 ignored/skipped,
   and fail for missing dependencies, empty output, panic, failed assertion or
   missing required case. The current two-test manifest would still fail;
   new assertions must not be replaced with import-only success.
4. **Numerical/sequence lane:** run the separately added parity target with
   `stonesoup` enabled and the named assertions in §3.1; a Python fixture producer
   must fail on missing packages rather than `importorskip`. Require fixture
   checksums, versions, expected frame counts, and seeds 0–2 for PR checks.
   Scheduled seeds 0–29/dense scenes report accuracy and runtime together.
   Add `ortools==9.14.6206` only to the explicitly selected MFA lane; no optional
   solver import can silently skip its tests.
5. **Newer-version watch lane:** scheduled, separately labeled compatibility
   against a resolved newer release/commit, recording its SHA and constraints.
   Re-run boundary/numerical tests, keep the pinned required lane unchanged, and
   report failures instead of automatically upgrading or relaxing tolerances.
   The listed unreleased numerical/measurement changes justify this lane.

## 4. Candidate dispositions

These are recommendations for reference testing/investigation, not approved
production additions. All symbols below were checked against v1.9.1.

| Candidate | Disposition and bounded evidence / next experiment |
|---|---|
| `JPDAwithEHM`, `JPDAwithEHM2` | **Adopt for testing after bridge repair.** In the 2×2 probe their marginals agree with enumerating `JPDA` to <1e-15. They use the shipped Python `_ehm` implementation; `pyehm` was not installed or required for these classes. Require independent tiny-event enumeration, then 2/5/10/25-target dense-scene runtime/hypothesis counts. This improves reference coverage of existing native association, not a reason to relabel its per-track approximation exact. [Source](https://github.com/dstl/Stone-Soup/blob/v1.9.1/stonesoup/dataassociator/probability.py) |
| `JPDAwithLBP` | **Investigate, not adopt as an exact oracle.** Base dependencies sufficed. In the same tiny probe max absolute marginal difference from exact JPDA is about **0.117368**. Check normalization/feasibility and finite convergence, measure marginal error, ID metrics and runtime over sparse/dense/clutter sweeps; preregister acceptable quality/runtime tradeoffs before any native BP/SPA proposal. No performance improvement was measured here. [Source](https://github.com/dstl/Stone-Soup/blob/v1.9.1/stonesoup/dataassociator/probability.py) |
| `Singer` / `SingerApproximate` | **Adopt full Singer as a test reference; investigate production only after noise-contract agreement.** Existing CV/CA/CT models do not make correlated-acceleration reference tests redundant. Use `A=[[0,1,0],[0,0,1],[0,0,-alpha]]`, `G=[0,0,1]^T`, `F=expm(A dt)`, `Q=integral exp(A t) G q G^T exp(A^T t) dt`. In the actual implementation `noise_diff_coeff=q` multiplies covariance, so distinguish it from a standard deviation. Test `dt={.01,.1,1,10}`, `alpha={.01,.1,1,10}`, `q={.01,.2,2}` against independent quadrature/Van Loan (`atol=1e-9,rtol=1e-7`), symmetry/PSD and semigroup composition. Full/approximate covariance norm differences at `q=.2,alpha=.1`, dt .1/1/10 were `.000199171`, `.0240257`, `435.651`; the approximation is not interchangeable at long dt. No Rust Singer was added. [Source](https://github.com/dstl/Stone-Soup/blob/v1.9.1/stonesoup/models/transition/linear.py) |
| `KalmanSmoother` (RTS) | **Adopt for offline test design; investigate fixed-lag implementation separately.** It consumes prediction/update history. Proposed scalar/linear assertions: final smoothed state equals final filtered state; backward recursion agrees with a hand-computed two-step result; `P_filtered-P_smoothed` is PSD within tolerance under matched linear assumptions. Native state-history/cross-covariance plumbing and latency policy need a separate change. No smoother runtime/numerical probe was run here. [Source](https://github.com/dstl/Stone-Soup/blob/v1.9.1/stonesoup/smoother/kalman.py) |
| OOSM example | **Adopt known-delay scenarios; investigate general delayed-data handling.** The cited example assumes a known constant delay (5 s), uses two sensors and compares ignoring timing, lagging, and dropping delayed measurements. It is not a general arbitrary-delay retrodiction algorithm. Our fixture must keep observation/arrival times distinct even though the example offsets one sensor's detection timestamps. General reorder/replay/too-old/duplicate policies require native tracker work and separate causal/fixed-lag metrics. Source inspected, example not executed. [Example](https://github.com/dstl/Stone-Soup/blob/v1.9.1/docs/examples/oosm/KalmanFilterOOSMExample.py) |
| `ChernoffUpdater` / covariance intersection | **Adopt for parity testing**, not a duplicate fusion algorithm. Native `thresh-fusion/src/{covariance_intersection,t2t}.rs` already covers unknown-correlation fusion. Match means, covariance, weight orientation, and optimization domain; verify the direct information formula and conservative consistency under deliberately shared sensor noise. The gap is independent numerical coverage, not missing CI. Import/source contract assessed; full numerical parity not run. [Source](https://github.com/dstl/Stone-Soup/blob/v1.9.1/stonesoup/updater/chernoff.py) |
| `LCCUpdater` (GM-LCC) | **Defer.** Cardinality variance under non-Poisson clutter/births is a plausible future gap, but no such comparative failure was measured here. First repair and benchmark GM-PHD; LCC adds cumulant/clutter-variance state and calibration/maintenance beyond that baseline. Do not infer stable IDs from an intensity filter. [Source](https://github.com/dstl/Stone-Soup/blob/v1.9.1/stonesoup/updater/pointprocess.py) |
| `VisibilityInformedBernoulliParticleUpdater` | **Defer production; investigate only if the burst/occlusion matrix demonstrates a gap.** It estimates existence for **one target**, requires particles/resampling and compatible sensor visibility, and is not a multi-target tracker. Existing native lifecycle/missed-detection handling and synthetic sensor visibility provide the baseline to challenge first. [Source](https://github.com/dstl/Stone-Soup/blob/v1.9.1/stonesoup/updater/particle.py) |
| Broader PMBM/GLMB/JIPDA/RFS | **Defer pending a single architecture decision and measured cardinality/existence gap.** Reconcile with July's JIPDA-versus-PMBM decision, native JPDA/MHT/IMM and the broken optional PHD surface. Birth models, global hypotheses, pruning, identities, solver/runtime and cross-language upkeep are real costs. No candidate is selected or ruled equivalent by this review. |

## 5. Prioritized follow-up scopes (approval required)

| Order | Scope / affected code and specs | Dependencies and measurable acceptance |
|---|---|---|
| P0-A | **Bridge boundary repair:** `thresh-bridge/src/{lib,convert,detection,error}.rs`, integration tests; `openspec/specs/stonesoup-bridge/spec.md` plus measurement/time capabilities if contracts change | Decide interpreter ownership, relative-time origin, supported frames/model/noise conversion and error precedence. Fresh-process calls must not panic; supported conversions retain actual time and `R`; unsupported sensors and malformed arrays have exact tested errors; absent package versus absent submodule distinguished. Both existing smoke tests execute, not skip. |
| P0-B | **Executable JPDA and GM-PHD reference graphs:** `src/{jpda,phd}.rs`, explicit reference fixtures; `stonesoup-bridge` | Depends on P0-A. Constructor and one-step tests with supported components; JPDA all-missed/normalized marginals; PHD birth/survival/detection/miss and intensity checks; reducer settings map with units. These repairs do not certify whole-track accuracy. |
| P0-C | **Resolve MHT/IMM reference claims:** `src/{mht,imm}.rs`; `stonesoup-bridge`, and `mht-tracking`/IMM specs only if behavior is deliberately changed | Independently evaluate MFA history versus required sliding-window semantics and select an IMM oracle. Tiny exhaustive three-scan and hand-derived mixing/update tests are acceptance prerequisites. Either implement the promised contract or explicitly approve a capability amendment; never silently delete/rename requirements to fit a wrapper. |
| P1-A | **Deterministic parity + pinned runtime CI:** bridge tests, native filter/association/fusion tests, isolated dependency constraints and `.github/workflows/ci.yml`; `state-estimation`, `cubature-kalman-filter`, `jpda-association`, `mht-tracking`, `sensor-fusion`, `stonesoup-bridge` | Depends on boundary and relevant reference decisions. Implement §3.1 cases with measured residuals, independent oracles, invariants and exact executed-test manifests. No dependency skip or zero-test success in the optional lane. Native/no-Python lane remains green. |
| P1-B | **Synthetic sequence comparisons:** `thresh-synth`, `thresh-eval`, tracker/reference harness; `synthetic-data`, `evaluation-metrics`, `filter-consistency-metrics`, `gospa-metric`, `consistency-benchmark-gates` | Depends on deterministic correctness. Materialized equal observations and independent truth; complete §3.2 frame/seed counts, paired metrics/uncertainty and causal labels; deliberate bad-covariance/identity/time controls must fail. Reuse existing metrics; no provider captures. |
| P2-A | **Association scaling study, possible BP follow-up** | Depends on P1 results. Compare native approximation, JPDA/EHM/EHM2/LBP on small correctness and dense runtime/quality. Advance production work only with an agreed improvement threshold and bounded maintenance cost. |
| P2-B | **Singer/process-noise study**: filter model references then, only if approved, `MotionModel`/IMM composition | Agree continuous/discrete `Q` units before matching numbers. Pass transition/covariance grid and consistency checks; show maneuver benefit without no-regression failure. Do not change the learned model bank or checkpoint contract incidentally. |
| P2-C | **RTS/fixed-lag/OOSM design**: filter history and tracker temporal policy | Depends on timing fixtures and explicit look-ahead limits. Offline backward recursion checks and latency-bounded online replay/rejection tests must be separate; order invariance only where the chosen algorithm guarantees it. |
| Deferred | **LCC, visibility Bernoulli, broader RFS** | Require a measured failure in the cardinality/occlusion matrix and one approved architecture decision; no parallel speculative replacements. |

No follow-up OpenSpec changes are opened by this report. Maintainers should first
agree on P0-A/P0-B and the MHT/IMM decisions; the test and optional-algorithm
scopes can then be proposed separately. The completed review is not automatically
archived, and none of its findings authorizes release or relaxes model/data rights.

### Verification and handoff

The report's Python code blocks were extracted and executed together with the
isolated interpreter; all recorded constructor failures and successful probe
values reproduced. A separate root-agent review checked every report section
against the 16 OpenSpec tasks and caught two proposal details before completion:
exclude the unconditional `thresh-py` package from the no-Python lane, and fully
specify the CV fixture's noise coefficient. Both are corrected above. The
no-Python dependency-tree check (`cargo tree --locked --workspace --exclude
thresh-py --no-default-features`) contained no PyO3 package.

Strict validation passed for this change, all 52 live specifications/active
changes, and all 26 archives; `git diff --check` was clean. These documentation
checks do not override the two failing bridge runtime tests. The review is ready
for maintainer agreement on follow-up scope; no repair, adoption, release, or
archive is implied by completing its review checklist.

## Appendix A. Compact Python reproductions and observed outputs

Run with the isolated interpreter. Inputs are invented, fixed and float64.
These snippets are API probes, deliberately not a full correctness test suite.

```python
from datetime import datetime, timedelta, timezone
from importlib import import_module
import numpy as np
from stonesoup.models.transition.linear import ConstantVelocity
from stonesoup.models.measurement.linear import LinearGaussian
from stonesoup.predictor.kalman import KalmanPredictor
from stonesoup.updater.kalman import KalmanUpdater
from stonesoup.hypothesiser.probability import PDAHypothesiser
from stonesoup.dataassociator.probability import JPDA, JPDAwithEHM, JPDAwithEHM2, JPDAwithLBP
from stonesoup.tracker.simple import MultiTargetMixtureTracker
from stonesoup.updater.pointprocess import PHDUpdater
from stonesoup.types.state import GaussianState, TaggedWeightedGaussianState
from stonesoup.types.track import Track
from stonesoup.types.detection import Detection

def probe(label, call):
    try:
        print(label, call())
    except Exception as error:
        print(label, type(error).__name__, str(error))

predictor = KalmanPredictor(ConstantVelocity(.1))
model = LinearGaussian(2, [0], np.array([[1.]]))
updater = KalmanUpdater(model)
probe('JPDA', lambda: PDAHypothesiser(gate_probability=.99, clutter_spatial_density=1e-6))
probe('gate', lambda: PDAHypothesiser(predictor, updater, gate_probability=.99))
probe('MHT', lambda: MultiTargetMixtureTracker(max_num_hypotheses=100, prune_threshold=-10.))
probe('IMM', lambda: import_module('stonesoup.predictor.interacting'))
probe('PHD', lambda: PHDUpdater(updater=updater, predictor=predictor,
    prob_survival=.99, prob_detect=.9, clutter_intensity=1e-5,
    prune_threshold=1e-5, merge_threshold=4., max_components=100))
phd = PHDUpdater(updater, prob_survival=.99, prob_detection=.9, clutter_spatial_density=1e-5)
probe('PHD step', lambda: phd.update([], set()))
print('track method?', hasattr(MultiTargetMixtureTracker, 'track'))
probe('MFA import', lambda: import_module('stonesoup.dataassociator.mfa').MFADataAssociator)

t0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
t1 = t0 + timedelta(seconds=1)
tracks = [Track([GaussianState([[x], [0.]], np.diag([1., 1.]), timestamp=t0)], id=str(i))
          for i, x in enumerate((0., 1.))]
detections = [Detection([[x]], timestamp=t1, measurement_model=model, metadata={'id': i})
              for i, x in enumerate((.2, 1.2))]
hypothesiser = PDAHypothesiser(predictor, updater, clutter_spatial_density=.1,
                              prob_detect=.9, prob_gate=.99)
for cls in (JPDA, JPDAwithEHM, JPDAwithEHM2, JPDAwithLBP):
    result = cls(hypothesiser).associate(set(tracks), set(detections), t1)
    print(cls.__name__, {tr.id: sorted((str(h.measurement.metadata['id']) if h else 'miss',
                        float(h.probability)) for h in result[tr]) for tr in tracks})
print('all missed', [float(h.probability) for h in
      JPDA(hypothesiser).associate({tracks[0]}, set(), t1)[tracks[0]]])
det = Detection([[100.], [.2], [.1]], metadata={'time': 1.})
print('timestamp/model', det.timestamp, det.measurement_model)

from stonesoup.hypothesiser.distance import DistanceHypothesiser
from stonesoup.hypothesiser.gaussianmixture import GaussianMixtureHypothesiser
from stonesoup.measures import Mahalanobis
from stonesoup.mixturereducer.gaussianmixture import GaussianMixtureReducer
gm_hyp = GaussianMixtureHypothesiser(
    DistanceHypothesiser(predictor, updater, Mahalanobis(), missed_distance=10),
    order_by_detection=True)
component = TaggedWeightedGaussianState([[0.], [0.]], np.diag([1., 1.]),
                                       timestamp=t0, weight=.8, tag='track')
birth = TaggedWeightedGaussianState([[1.], [0.]], np.diag([2., 2.]),
                                   timestamp=t1, weight=.2, tag='birth')
hypotheses = gm_hyp.hypothesise({component, birth}, set(detections), t1)
update = phd.update(hypotheses)
intensity = float(sum(c.weight for c in update.components))  # before reduction mutates components
reduced = GaussianMixtureReducer(prune_threshold=1e-5, merge_threshold=4.,
                                max_number_components=100).reduce(update.components)
print('PHD', len(hypotheses), len(update.components), intensity, len(reduced))
```

Observed exceptions are listed in §2. Association probabilities, columns ordered
as detection 0 / detection 1 / miss (last digits may vary with summation order):

| Method | Track 0 | Track 1 |
|---|---|---|
| JPDA / EHM / EHM2 | `.548866652 / .396796966 / .054336382` | `.399900894 / .548866652 / .051232455` |
| LBP | `.604441948 / .279429011 / .116129041` | `.286068697 / .604436051 / .109495252` |

MFA continuation, after the optional dependency installation:

```python
from ordered_set import OrderedSet
from stonesoup.dataassociator.mfa import MFADataAssociator
from stonesoup.hypothesiser.mfa import MFAHypothesiser
from stonesoup.types.mixture import GaussianMixture
from stonesoup.types.update import GaussianMixtureUpdate
mfa = MFADataAssociator(MFAHypothesiser(hypothesiser), slide_window=3)
mfa_tracks = OrderedSet(Track([GaussianMixture([
    TaggedWeightedGaussianState([[x], [v]], np.diag([1., 1.]),
                                timestamp=t0, weight=1., tag=[])])], id=str(i))
    for i, (x, v) in enumerate(((0., 1.), (4., -1.))))
for step in range(1, 5):
    stamp = t0 + timedelta(seconds=step)
    scan = OrderedSet(Detection([[x]], timestamp=stamp, measurement_model=model)
                      for x in (float(step), 4. - step))
    for track, hypotheses in mfa.associate(mfa_tracks, scan, stamp).items():
        components = [updater.update(h) if h else h.prediction for h in hypotheses]
        track.append(GaussianMixtureUpdate(components=components, hypothesis=hypotheses))
    print(step, [(len(tr.state.components), sorted({len(c.tag) for c in tr.state.components}),
                  len({c.tag[0] for c in tr.state.components})) for tr in mfa_tracks])
```

For both tracks, output `(component count, tag lengths, first-tag choices)` is
`(3,[1],3)`, `(9,[2],3)`, `(9,[3],1)`, `(9,[4],1)` across the four scans.

Singer covariance continuation:

```python
from stonesoup.models.transition.linear import Singer, SingerApproximate
for dt in (.1, 1., 10.):
    full = Singer(noise_diff_coeff=.2, damping_coeff=.1)
    approx = SingerApproximate(noise_diff_coeff=.2, damping_coeff=.1)
    q = full.covar(time_interval=timedelta(seconds=dt))
    qa = approx.covar(time_interval=timedelta(seconds=dt))
    print(dt, np.linalg.eigvalsh(q).min(), np.linalg.norm(q-qa))
```

Minimum eigenvalues were `2.77084098e-9`, `.000220249256`, `.173674718`.
This confirms those three returned covariances are positive definite; it does
not replace the proposed independent transition/covariance grid.

## Appendix B. Initialized Rust probe recipe

In a temporary crate outside the repo, depend on `pyo3 = "=0.29.0"`,
`nalgebra = "0.33"`, and the reviewed `thresh-core` and `thresh-bridge` paths
(enable `thresh-bridge`'s `stonesoup` feature). The essential constructor probe is:

```rust
use pyo3::prelude::*;
use pyo3::types::PyList;
use thresh_bridge::{imm, jpda, mht, phd};

fn main() {
    Python::initialize();
    println!("JPDA: {:?}", jpda::JpdaAssociator::new(&Default::default()).map(|_| ()));
    println!("MHT: {:?}", mht::MhtTracker::new(&Default::default()).map(|_| ()));
    Python::attach(|py| {
        println!("IMM: {:?}", imm::ImmFilter::new(py, &Default::default(),
                 &PyList::empty(py)).map(|_| ()));
        let none = py.None();
        println!("PHD: {:?}", phd::PhdFilter::new(py, Default::default(),
                 none.bind(py), none.bind(py)).map(|_| ()));
    });
}
```

The PHD constructor rejects the unsupported keyword before inspecting the dummy
predictor/updater; the valid-component Python probe independently confirms the
same mismatch. For conversion reproduction, call `measurement_to_detection` in
that initialized closure on the measurement values in §2.5 and inspect
`state_vector`, `timestamp`, and `measurement_model`. Call `numpy_to_dvector`
with `np.array([[1.,2.],[3.,4.]])` and `np.array([NaN])`, and
`numpy_to_dmatrix` with `np.array([1.,2.])`; the current flatten/accept/reject
outcomes are as recorded above. These scratch programs are not installed as
project tests and must not be mistaken for the missing CI coverage.
