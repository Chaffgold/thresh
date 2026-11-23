# Design — Stone Soup Bridge Boundary

## Context

See `proposal.md` for motivation and scope. This plan is grounded in `develop`
`4bdcbdbde80df4ca27b91c2d1e1c23e032f6038c`, the completed September reference
review, and bounded source inspection on 2026-11-21. Graph tools were unavailable;
no graph generation or index-coverage claim is made.

- `thresh-bridge` gates all modules and optional PyO3 0.29 behind `stonesoup`.
  Token-free wrapper calls use `Python::attach`; the two integration tests are
  ignored and do not initialize Python. No bridge-runtime lane exists in CI.
- `detection::measurement_to_detection(py, &Measurement)` accepts five shapes
  across Radar, EoIr, and AdsB, but supplies only values and metadata. It has no
  Rust callers in the inspected repository; there is no batch converter.
- `convert.rs` flattens arbitrary vector inputs and does not validate dtype or
  finiteness. `error.rs` reattaches while formatting errors and classifies imports
  by substring, confusing missing submodules with an absent package.
- Core `Measurement` has relative seconds and no noise/frame attachment. Radar
  and EO/IR use sensor-local spherical coordinates. ADS-B has degrees, MSL height,
  and an untagged velocity; its default noise uses metric position variances.
- `SensorRegistration` supplies pose but no frame identity. Core `Epoch` uses
  physical TAI arithmetic and nanosecond UTC rendering; its Julian-date float
  accessors are too coarse for this boundary. Core geodetic helpers use radians
  and ellipsoid height and already support WGS84/ECEF/local ENU conversion.

OpenSpec 1.13.2 reports a healthy planning root. Before this change, all 52 live
items and 26 archives pass strict local validation. GitHub PRs #162/#163 fail
existing OpenSpec checks while installing `@latest`; the CI runs do not establish
their validator version. This discrepancy is separate from bridge implementation.

## Goals / Non-Goals

**Goals:** Make boundary success mean that time, values, frame, covariance, and
errors have defined, independently tested semantics. Keep attachment, validation,
normalization, Python-object construction, and result checking as small phases.

**Non-Goals:** Algorithm constructor/stepping repair; general frame transforms;
moving sensor dynamics; automatic geoid acquisition; raw-provider acquisition;
changes to core measurement semantics or the training pipeline's ADS-B-as-truth
rule; model acceptance/release; a floating upstream upgrade; workspace-wide
OpenSpec cleanup. This plan does not certify all existing bridge requirements.

## Decisions

### 1. Host owns the interpreter; bridge entry is fallible

Standalone Rust hosts explicitly call `Python::initialize()` during startup;
Python-hosted callers reuse their interpreter. Do not enable `auto-initialize`
as the fix. A private `with_python` helper uses `Python::try_attach` and returns
`PythonNotAvailable` when attachment is unavailable. Token-taking helpers reuse
the supplied token. Apply the shared entry/error plumbing to all four wrapper
modules without changing their algorithm graphs or declaring those graphs working.

Lifecycle verification separates standalone calls from calls requiring existing
Python objects. Test standalone constructors, the measurement converter, and the
shared attachment helper in fresh processes before initialization. IMM/PHD
construction requires a live Python token, and wrapper stepping requires objects
created by Python; exercise those calls under an explicitly initialized host.
Do not fabricate invalid Python objects or finalize the host to force an otherwise
unreachable startup case. Synthetic dispatch doubles can verify wrapper plumbing
without claiming that the upstream algorithm graphs work.

Classify Python errors while a token is already held. Replace reattaching error
conversion with an explicit token-aware mapper and structured cause information.
The library never finalizes Python, changes `sys.path`, or changes environment
variables. Hosts must keep Python alive while bridge objects exist; concurrent
host finalization, subinterpreters, and free-threaded builds are not certified by
the initial CPython 3.12 lane. PyO3 documents attachment during shutdown as
best-effort; do not claim a stronger concurrent-shutdown guarantee.

**Alternative rejected:** automatic initialization makes library behavior depend
on global feature unification and obscures ownership in Python-hosted applications.
[Pinned PyO3 lifecycle source](https://github.com/PyO3/pyo3/blob/v0.29.0/src/marker.rs).

### 2. Require typed Rust observation context and break the incomplete API

Replace the converter with a token-free entry taking `&Measurement` and
`&DetectionContext`; an internal token-taking phase constructs Python objects.
The context is a Rust type, not an opaque caller-supplied Python model. It holds:

- An absolute `Epoch` origin and optional separate arrival epoch.
- A fixed WGS84 ENU origin with latitude/longitude and **ellipsoid** height; common
  state order is `[x,vx,y,vy,z,vz]`, with metres and metres/second.
- A typed observation profile and explicit covariance; no implicit call to
  `Measurement::default_noise()`.
- Radar/EOIR `SensorRegistration`, matching sensor identity, and fixed geometry;
  radar-with-Doppler additionally declares positive-away or positive-towards.
- ADS-B MSL-to-ellipsoid geoid separation, optional-velocity frame, and metric
  noise convention as defined below.

Validate required fields, profile/measurement agreement, finite values, source
ranges, dimensions, and a proper orthogonal rotation before Python access. Radar
range is nonnegative; elevation is within [-pi/2,pi/2]; normalize finite azimuth
to [-pi,pi). ADS-B latitude/longitude are within [-90,90]/[-180,180]. Reject
inertial/unspecified frames and moving-platform profiles. Fixed sensor velocity
is zero because the chosen profile is explicitly stationary, not because an
absent measured Doppler value is being filled.

Finite inputs do not guarantee finite derived values. Check datum addition,
coordinate/velocity normalization, covariance transforms, and constructed outputs
for overflow/nonfinite results; reject them instead of passing infinities to Python.

No legacy success shim: missing context must not reproduce incomplete detections.
Document the public signature/error changes and update all examples. Native
`Measurement::to_vector`, default covariance, and native tracking remain unchanged.

**Alternative rejected:** a caller-supplied arbitrary Python model cannot enforce
the relationship between declared frames, values, and covariance. Silent identity
or Unix-epoch defaults conceal the original defects.

### 3. Preserve physical time and make datetime limitations explicit

Prevalidate origin, finite offsets, and representable bounds before arithmetic
that could overflow. Compute observation time with core `Epoch`, not Julian-day
floats. Convert its UTC calendar representation to timezone-aware Python datetime,
rounding integer nanoseconds to microseconds with ties-to-even and checked carry.
The supported calendar range is years 1–9999. Preserve source seconds, absolute
origin, and optional arrival epoch in metadata; conversion is stateless and does
not sort, enforce arrival order, or add artificial increments.

Python datetime cannot represent UTC leap seconds. Reject a leap-second instant
or an origin-to-observation interval whose UTC calendar elapsed time differs from
physical elapsed time by a leap discontinuity, even when both endpoints are
ordinary calendar instants. Test positive and negative crossings. Otherwise an
upstream timestamp subtraction would silently lose a physical second. Retain
full core time precision/semantics elsewhere; this is a boundary limitation.

Validate origin and observation with a lossless UTC render/parse identity check
or equivalent leap-aware detection before microsecond rounding. Core's calendar
renderer can fold a leap instant into an ordinary second; checking only a printed
second of 60 or elapsed-time differences is insufficient. Include a leap instant
as origin with zero and fractional offsets in rejection fixtures.

**Alternative rejected:** interpreting relative seconds as Unix time or truncating
to whole seconds changes observations; pretending Python datetime is leap-aware
corrupts downstream propagation intervals.

### 4. Construct five explicit measurement-model profiles

Use `Detection` plus an attached model carrying `R`; do not treat measurement
noise as track covariance. All output vectors are float64 `(n,1)`. Preserve raw
source metadata and tag ADS-B as a cooperative report, not a newly synthesized
sensor return or a change to its training-data role.

| Profile | Output observation order | Stone Soup 1.9.1 model / mapping |
|---|---|---|
| Radar3 | elevation, azimuth, range | `CartesianToElevationBearingRange`; position indices (0,2,4) |
| Radar4 | elevation, azimuth, range, range-rate-away | `CartesianToElevationBearingRangeRate`; position (0,2,4), velocity (1,3,5) |
| EOIR2 | elevation, azimuth | `CartesianToElevationBearing`; position (0,2,4) |
| ADSB3 | x,y,z in common ENU | `LinearGaussian`; mapping (0,2,4) |
| ADSB6 | x,y,z,vx,vy,vz in common ENU | `LinearGaussian`; mapping (0,2,4,1,3,5) |

All models use `ndim_state=6`. Source radar noise order is range/azimuth/elevation
plus optional range rate; EOIR noise order is azimuth/elevation. Apply the same
permutation and Doppler sign transform `A` to observation noise: `R_out=A R A^T`,
including off-diagonal terms. Encode the source sensor-to-common rotation and
position in the model's inverse observation transform. Verify this with an
independent hand-computed target, including a 90-degree rotation, rather than
assuming upstream Euler-angle conventions match the Rust matrix convention.
Document the chosen matrix-to-upstream-rotation convention in the adapter tests.

For ADS-B, set ellipsoid height to MSL height plus caller-supplied geoid separation.
Use existing WGS84-to-ECEF-to-ENU helpers for position. Accept velocity in either
ECEF or ENU at the report location, with a required tag when velocity exists.
Source position-error covariance is metric ENU at the report location, not a
covariance over latitude/longitude degrees. For ADSB6, noise order is position
ENU followed by velocity in its declared frame. Form block transforms into the
common ENU frame and compute the full `T R T^T`, retaining position/velocity cross
terms. Caller-supplied noise must include applicable datum/registration uncertainty;
the bridge does not manufacture uncertainty estimates or fetch a geoid model.

Absent optional velocity or Doppler selects the smaller model, never a zero-valued
observation. Model dimensions, mapping, covariance, and actual `timestamp` are
postconditions, not metadata-only claims.

**Alternative rejected:** directly applying metric default noise to a degree-valued
ADS-B vector is dimensionally invalid. Requiring an explicit noise convention
allows normalization without altering existing core defaults.
[Pinned models](https://github.com/dstl/Stone-Soup/blob/v1.9.1/stonesoup/models/measurement/nonlinear.py),
[linear model](https://github.com/dstl/Stone-Soup/blob/v1.9.1/stonesoup/models/measurement/linear.py),
[detection type](https://github.com/dstl/Stone-Soup/blob/v1.9.1/stonesoup/types/detection.py).

### 5. Check arrays and covariance before materializing them

Generic converters accept only nonempty float64 arrays: vectors `(n,)` or `(n,1)`,
matrices `(m,n)`. Preserve logical C-order values even for strided inputs; reject
row vectors `(1,n)` for n>1, arbitrary flattening, unsupported dtypes, nonfinite
entries, and inconsistent shape/element counts. Check size multiplication before
allocations/construction. Outbound nalgebra vectors remain 1-D in the generic
utility; the detection builder explicitly constructs column StateVectors.

Observation covariance is square at the selected measurement dimension. Use
`tol = 1e-12 * max(1, max_row_sum(abs(R)))` for symmetry and minimum-eigenvalue
validation, and document this scale-aware rule in tests. Accept PSD matrices
within tolerance without modifying them; reject violations. Never add jitter or
silently symmetrize. Singular noise may still be unsuitable for a later algorithm;
passing boundary checks is not a claim that an updater can invert every PSD matrix.

Compute norm/tolerance and eigenvalue checks using overflow-safe scaling. If a
finite validation bound cannot be obtained, reject the covariance; never let an
infinite tolerance make an indefinite matrix pass. Validate transformed covariance
again and reject nonfinite intermediate/results rather than repairing them.

**Alternative rejected:** permissive flattening/coercion hides order and type
mistakes. Restricting float64 matches the Rust API and reviewed reference baseline.

### 6. Stable diagnostic categories and precedence

For the high-level converter, order checks as: unsupported enum variant, Rust
context/input validation, interpreter availability, dependency availability,
selected model API, then Python execution. Keep OTHR/LiDAR/Camera explicit errors.
Tests block imports in fresh children to prove that unsupported input still wins.

Keep `PythonNotAvailable`, `StoneSoupNotInstalled`, `ConversionError`, and ordinary
Python/algorithm errors. Add structured missing-dependency and unsupported-API
diagnostics as needed; this is part of the documented error-API migration. Inspect
`ModuleNotFoundError.name` and exception type while attached. Only a missing
top-level `stonesoup` is `StoneSoupNotInstalled`; missing NumPy/transitive packages
name that package. An absent submodule or symbol in an installed package is an
API/version mismatch. Preserve exception type, message, and operation context;
retain traceback where available. Ordinary TypeError/ValueError is not a package
diagnostic even if its message contains a package name.

**Alternative rejected:** substring classification is observably wrong for the
current absent IMM submodule and can misclassify arbitrary exception text.

### 7. Pinned runtime lane with a test manifest, not import-only coverage

Use isolated CPython 3.12.13, Stone Soup 1.9.1, NumPy 2.5.3, SciPy 1.18.1 and the
workspace-locked PyO3 baseline. Commit dedicated fully resolved Python constraints
under `crates/thresh-bridge/tests/`, separate from the training environment/lock.
No OR-Tools is needed for boundary work. Record versions and select the same
interpreter for PyO3 build and runtime; configure its site-packages explicitly in
the CI process where embedding requires it, never inside the library.

Remove `#[ignore]` from the two existing integration tests and initialize their
host explicitly. Add boundary tests plus a process-contract harness. Launch a
fresh child for uninitialized runtime and each missing-package/submodule case so
module caches and other tests cannot mask a failure. Use test-local import hooks;
do not uninstall packages or alter a user's environment.

The CI job first lists tests for the explicit integration targets, compares names
with a checked manifest, then executes them and asserts the expected passed count
with zero ignored/failed required tests. A manifest self-test must fail for an
empty selection, missing name, ignored test, and failing child. The manifest covers
the two smoke tests plus all five profiles, pose/covariance, time edge cases,
array rejection, context validation, and every error category. Ordinary
`cargo test` exit zero alone is insufficient.

Run this lane on PRs targeting `develop`, without path-skipping its required tests.
Native isolation separately runs `cargo tree --workspace --exclude thresh-py
--no-default-features` and rejects any PyO3 dependency, then tests that workspace
configuration with Python discovery disabled. `thresh-py` is excluded only from
this no-Python lane because its dependency is unconditional, not from existing
Python-binding CI. Keep existing lint, benchmark, and OpenSpec gates unchanged.

**Alternative rejected:** checking only imports, ignoring missing dependencies,
or using floating versions reproduces the review's false-confidence problem.

## Risks / Trade-offs

- **Link/load precedes Rust:** a missing linked Python shared library may prevent
  startup before any bridge error can be returned. Document/test supported
  deployment prerequisites. The existing broad missing-Python requirement remains
  a known gap; this plan neither silently narrows it nor adds dynamic loading.
- **Narrow frame profile:** fixed local ENU is deliberate. Unsupported frames or
  moving sensors fail explicitly; broadening them requires another approved change.
- **Datum/noise burden:** callers must supply information absent from core types.
  Typed constructors, independent geometry cases, and clear migration docs make
  the cost visible rather than silently assuming units.
- **Optional tests cost CI time:** use a separate cached environment and explicit
  targets; do not reduce required assertions to keep the job green.
- **OpenSpec CI drift:** local 1.13.2 success is not proof that floating GitHub
  validation will pass. Reconcile validator version/metadata compatibility in a
  separate approved tooling change; never disable strict validation as a workaround.

## Migration Plan

1. Implement context/errors and tests before replacing the public converter.
2. Route wrapper entry/error plumbing through the checked boundary; do not repair
   or certify their algorithm APIs incidentally.
3. Document explicit host initialization and all five context examples, plus the
   conversion/error source break and known shared-library deployment limitation.
4. Enable the pinned runtime and native-isolation checks; publish their executed
   test manifest with the implementation PR's validation evidence.
5. Roll back a faulty boundary implementation by reverting its code/CI commit,
   not by changing fixtures, tolerances, feature defaults, or algorithm claims.
   Keep this plan and the original review as records of requirements and gaps.

No production implementation, dependency installation, CI modification, spec
synchronization, archive, or release occurs while authoring these artifacts.
