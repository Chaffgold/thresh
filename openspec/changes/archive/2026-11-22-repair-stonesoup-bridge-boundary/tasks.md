# Tasks

## 1. Runtime ownership and diagnostic foundation

- [x] 1.1 Add a dedicated, fully resolved boundary-test environment under `crates/thresh-bridge/tests/` for CPython 3.12.13, Stone Soup 1.9.1, NumPy 2.5.3, and SciPy 1.18.1 without changing the training lock; verify a clean isolated install and record all resolved versions alongside the locked PyO3 version.
- [x] 1.2 Implement fallible host-owned interpreter attachment without enabling auto-initialize; verify fresh-process tests return `PythonNotAvailable` before initialization and reuse an explicitly initialized host without changing its lifetime, path, or environment.
- [x] 1.3 Replace reattaching/substr-matching error conversion with token-aware structured diagnostics; verify isolated import-hook tests distinguish missing top-level Stone Soup, NumPy/transitive dependency, installed-package submodule/symbol, and ordinary TypeError/ValueError while preserving available exception details.
- [x] 1.4 Route JPDA, MHT, IMM, and PHD entry/error plumbing through the shared runtime helpers without changing algorithm graphs; verify non-panicking fresh-process failures for standalone token-free constructors and the shared attachment helper, exercise calls requiring Python-owned objects under an explicitly initialized host, and review the diff for constructor/stepping changes outside that plumbing.
- [x] 1.5 Document explicit standalone-host initialization, Python-host reuse, structured error migration, interpreter lifetime, and linked-library deployment prerequisites; verify the initialization example runs and documentation does not claim missing shared libraries can always become Rust errors or that algorithm wrappers are repaired.

## 2. Typed observation context and physical time

- [x] 2.1 Introduce typed detection context for absolute origin, optional arrival epoch, fixed WGS84 local ENU origin, interleaved six-state mapping, observation profile, and explicit noise; verify constructor/validation tests reject absent or conflicting required context without Python access.
- [x] 2.2 Validate sensor identity, stationary pose, proper rotation, source ranges, optional-observation/profile agreement, Doppler convention, and ADS-B datum/velocity/noise tags; verify synthetic invalid-context tests cover unsupported frames/motion, nonfinite values, and missing metadata without inventing defaults.
- [x] 2.3 Implement checked physical-time conversion using core Epoch and integer calendar precision, with nearest-microsecond ties-to-even rounding and leap-aware UTC identity validation; verify positive/negative offsets, out-of-order calls, delayed arrival metadata, rounding ties/carry/collisions, year boundaries, nonfinite/overflow rejection, leap instants as origin with zero/fractional offsets, and positive/negative leap-crossing intervals.
- [x] 2.4 Document context units, fixed-frame restrictions, covariance component order, timestamp representability, and preserved source/arrival metadata; verify examples instantiate valid context for all five observation profiles and invalid-context examples return the documented error category.

## 3. Strict numeric and covariance conversion

- [x] 3.1 Harden nalgebra-to-NumPy conversion to produce nonempty finite float64 arrays with checked dimensions and explicit logical order; verify vector and nonsquare-matrix tests compare every element and preserve the generic utility's one-dimensional vector output.
- [x] 3.2 Harden NumPy-to-nalgebra conversion for float64 vectors `(n,)`/`(n,1)` and two-dimensional matrices; verify strided round trips and rejection tests for row vectors, arbitrary flattening, empty/malformed shapes, unsupported ranks/dtypes, nonfinite values, and checked size overflow.
- [x] 3.3 Implement observation covariance validation at the declared dimension with the design's overflow-safe scale-aware symmetry/eigenvalue tolerance; verify valid correlated/PSD cases remain unchanged and invalid/extreme finite cases fail without infinite bounds, jitter, or symmetrization, and document both tolerance and downstream singular-noise limitations in the conversion API.

## 4. Complete measurement detections

- [x] 4.1 Replace the incomplete converter with the token-free measurement/context API and phased Rust validation, attachment, construction, and postcondition checks; verify unsupported OTHR/LiDAR/Camera and invalid supported input take precedence over unavailable runtime/packages, and successful detections carry actual UTC timestamps and preserved provenance metadata.
- [x] 4.2 Implement Radar3 and Radar4 column observations with matching six-state nonlinear models, fixed sensor pose, Doppler sign, and full covariance permutation/sign transforms; verify independent synthetic geometry with nonzero translation/rotation, absent-Doppler dimension, both sign conventions, and unequal correlated noise.
- [x] 4.3 Implement EOIR2 elevation/bearing observations and the attached pose-aware model; verify the specified 90-degree-z/nonzero-translation case against independently transformed target geometry and verify complete covariance reordering, documenting the matrix-to-upstream-rotation convention in adapter tests.
- [x] 4.4 Implement ADSB3 WGS84/MSL-to-ellipsoid-to-common-ENU conversion using explicit geoid separation and metric source noise; verify the invented zero-latitude/longitude height case, a nonzero-origin case, complete covariance rotation, rejection of overflowing datum/position/noise results from finite inputs, cooperative-report provenance, and absence of fabricated velocity.
- [x] 4.5 Implement ADSB6 position/velocity conversion for explicitly tagged ECEF or report-local ENU velocity and full block covariance transforms; verify both velocity conventions at nontrivial coordinates, transformed cross-covariances, rejection of nonfinite derived velocity/noise, output order, and model mapping `(0,2,4,1,3,5)` against independent expected values.
- [x] 4.6 Update bridge API examples and migration documentation for the breaking converter/error changes; verify executable examples cover all five profiles and assert float64 column shape, timestamp, model dimension, state mapping, and covariance, while leaving core/native measurement semantics and algorithm-access claims unchanged.

## 5. Required runtime and native-isolation checks

- [x] 5.1 Remove ignoring from the existing import and nalgebra/NumPy integration tests and explicitly initialize their host; add a checked manifest covering these and the boundary/process tests as they now exist, then verify every listed required test is discovered and executed with zero ignored or failed tests.
- [x] 5.2 Add the manifest runner and its failure-mode self-tests; verify empty selection, missing/renamed test, ignored required test, failed child, and missing dependency all fail the gate, and document the exact reproducible local runtime command and expected test counts.
- [x] 5.3 Add a PR-to-develop runtime CI lane using the isolated pinned environment and matching PyO3 build/runtime interpreter, without required-test path skipping; verify a clean CI-equivalent run prints versions and passes the manifest, and document CI-only site-packages setup without moving environment mutation into the library.
- [x] 5.4 Add the separate no-Python workspace dependency/build/test check excluding only the unconditional `thresh-py` package; verify `cargo tree --workspace --exclude thresh-py --no-default-features` contains no PyO3 and native tests pass with Python discovery disabled, while existing Python-binding and other CI gates remain intact.

## 6. Cross-cutting acceptance

- [x] 6.1 Run workspace formatting, warning-strict build/tests, Clippy, and rustdoc plus the feature-enabled bridge runtime manifest and native-isolation lane; attach results to the implementation review and verify every new requirement/scenario maps to an executed assertion rather than an ignored or import-only check.
- [x] 6.2 Run strict OpenSpec validation for live and archived artifacts plus the repository's metadata checks, review the final scope diff, and record remaining algorithm/deployment gaps; verify no provider data, algorithm graph repair, dependency upgrade, training/model release, crates.io publication, review archive, or unrelated validator-workflow repair was included.

## Approved acceptance clarification

On 2026-11-22, the user approved separating standalone startup tests from calls
requiring Python-owned objects. IMM/PHD construction requires an existing Python
token, and wrapper step methods require already-created Python objects. Task 1.4
therefore tests standalone constructors and the shared attachment helper before
initialization, and Python-owned calls under an explicitly initialized host.
This corrects the test plan without relaxing fallible attachment or host ownership,
changing algorithm graphs, or claiming algorithm acceptance.
