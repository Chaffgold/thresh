# stonesoup-bridge Specification

## Purpose
Optional Rust-to-Python bridge, built on PyO3, for instantiating and calling Stone Soup tracking components from Rust, including GIL management and conversion between Rust (nalgebra) types and Stone Soup's numpy/State objects. It exposes Stone Soup's JPDA, MHT, IMM, and Gaussian Mixture PHD algorithms as reference implementations and operational fallbacks, and sits behind the `stonesoup` Cargo feature so that the Rust-native tracking algorithms compile and run without Python or Stone Soup installed.

## Requirements
### Requirement: PyO3 bridge to Stone Soup
The system SHALL provide a Rust-to-Python bridge via PyO3 that can instantiate and call Stone Soup tracking components (filters, data associators, hypothesisers, trackers) from Rust code. The bridge SHALL handle Python GIL management and type conversion between Rust types and Stone Soup's numpy/State objects.

#### Scenario: Instantiate a Stone Soup JPDA tracker
- **WHEN** Rust code requests a JPDA data associator from Stone Soup
- **THEN** the bridge SHALL create a `stonesoup.dataassociator.probability.JPDA` instance with the specified parameters and return a handle usable from Rust

#### Scenario: Pass measurements to Stone Soup
- **WHEN** Rust provides a set of detections as nalgebra vectors
- **THEN** the bridge SHALL convert them to Stone Soup `Detection` objects with appropriate `StateVector` and `CovarianceMatrix` and pass them to the Stone Soup component

### Requirement: Stone Soup algorithm access
The system SHALL expose the following Stone Soup algorithms via the bridge: Joint Probabilistic Data Association (JPDA), Multi-Hypothesis Tracking (MHT), Interacting Multiple Model (IMM) filter, and Gaussian Mixture PHD filter. These serve as reference implementations and operational fallbacks until Rust-native versions exist.

#### Scenario: IMM filter for maneuvering target
- **WHEN** a target transitions between constant velocity and coordinated turn maneuvers
- **THEN** the IMM filter accessed via Stone Soup SHALL maintain multiple model hypotheses and correctly adapt the blended state estimate as the target maneuvers

#### Scenario: MHT for ambiguous associations
- **WHEN** multiple detections fall within the gates of multiple tracks creating association ambiguity
- **THEN** the MHT accessed via Stone Soup SHALL maintain a hypothesis tree and defer hard association decisions over a configurable sliding window

### Requirement: Graceful degradation without Python
The system SHALL compile and run without Python/Stone Soup installed, with the bridge module disabled via a Cargo feature flag (`stonesoup`). Core tracking functionality (KF, EKF, UKF, Hungarian) SHALL work without the Python dependency.

#### Scenario: Build without stonesoup feature
- **WHEN** the crate is built with `--no-default-features` or without the `stonesoup` feature
- **THEN** the crate SHALL compile successfully and all Rust-native tracking algorithms SHALL function normally

#### Scenario: Runtime error on missing Python
- **WHEN** the `stonesoup` feature is enabled but Python or Stone Soup is not installed at runtime
- **THEN** the system SHALL return a clear error message indicating the missing dependency rather than panicking

### Requirement: Explicit host-owned interpreter lifecycle
The bridge SHALL require host initialization of Python, reuse an initialized interpreter, and neither initialize nor finalize it implicitly. A bridge entry without an available interpreter SHALL return an interpreter-unavailable error rather than panic.

#### Scenario: Fresh process without host initialization
- **WHEN** an otherwise valid bridge call is made in a fresh process before Python initialization
- **THEN** it returns an interpreter-unavailable error without importing packages or panicking

#### Scenario: Explicitly initialized host
- **WHEN** the host initializes Python and makes repeated bridge calls
- **THEN** calls use that interpreter without changing its lifetime, Python path, or environment

### Requirement: Required observation context
Measurement conversion SHALL require an explicit time origin, common coordinate frame, source-specific geometry, and observation covariance. Missing or inconsistent context SHALL be rejected; the bridge SHALL NOT invent frame, noise, datum, or optional observation values.

#### Scenario: Missing or conflicting context
- **WHEN** time origin, frame origin, required sensor pose, source covariance, or required ADS-B metadata is absent, or the sensor identity conflicts with the pose
- **THEN** conversion returns a context/conversion error before Python imports

#### Scenario: Supported common frame
- **WHEN** a valid context declares a fixed WGS84 local east-north-up frame, a six-component interleaved position/velocity state, and a supported observation profile
- **THEN** the output measurement model maps observations to that declared frame and state order

#### Scenario: Unsupported frame or sensor motion
- **WHEN** a context requests an inertial frame, an unspecified Cartesian frame, or moving sensor geometry
- **THEN** conversion explicitly rejects the unsupported context instead of assuming a fixed local frame

#### Scenario: Finite inputs with overflowing derived values
- **WHEN** finite source values produce nonfinite datum, coordinate, velocity, or covariance results during normalization
- **THEN** conversion rejects the result before constructing the detection

### Requirement: Absolute observation timestamps
Each converted detection SHALL carry a timezone-aware UTC observation timestamp derived from an explicit absolute origin and physical elapsed seconds. The bridge SHALL preserve original relative time and any supplied arrival time separately in metadata, without substituting either for observation time.

#### Scenario: Positive and negative relative offsets
- **WHEN** the origin is 2020-01-01T00:00:00Z and relative times are 0, 1.25, and -0.5 seconds
- **THEN** timestamps are respectively 00:00:00Z, 00:00:01.250000Z, and 2019-12-31T23:59:59.500000Z, regardless of call order

#### Scenario: Delayed arrival
- **WHEN** an observation arrives after a newer observation and includes a distinct arrival time
- **THEN** conversion retains its original observation timestamp and separately records arrival time without sorting, retiming, or rejecting it solely for being late

### Requirement: Explicit timestamp representability
Timestamp conversion SHALL round to the nearest microsecond with ties to even and retain the unrounded source time. It SHALL reject nonfinite offsets, overflow, dates outside years 1 through 9999, leap-second instants, and origin-to-observation intervals crossing a UTC leap-second discontinuity.

#### Scenario: Sub-microsecond observations
- **WHEN** two valid instants round to the same microsecond
- **THEN** both receive that timestamp, with their distinct original relative times retained; no artificial time increment is added

#### Scenario: Unrepresentable time
- **WHEN** the offset is NaN, infinite, out of range, or the physical elapsed interval crosses a leap-second insertion
- **THEN** conversion returns a time/conversion error rather than panicking, saturating, or silently changing the interval

#### Scenario: Leap instant disguised by calendar rendering
- **WHEN** the absolute origin is a leap-second instant with zero or fractional relative offset, even if its calendar rendering resembles an ordinary second
- **THEN** conversion rejects the unrepresentable origin rather than relying solely on a printed second of 60 or an elapsed-time difference

### Requirement: Radar and EOIR geometry fidelity
Radar and EO/IR conversion SHALL retain the core azimuth-from-positive-x and elevation-from-horizontal conventions. It SHALL apply the declared fixed sensor pose and explicit Doppler sign convention, reorder values and covariance together, and attach the matching observation model.

#### Scenario: Radar without Doppler
- **WHEN** radar values are range 100 metres, azimuth 0.2 radians, and elevation 0.1 radians with no range rate
- **THEN** the output column is [0.1, 0.2, 100] with a three-dimensional measurement model; no zero-Doppler observation is invented

#### Scenario: Radar with Doppler
- **WHEN** the same radar observation includes range rate -3 metres/second declared positive-away
- **THEN** the output column is [0.1, 0.2, 100, -3] with a four-dimensional model and the correspondingly permuted covariance

#### Scenario: Doppler sign reversal
- **WHEN** a radar source declares positive-towards range rate
- **THEN** the bridge reverses its sign and the corresponding covariance cross terms to match the model's positive-away convention

#### Scenario: EOIR bearings and nonzero pose
- **WHEN** EO/IR azimuth/elevation are [0.2, 0.1] and the declared sensor has a nonzero translation and a 90-degree rotation about z
- **THEN** the output is [0.1, 0.2], and the attached model's noiseless prediction agrees with the independently transformed target geometry

### Requirement: ADSB datum and Cartesian conversion
ADS-B conversion SHALL require an explicit MSL-to-ellipsoid height correction, a declared velocity frame when velocity exists, and metric observation covariance with declared component order. It SHALL convert position, velocity, and covariance consistently into the declared local frame, retaining cooperative-report provenance.

#### Scenario: Position-only cooperative report
- **WHEN** an invented ADS-B report at latitude/longitude zero has MSL height 100 metres, geoid separation 20 metres, and a local-frame origin at latitude/longitude zero with ellipsoid height zero
- **THEN** the output position is [0, 0, 120] metres with a three-dimensional model and no fabricated velocity

#### Scenario: Report with velocity
- **WHEN** a report includes velocity in either ECEF axes or local ENU axes at the reported position, with that choice explicit
- **THEN** the output is [x, y, z, vx, vy, vz] in the common ENU frame and its six-dimensional covariance includes transformed cross-covariances

#### Scenario: Ambiguous cooperative report
- **WHEN** height correction or velocity frame is missing, or degree-valued geodetic covariance is supplied where metric local covariance is required
- **THEN** conversion fails explicitly without reinterpreting degrees as metres or applying an undeclared default covariance

### Requirement: Observation model and covariance attachment
Every successful detection SHALL contain a float64 column vector, an actual observation timestamp, and an attached model with matching observation dimension and transformed covariance. Model state mapping, units, pose, and covariance order SHALL agree with the declared context.

#### Scenario: Correlated observation noise
- **WHEN** a radar or EO/IR covariance has unequal variances and nonzero off-diagonal entries
- **THEN** the model covariance equals the independently calculated permutation/sign transform of the complete matrix, not just a reordered diagonal

#### Scenario: Invalid observation covariance
- **WHEN** covariance has wrong dimensions, nonfinite values, asymmetry beyond tolerance, or a negative eigenvalue beyond tolerance
- **THEN** conversion rejects it without adding jitter, silently symmetrizing, or changing its units

#### Scenario: Extreme finite covariance entries
- **WHEN** finite covariance entries overflow an unscaled row sum or transformation
- **THEN** validation uses a finite overflow-safe bound or rejects the covariance, never accepting it through an infinite tolerance

### Requirement: Strict array conversion
Array conversion SHALL accept nonempty float64 vectors shaped (n,) or (n,1) and nonempty two-dimensional float64 matrices. It SHALL preserve logical element order for non-contiguous arrays and reject other dtypes, nonfinite values, unsupported ranks, and malformed shapes before constructing Rust matrices.

#### Scenario: Non-contiguous and nonsquare inputs
- **WHEN** a valid float64 strided vector or nonsquare matrix is converted in either direction
- **THEN** dimensions and all logical values are preserved without a row/column-major transpose

#### Scenario: Accidental flattening or coercion
- **WHEN** a vector input has shape (1,n) with n greater than one, is a general matrix, or uses integer, float32, complex, boolean, or object dtype
- **THEN** it is rejected rather than flattened or silently coerced

### Requirement: Deterministic boundary error precedence
For token-free measurement conversion, the bridge SHALL validate unsupported variants and Rust-side context/input before interpreter access, then check runtime and dependencies before invoking the selected model. Errors SHALL identify the failing stage and retain available original exception details.

#### Scenario: Unsupported sensor with unavailable dependencies
- **WHEN** an OTHR, LiDAR, or Camera measurement is supplied with missing Python packages or before host initialization
- **THEN** the same explicit unsupported-measurement error is returned without an import attempt

#### Scenario: Invalid input with unavailable runtime
- **WHEN** a supported measurement has invalid context or numeric values and the interpreter is unavailable
- **THEN** the input/context error takes precedence over the interpreter error

### Requirement: Precise Python dependency diagnostics
The bridge SHALL distinguish an absent top-level Stone Soup package, another missing dependency, an unavailable API within an installed package, and an unrelated Python exception. It SHALL classify using exception type and missing-module identity, not substring matching of display text.

#### Scenario: Package absent
- **WHEN** a valid initialized call cannot import the top-level Stone Soup package
- **THEN** it reports that package as missing

#### Scenario: Missing API or transitive dependency
- **WHEN** Stone Soup is present but a requested submodule/symbol is absent, or a required dependency such as NumPy cannot be imported
- **THEN** it reports respectively unavailable upstream API or the actual missing dependency, not a missing interpreter or missing top-level Stone Soup

#### Scenario: Ordinary upstream failure
- **WHEN** a Python value/type error occurs, even if its message mentions a package name
- **THEN** the original exception type and message remain available without reclassification as a missing package

### Requirement: Executed synthetic boundary verification
The required bridge-runtime CI lane SHALL install an exact isolated dependency baseline and execute an explicit manifest of boundary tests using wholly synthetic inputs. Missing packages, missing test names, ignored required tests, zero executed tests, or failed assertions SHALL fail the lane.

#### Scenario: Runtime coverage cannot silently disappear
- **WHEN** a required import, round-trip, lifecycle, geometry, timestamp, covariance, or diagnostic test is ignored, renamed without a manifest update, or not selected
- **THEN** the runtime lane fails rather than reporting successful bridge coverage

#### Scenario: No-Python isolation remains intact
- **WHEN** the native workspace is checked without the optional bridge feature, excluding the unconditional Python-binding package
- **THEN** its dependency tree contains no PyO3 and its native build/tests do not require Python or Stone Soup

#### Scenario: Boundary success is not algorithm acceptance
- **WHEN** all boundary tests pass
- **THEN** the result is reported only as boundary conformance; unresolved JPDA, GM-PHD, MHT, IMM, and deployment requirements are not marked complete by those results
