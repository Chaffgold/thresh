## ADDED Requirements

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
