## Capability: Cubature Kalman Filter

### Overview

A third-order spherical-radial cubature Kalman filter for nonlinear state estimation. Sits alongside `UnscentedKalmanFilter` in `thresh-filter`; usable wherever the latter is. Differs from UKF in three concrete ways: parameter-free (no `α`/`β`/`κ`), uses `2n` equally-weighted points instead of UKF's `2n+1` weighted points, and guarantees positive-definite covariance updates under the standard third-order rule.

## ADDED Requirements

### Requirement: CKF struct and constructor

`thresh-filter` MUST expose a public `CubatureKalmanFilter` type with the same construction surface as `UnscentedKalmanFilter` minus the tuning-parameters struct.

#### Scenario: Constructing a CKF

**WHEN** a developer calls `CubatureKalmanFilter::new(x, p)` with an `n`-vector mean `x` and an `n × n` covariance matrix `p`

**THEN** the constructor returns a fully initialised filter whose state is `(x, p)` and whose subsequent `predict` / `update` calls advance the standard third-order cubature-Kalman recursion

**SHALL** require no other parameters. There is no `CkfParams` struct (CKF is parameter-free).

#### Scenario: Constructor rejects mismatched dimensions

**WHEN** the constructor is called with mismatched `x` and `p` dimensions

**THEN** the constructor panics with a clear message OR returns a `Result` per whatever convention `UnscentedKalmanFilter::new` uses

**SHALL** match `UnscentedKalmanFilter::new`'s error-handling convention for consistency.

### Requirement: Cubature-point generation

The CKF MUST generate `2n` cubature points from its current `(x, p)` state using the third-order spherical-radial rule: form `S = chol(P)`, then emit points `x ± √n · S_{:,i}` for each column `i` of `S`. All points carry weight `1/(2n)`.

#### Scenario: Cubature points reproduce mean and covariance

**WHEN** the filter has state `(x, p)` of dimension `n`, and `2n` cubature points are generated

**THEN** the weighted mean of the points equals `x` and the weighted sample covariance of the points equals `p` within numerical tolerance (default: `1e-9` per element on a 6D test)

**SHALL** be enforced by a unit test that constructs a random non-degenerate `(x, p)` and round-trips it through cubature-point generation.

### Requirement: Predict step

The CKF's predict step MUST propagate each cubature point through the supplied `MotionModel`, recompute the mean and covariance with equal weights, and add the model's process-noise covariance.

#### Scenario: Predict over an identity motion model preserves state

**WHEN** the filter is constructed with arbitrary `(x, p)` and `predict` is called with an identity motion model and zero process noise

**THEN** the post-predict `(x, p)` equals the pre-predict `(x, p)` within numerical tolerance

**SHALL** verify that no spurious drift is introduced by the cubature-point round-trip.

#### Scenario: Predict adds process noise correctly

**WHEN** the predict step is called with a non-zero process-noise matrix `Q` from the motion model

**THEN** the post-predict covariance is `cov(propagated points) + Q`

**SHALL** preserve positive-definiteness of `P` provided `Q` is positive-semidefinite.

### Requirement: Update step

The CKF MUST expose both a closure-based nonlinear update and a closed-form linear update, matching `UnscentedKalmanFilter`'s API.

#### Scenario: Linear update converges to truth

**WHEN** the filter is initialised with a coarse prior, given a linear measurement function `h`, and stepped with repeated noisy measurements drawn from the true state

**THEN** the estimated state converges to the truth and the trace of the covariance decreases monotonically over the measurement sequence

**SHALL** be tested at `n = 4` over at least 20 measurements with a fixed seed for reproducibility.

#### Scenario: Nonlinear bearings-only update reduces uncertainty

**WHEN** the filter is given a bearings-only measurement function (atan2 of state position components) and stepped with multiple measurements at different sensor positions

**THEN** the position covariance shrinks over the measurement sequence

**SHALL** mirror the equivalent test in `ukf::tests` so the two filters can be compared.

### Requirement: Covariance invariants under perturbation

After every `predict` and `update` call, the CKF's covariance matrix MUST remain symmetric (within `||P - P^T|| < 1e-9`) and positive-definite (smallest eigenvalue strictly greater than zero).

#### Scenario: Random predict/update sequence preserves invariants

**WHEN** the filter is driven with a deterministically-seeded sequence of random `predict` and `update` calls for at least 100 iterations

**THEN** the symmetry and positive-definiteness checks above pass after every step

**SHALL** be a property test asserting both invariants in a single loop.

### Requirement: Parity with UKF on linear problems

CKF and UKF MUST produce posterior estimates that agree within tight tolerances on linear-Gaussian problems (where both filters are theoretically equivalent to the standard Kalman filter).

#### Scenario: CKF and UKF agree on a linear-Gaussian benchmark

**WHEN** both filters are initialised with the same `(x, p)` and stepped with the same linear motion model, the same measurement matrix `H`, and the same noisy measurement sequence

**THEN** the posterior means agree within `1e-6` per element and the posterior covariances agree within `1e-4` per element

**SHALL** be a shared test fixture that exercises both filters in parallel for at least 10 measurement steps.

### Requirement: IMM filter-bank integration

The CKF MUST be usable as a leaf filter inside the IMM filter bank in `thresh-filter::imm`. This requires a `StateMapping` implementation whose `to_common` / `from_common` semantics match those of the existing UKF leaf.

#### Scenario: CKF leaf in an IMM bank

**WHEN** an IMM filter bank is configured with a CKF leaf and stepped through `predict` and `update` over a synthetic trajectory

**THEN** the bank produces mode probabilities that sum to one and a combined state estimate that tracks the truth within reasonable tolerance

**SHALL** match the behaviour of the equivalent UKF-leaf IMM bank to within statistical equivalence over a fixed-seed run.

### Requirement: Filter-kind selector (conditional)

If `thresh-tracker` exposes a `FilterKind`-style enum or selector for choosing between KF / EKF / UKF, that enum MUST be extended with a `Ckf` variant.

#### Scenario: Selecting CKF at the tracker level

**WHEN** a tracker is configured with `FilterKind::Ckf` (or equivalent selector value)

**THEN** the tracker instantiates a `CubatureKalmanFilter` for each new track and uses it for all subsequent predict/update calls

**SHALL** be a no-op requirement if `thresh-tracker` does not expose a filter-kind selector — see design.md Decision 4 and Open Questions.

### Requirement: No new external dependencies

The CKF implementation MUST NOT add any new external crates to `thresh-filter`'s `Cargo.toml`. `nalgebra` provides the Cholesky decomposition and matrix arithmetic needed.

#### Scenario: Cargo lockfile remains unchanged for thresh-filter

**WHEN** `cargo build -p thresh-filter` is run after the CKF implementation lands

**THEN** the diff to `Cargo.lock` is limited to thresh-filter's own version bump (if any), with no new third-party entries

**SHALL** be verified by inspecting the PR diff before merge.
