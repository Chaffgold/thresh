# Design — Cubature Kalman Filter

## Context

`thresh-filter` ships KF, EKF, and UKF today. UKF is the workhorse for non-linear motion models (CTRV with sensor-domain measurement updates, ballistic re-entry, etc.). The UKF's sigma points use Van der Merwe's scaling parameters `(α, β, κ)`; in some corner cases the central sigma point can carry negative weight, and under sharp maneuvers the predicted covariance can lose positive-definiteness.

CKF (Arasaratnam & Haykin, 2009) is a structurally simpler nonlinear Kalman filter with the same asymptotic accuracy as UKF at the third order, no tuning parameters, and provably positive-definite covariance updates. The two filters share the same API contract (predict-with-`MotionModel`, update with a closure or linear `H`); a user choosing between them is making a robustness-vs-tuning-flexibility tradeoff, not a fundamental architectural choice.

## Goals / Non-Goals

**Goals:**

- Ship a `CubatureKalmanFilter` whose public API matches `UnscentedKalmanFilter` line-for-line so users can swap them.
- Wire it into the existing filter-kind selector exposed by `thresh-tracker` (whatever it is — TBD in Decision 4 after a quick survey) and into the IMM filter bank as a permitted leaf.
- Identical test coverage to UKF: linear-converges, nonlinear-reduces-uncertainty, covariance symmetry/PD invariants.

**Non-Goals:**

- Higher-order cubature rules (5th, 7th-order spherical-radial integration). The third-order rule is what production radar tracking uses; higher orders are an academic refinement.
- Square-root CKF (SR-CKF). Numerically robust on limited-precision hardware; not needed for thresh's host-CPU targets.
- Benchmarking CKF vs. UKF on existing thresh scenarios. A separate evaluation change.
- A `CkfParams` struct. CKF is parameter-free by design — adding a knob would defeat the point.

## Decisions

### 1. CKF lives next to UKF, not under it

**Decision:** New `crates/thresh-filter/src/ckf.rs`, parallel to `ukf.rs`. The two share traits (`MotionModel`) but not implementation.

**Rationale:** UKF's sigma-point generation has tuning parameters; CKF's doesn't. Trying to share a generic sigma-point generator would either (a) require dependency injection of weight schemes, complicating both APIs, or (b) limit each to a lowest-common-denominator. Side-by-side modules keep each readable.

### 2. No `CkfParams` struct

**Decision:** Constructor is `CubatureKalmanFilter::new(x, p)`. No tuning parameters.

**Rationale:** Third-order spherical-radial cubature is uniquely determined by the state dimension. Adding a params struct would be cargo-culting from UKF and would mislead callers into thinking there are knobs.

### 3. Sigma-point implementation: `±√n · e_i` from a Cholesky factor

**Decision:** Compute `S = chol(P)`, then form `2n` cubature points `χ_i = x ± √n · S_{:,i}` for `i ∈ [0, n)`. All points carry weight `1/(2n)`.

**Rationale:** This is the textbook third-order spherical-radial rule. Cholesky is already used in `ukf.rs` via nalgebra; reuse the same primitive. The integer `√n` factor lives inside a single `f64::sqrt(n as f64)` and is fine numerically.

### 4. Filter-kind wiring

**Decision:** Survey `thresh-tracker` for the existing filter-kind selector (probably an enum like `FilterKind { Kf, Ekf, Ukf }`) during Task 3 implementation, then add a `Ckf` variant. If no such enum exists, add nothing tracker-side and document the CKF as a `thresh-filter`-level offering only. The IMM filter bank in `thresh-filter::imm` is a separate wiring point and gets the same treatment.

**Rationale:** Need to read code before committing to a tracker-side change. The IMM wiring is well-defined (the `StateMapping` trait must be implementable for CKF state); the tracker-side wiring depends on what's already there.

### 5. Test parity with UKF

**Decision:** Mirror `ukf::tests` one-for-one. Specifically: linear-system convergence, nonlinear bearings-only example with covariance reduction over multiple updates, and a random-perturbation invariant test that asserts the covariance stays symmetric and positive-definite after `predict` and `update`.

**Rationale:** The CKF and UKF should be empirically interchangeable on linear and mildly-nonlinear problems. Side-by-side tests pin that behaviour and catch regressions if either filter drifts.

### 6. Documentation includes the Arasaratnam-Haykin reference

**Decision:** The module-level docstring on `ckf.rs` cites Arasaratnam & Haykin 2009 ("Cubature Kalman Filters", IEEE Transactions on Automatic Control, vol. 54, no. 6) and points to `docs/reference/` for the worked derivation if one is added.

**Rationale:** This is standard for the existing filter modules and helps anyone debugging cubature rules later.

## Risks

- **IMM-bank wiring.** The IMM filter expects each leaf to map to/from a common state representation via `StateMapping`. CKF's state representation is identical to UKF's (a flat mean + covariance), so the existing `StateMapping` impl on `UnscentedKalmanFilter` should be replicable verbatim — but worth verifying during Task 3.
- **Cholesky failure on near-degenerate P.** If the covariance becomes near-singular (e.g. after a zero-noise predict on a stationary target), Cholesky can fail. The UKF handles this the same way; CKF will use the same fallback (return an error, propagated up).
- **Performance.** CKF is 2n cubature points; UKF is 2n+1 sigma points. The compute difference is negligible (one extra column of state propagation), but worth a `criterion` benchmark eventually — captured as a follow-up, not in scope here.

## Open Questions

- Does `thresh-tracker` expose a `FilterKind` enum that needs the `Ckf` variant, or does it dispatch on a different abstraction? Resolved during Task 3 implementation.
- Should `CubatureKalmanFilter::new` accept an optional `process_noise_floor` parameter to clamp the covariance away from singularity? Likely yes if UKF has the same, but check first.
- Worth comparing CKF + IMM against UKF + IMM on the `thresh-eval` ADS-B scenario to confirm no regression — separate evaluation PR after this lands.
