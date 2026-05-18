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
- **CKF as an IMM-bank leaf.** Deferred to a separate change (`imm-pluggable-leaf-filter`) — see Decision 7. The IMM bank hardcodes an EKF leaf and has no pluggable-filter trait, so this is a real `imm.rs` refactor, not the verbatim copy the original design assumed.

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

### 7. IMM-leaf integration deferred to a separate change (post-survey amendment)

**Decision:** Phases 3 (IMM integration) is **descoped from this change**. CKF ships as a `thresh-filter`-level filter (Phases 1, 2, 5, 6). A new change, `imm-pluggable-leaf-filter`, will introduce the abstraction that lets CKF (and UKF) be IMM-bank leaves.

**Rationale (what the Task 3 survey actually found):** Decisions 1 and 5 and the Risk below assumed there was *"an existing `StateMapping` impl on `UnscentedKalmanFilter`"* to copy. There is not. In the real code:

- `StateMapping` is implemented **per motion model** (`CvMapping`, `CaMapping`, `CtrvMapping`, `CtMapping`) — it maps state *layouts* (CV's 6D, CA's 9D, CTRV's 5D) to a common 6D space. It is orthogonal to filter type.
- `ModelConditionedFilter` **hardcodes `ExtendedKalmanFilter`** as the per-mode leaf, and `ImmFilter::update_step` even constructs a temporary `ExtendedKalmanFilter` internally. There is no leaf-filter trait that UKF or CKF implements, so there is nothing to "copy verbatim".

Making CKF an IMM leaf therefore requires generalizing the IMM leaf into a trait — a genuine `imm.rs` refactor with regression risk against the existing IMM test suite. That is out of scope for a single-PR filter addition and is tracked as its own change.

### 8. Tracker-side filter-kind wiring confirmed unnecessary (post-survey amendment)

**Decision:** Phase 4 is a no-op. Survey of every file under `crates/thresh-tracker/src/` (grep for `FilterKind` / `filter_kind` / `Kf` / `Ekf` / `Ukf` / `kalman`) found **no filter-kind selector of any kind**. Per Decision 4, nothing is added tracker-side; CKF is reachable through the `thresh-filter` API.

### 5. Test parity with UKF

**Decision:** Mirror `ukf::tests` one-for-one. Specifically: linear-system convergence, nonlinear bearings-only example with covariance reduction over multiple updates, and a random-perturbation invariant test that asserts the covariance stays symmetric and positive-definite after `predict` and `update`.

**Rationale:** The CKF and UKF should be empirically interchangeable on linear and mildly-nonlinear problems. Side-by-side tests pin that behaviour and catch regressions if either filter drifts.

### 6. Documentation includes the Arasaratnam-Haykin reference

**Decision:** The module-level docstring on `ckf.rs` cites Arasaratnam & Haykin 2009 ("Cubature Kalman Filters", IEEE Transactions on Automatic Control, vol. 54, no. 6) and points to `docs/reference/` for the worked derivation if one is added.

**Rationale:** This is standard for the existing filter modules and helps anyone debugging cubature rules later.

## Risks

- ~~**IMM-bank wiring.**~~ **Resolved (Decision 7):** the assumed `StateMapping` impl on `UnscentedKalmanFilter` does not exist; `StateMapping` is per motion model and the IMM leaf is a hardcoded EKF. IMM-leaf integration is descoped to the `imm-pluggable-leaf-filter` change. CKF's state representation (flat mean + covariance) is already IMM-compatible and will be adopted by that refactor.
- **Cholesky failure on near-degenerate P.** If the covariance becomes near-singular (e.g. after a zero-noise predict on a stationary target), Cholesky can fail. The UKF handles this the same way; CKF will use the same fallback (return an error, propagated up).
- **Performance.** CKF is 2n cubature points; UKF is 2n+1 sigma points. The compute difference is negligible (one extra column of state propagation), but worth a `criterion` benchmark eventually — captured as a follow-up, not in scope here.

## Open Questions

- ~~Does `thresh-tracker` expose a `FilterKind` enum?~~ **Resolved (Decision 8):** no — no filter-kind selector exists anywhere in `thresh-tracker`. Phase 4 is a no-op.
- ~~Should `CubatureKalmanFilter::new` accept an optional `process_noise_floor` parameter?~~ **Resolved:** no. UKF's `new` has no such parameter; CKF matches the sibling-filter convention (store `(x, p)` as-is, repair via the shared `ensure_psd` step). Adding a knob would also contradict the parameter-free goal.
- Worth comparing CKF + IMM against UKF + IMM on the `thresh-eval` ADS-B scenario to confirm no regression — belongs to the `imm-pluggable-leaf-filter` follow-up, after IMM can take a CKF leaf.
