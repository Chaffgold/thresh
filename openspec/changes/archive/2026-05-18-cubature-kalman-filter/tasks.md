# Tasks — Cubature Kalman Filter

> Single-PR change. Phases are sequential — each builds on the last; tests gate the next phase. Phase 5 may be skipped if `thresh-tracker` exposes no filter-kind selector.

## 1. Core implementation

- [x] 1.1 Create `crates/thresh-filter/src/ckf.rs` modelled on `ukf.rs`. Module-level docstring citing Arasaratnam & Haykin 2009.
- [x] 1.2 Implement `pub struct CubatureKalmanFilter { x: DVector<f64>, p: DMatrix<f64> }`.
- [x] 1.3 Implement `pub fn new(x: DVector<f64>, p: DMatrix<f64>) -> Self`. No params struct. _Matches `EKF::new`/`UKF::new`: stores `(x, p)` as-is, no dim validation (sibling-filter convention)._
- [x] 1.4 Implement `fn cubature_points(&self) -> Vec<DVector<f64>>` — Cholesky of P, scale by `√n`, form `2n` points. Internal. _Takes `&mut self` to reuse the UKF `ensure_psd` repair before Cholesky._
- [x] 1.5 Implement `pub fn predict(&mut self, model: &dyn MotionModel, dt: f64)` — propagate each cubature point through the model, recompute mean and covariance with equal `1/(2n)` weights, add process noise.
- [x] 1.6 Implement `pub fn update<F: Fn(&DVector<f64>) -> DVector<f64>>(&mut self, z, h_fn, r)` — predict each cubature point into measurement space, compute innovation covariance + cross-covariance, apply standard Kalman update.
- [x] 1.7 Implement `pub fn update_linear(&mut self, z, h, r)` — closed-form linear update path (matches `ukf.rs`).
- [x] 1.8 Add `pub mod ckf;` to `crates/thresh-filter/src/lib.rs`.

## 2. Unit tests

- [x] 2.1 `cubature_points_are_2n_with_equal_weights` — n=4 state, assert exactly 8 points, mean of points equals `x`, sample covariance of points (with `1/(2n)` weight) equals `P` within tolerance.
- [x] 2.2 `predict_zero_noise_identity_motion_preserves_state` — identity motion model, zero process noise, predict-then-check `x` and `P` unchanged within numerical tolerance.
- [x] 2.3 `update_linear_converges_to_truth` — linear system, repeated noisy measurements, assert state converges to truth and covariance trace decreases monotonically.
- [x] 2.4 `update_nonlinear_bearings_only_reduces_uncertainty` — bearings-only multi-sensor test. _`ukf::tests` has no bearings-only case to mirror; implemented per spec scenario 2.4 (covariance shrinks; estimate improves on the prior). Tight convergence-to-truth is not asserted — bearings-only observability is geometry-dependent._
- [x] 2.5 `covariance_stays_symmetric_pd_under_perturbation` — random init, random predict/update sequence, assert P symmetric (||P - P^T|| < ε) and positive-definite (smallest eigenvalue > 0) after each step.
- [x] 2.6 `ckf_and_ukf_agree_on_linear_problem` — same input, same motion model, same measurements, assert posterior means within 1e-6 and covariances within 1e-4. _CV motion model + linear H, 15 steps; passes at the spec tolerances._

## 3. IMM filter-bank integration — DESCOPED (see design.md Decision 7)

> Survey outcome: there is **no** `StateMapping` impl on `UnscentedKalmanFilter` to copy. `StateMapping` is per motion model (`Cv/Ca/Ctrv/Ct`), and `ModelConditionedFilter` hardcodes an `ExtendedKalmanFilter` leaf with no pluggable-filter trait. Making CKF an IMM leaf is a real `imm.rs` refactor, tracked as the separate `imm-pluggable-leaf-filter` change. CKF's flat `(x, p)` state is already IMM-compatible.

- [x] 3.1 Inspect `thresh-filter::imm` to find the `StateMapping` impl pattern used for UKF leaves. _Done — found the mismatch above; no UKF-leaf `StateMapping` exists._
- [~] 3.2 Descoped to `imm-pluggable-leaf-filter`: no UKF `StateMapping` to copy; requires a leaf-filter trait refactor of `imm.rs`.
- [~] 3.3 Descoped to `imm-pluggable-leaf-filter`: the IMM-leaf parity test moves with the refactor.

## 4. Tracker-side filter-kind wiring (conditional)

- [x] 4.1 Survey `thresh-tracker` for any filter-kind enum or selector. _Surveyed all of `crates/thresh-tracker/src` (grep `FilterKind`/`filter_kind`/`Kf`/`Ekf`/`Ukf`/`kalman`): **no filter-kind selector exists**. Per design Decision 4, Phase 4 is skipped — CKF is a `thresh-filter`-level offering only._
- [~] 4.2 Skipped — no `FilterKind`-style enum exists (see 4.1).
- [~] 4.3 Skipped — no tracker-level filter selector to drive (see 4.1).

## 5. Documentation

- [x] 5.1 Add a section to `crates/thresh-filter/README.md` (or create one) documenting CKF alongside UKF, with a one-paragraph "when to choose which" guide. _Created `crates/thresh-filter/README.md` (none existed): filter table, CKF-vs-UKF guidance, Arasaratnam–Haykin citation, IMM-leaf pointer._
- [~] 5.2 N/A — conditional task. `docs/reference/` has no Kalman-filter notes file (only `benchmarks.md`, `profiling.md`, transformer references). No file to extend; not creating one to avoid scope creep beyond this change.
- [x] 5.3 Update `CLAUDE.md`'s `thresh-filter` line in the architecture diagram to mention CKF. _`(KF, EKF, UKF, CKF + motion models: …)`._

## 6. Wrap-up

- [x] 6.1 `cargo test -p thresh-filter` passes. _34 passed, 0 failed (6 new CKF tests + existing KF/EKF/UKF/IMM)._
- [x] 6.2 `cargo clippy --workspace --all-targets -- -D warnings` clean.
- [x] 6.3 `cargo fmt --all -- --check` clean.
- [x] 6.4 OpenSpec validate passes against the updated artifacts. _`openspec validate --all --strict --no-interactive` → 31 passed, 0 failed (incl. amended `cubature-kalman-filter` and new `imm-pluggable-leaf-filter`)._
- [x] 6.5 Open the PR against `develop`. _Pushed to `claude/review-prs-openspec-U60vd`; draft PR targets `develop`._
