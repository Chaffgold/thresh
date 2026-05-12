# Tasks — Cubature Kalman Filter

> Single-PR change. Phases are sequential — each builds on the last; tests gate the next phase. Phase 5 may be skipped if `thresh-tracker` exposes no filter-kind selector.

## 1. Core implementation

- [ ] 1.1 Create `crates/thresh-filter/src/ckf.rs` modelled on `ukf.rs`. Module-level docstring citing Arasaratnam & Haykin 2009.
- [ ] 1.2 Implement `pub struct CubatureKalmanFilter { x: DVector<f64>, p: DMatrix<f64> }`.
- [ ] 1.3 Implement `pub fn new(x: DVector<f64>, p: DMatrix<f64>) -> Self`. No params struct.
- [ ] 1.4 Implement `fn cubature_points(&self) -> Vec<DVector<f64>>` — Cholesky of P, scale by `√n`, form `2n` points. Internal.
- [ ] 1.5 Implement `pub fn predict(&mut self, model: &dyn MotionModel, dt: f64)` — propagate each cubature point through the model, recompute mean and covariance with equal `1/(2n)` weights, add process noise.
- [ ] 1.6 Implement `pub fn update<F: Fn(&DVector<f64>) -> DVector<f64>>(&mut self, z, h_fn, r)` — predict each cubature point into measurement space, compute innovation covariance + cross-covariance, apply standard Kalman update.
- [ ] 1.7 Implement `pub fn update_linear(&mut self, z, h, r)` — closed-form linear update path (matches `ukf.rs`).
- [ ] 1.8 Add `pub mod ckf;` to `crates/thresh-filter/src/lib.rs`.

## 2. Unit tests

- [ ] 2.1 `cubature_points_are_2n_with_equal_weights` — n=4 state, assert exactly 8 points, mean of points equals `x`, sample covariance of points (with `1/(2n)` weight) equals `P` within tolerance.
- [ ] 2.2 `predict_zero_noise_identity_motion_preserves_state` — identity motion model, zero process noise, predict-then-check `x` and `P` unchanged within numerical tolerance.
- [ ] 2.3 `update_linear_converges_to_truth` — linear system, repeated noisy measurements, assert state converges to truth and covariance trace decreases monotonically.
- [ ] 2.4 `update_nonlinear_bearings_only_reduces_uncertainty` — same bearings-only test as `ukf::tests` for direct parity.
- [ ] 2.5 `covariance_stays_symmetric_pd_under_perturbation` — random init, random predict/update sequence, assert P symmetric (||P - P^T|| < ε) and positive-definite (smallest eigenvalue > 0) after each step.
- [ ] 2.6 `ckf_and_ukf_agree_on_linear_problem` — same input, same motion model, same measurements, assert posterior means within 1e-6 and covariances within 1e-4.

## 3. IMM filter-bank integration

- [ ] 3.1 Inspect `thresh-filter::imm` to find the `StateMapping` impl pattern used for UKF leaves.
- [ ] 3.2 Add a `StateMapping` impl for `CubatureKalmanFilter` (likely a verbatim copy of the UKF impl since the state representation is identical).
- [ ] 3.3 Add a unit test asserting CKF can be used as a leaf in an IMM filter bank without errors.

## 4. Tracker-side filter-kind wiring (conditional)

- [ ] 4.1 Survey `thresh-tracker` for any filter-kind enum or selector. If absent, skip phase 4 entirely; CKF is available at the `thresh-filter` API level only.
- [ ] 4.2 If a `FilterKind`-style enum exists: add a `Ckf` variant and the dispatching arm in whatever match block uses it.
- [ ] 4.3 Add a tracker-level integration test that drives a synthetic scenario with `FilterKind::Ckf` selected.

## 5. Documentation

- [ ] 5.1 Add a section to `crates/thresh-filter/README.md` (or create one) documenting CKF alongside UKF, with a one-paragraph "when to choose which" guide.
- [ ] 5.2 If `docs/reference/` has a Kalman-filter notes file, add a short worked derivation of the third-order spherical-radial cubature rule.
- [ ] 5.3 Update `CLAUDE.md`'s `thresh-filter` line in the architecture diagram to mention CKF.

## 6. Wrap-up

- [ ] 6.1 `cargo test -p thresh-filter` passes.
- [ ] 6.2 `cargo clippy --workspace --all-targets -- -D warnings` clean.
- [ ] 6.3 `cargo fmt --all -- --check` clean.
- [ ] 6.4 OpenSpec validate passes against the updated artifacts.
- [ ] 6.5 Open the PR against `develop`.
