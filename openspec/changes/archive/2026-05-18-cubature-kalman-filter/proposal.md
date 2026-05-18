# Cubature Kalman Filter

## What

Add a Cubature Kalman Filter (CKF) implementation to `thresh-filter` alongside the existing KF / EKF / UKF, and wire it into the tracker's filter-kind enum and the IMM filter bank as a selectable variant.

## Why

The CKF (Arasaratnam & Haykin, 2009) is a deterministic-sample nonlinear Kalman filter that uses **2n equally-weighted cubature points** drawn from the third-order spherical-radial cubature rule. Compared to the UKF it offers:

- **Provable stability for highly nonlinear systems.** UKF's central sigma point can take negative weight under certain `(α, β, κ)` settings, which lets the predicted covariance become indefinite in rare maneuver geometries. CKF's 2n equally-weighted positive points avoid this entirely.
- **No tuning knobs.** UKF has three hyperparameters (`α`, `β`, `κ`) that interact with state dimensionality and require care. CKF is parameter-free — fewer surprises in production code.
- **Same asymptotic accuracy as UKF** at the third order, so the cost of the swap is only in tuning effort and corner-case robustness.

thresh tracks aerospace targets where these corner cases bite: ballistic re-entry (very high state-vector dynamic range), hard-maneuvering UAVs (sharp changes in turn rate), and OTHR ionospheric refraction (highly nonlinear measurement model). CKF is standard practice in many radar tracking shops as a UKF complement; adding it gives users a robust drop-in option without losing UKF's flexibility.

CKF is **independent of the flight-data-training-pipeline change** — they touch different parts of the tracker stack and can ship in either order.

## How

- New module `crates/thresh-filter/src/ckf.rs` modelled on the existing `ukf.rs` API:
  - `pub struct CubatureKalmanFilter { x, p, ... }`
  - `pub fn new(x, p) -> Self` (no params struct; CKF is parameter-free).
  - `pub fn predict(&mut self, model: &dyn MotionModel, dt: f64)`
  - `pub fn update<F>(&mut self, z, h_fn, r) where F: Fn(&DVector<f64>) -> DVector<f64>`
  - `pub fn update_linear(&mut self, z, h, r)` for the common linear-measurement case.
- Internal helpers compute the 2n cubature points via `±√n · e_i` columns of a Cholesky factor of P (standard third-order spherical-radial rule).
- Re-use `MotionModel` from `thresh-filter::traits`. No new traits.
- Add `ckf` as a variant to whichever filter-kind selector `thresh-tracker` exposes (TBD in design.md after a brief survey of the existing tracker selection code) and add it to the IMM filter bank as a permitted leaf.
- Tests mirror `ukf::tests` one-for-one: identity motion model + linear measurement converges; non-linear bearings-only example reduces position covariance over multiple updates; covariance stays symmetric and positive-definite under random perturbations.

## Out of scope

- High-degree cubature rules (5th, 7th order). The standard third-order rule is what every real-world tracker uses; higher-order variants are an academic refinement.
- Square-root CKF (SR-CKF). Useful for limited-precision hardware (FPGA, embedded); not needed for the workspace's host-CPU targets.
- Comparative benchmark against UKF on thresh's existing scenarios. Worth doing but belongs in a separate evaluation change.
- New motion models. CKF reuses everything in `models/`.

## Affected crates and paths

- `crates/thresh-filter/src/ckf.rs` — new module.
- `crates/thresh-filter/src/lib.rs` — `pub mod ckf;`.
- `crates/thresh-filter/Cargo.toml` — no new dependencies (CKF needs only `nalgebra`, already in tree).
- `crates/thresh-tracker/` — extend the filter-kind enum / selector. Exact files to be determined during Task 3.
- `crates/thresh-filter/src/imm.rs` — accept CKF as a leaf-filter implementor of `StateMapping`. May require a small trait-impl block.

## Dependencies

- None. CKF is self-contained inside `thresh-filter` plus a tiny wiring change in `thresh-tracker` and `imm.rs`.
- Independent of the `flight-data-training-pipeline` change.
