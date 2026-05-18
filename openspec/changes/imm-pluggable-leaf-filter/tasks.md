# Tasks — IMM Pluggable Leaf Filter

> Single-PR change. Depends on `cubature-kalman-filter` having landed. Phases are sequential; the existing `imm::tests` module must stay green with zero edits throughout (backward-compatibility gate).

## 1. LeafFilter trait

- [ ] 1.1 Define `pub trait LeafFilter: Send + Sync` in `imm.rs` (or a new `leaf.rs`) with `predict(&mut self, &dyn MotionModel, f64)`, `update_linear(&mut self, &DVector<f64>, &DMatrix<f64>, &DMatrix<f64>)`, `x(&self) -> &DVector<f64>`, `p(&self) -> &DMatrix<f64>`, `set_x(&mut self, DVector<f64>)`, `set_p(&mut self, DMatrix<f64>)`.
- [ ] 1.2 `impl LeafFilter for ExtendedKalmanFilter`.
- [ ] 1.3 `impl LeafFilter for UnscentedKalmanFilter` (construct with `UkfParams::default()` where the bank instantiates it).
- [ ] 1.4 `impl LeafFilter for CubatureKalmanFilter`.
- [ ] 1.5 `enum ImmLeafKind { Ekf, Ukf, Ckf }` with `Default = Ekf`; a constructor helper `make_leaf(kind, x, p) -> Box<dyn LeafFilter>`.

## 2. IMM bank refactor

- [ ] 2.1 Replace `ModelConditionedFilter.ekf: ExtendedKalmanFilter` with `leaf: Box<dyn LeafFilter>`; update all field accesses (`f.ekf.x` → `f.leaf.x()`, writes → `set_x` / `set_p`).
- [ ] 2.2 Add `leaf_kind: ImmLeafKind` to `ImmConfig`; default it to `Ekf` in `cv_ca`, `cv_ctrv`, `cv_ca_ctrv_ct` and any other constructor.
- [ ] 2.3 `ImmFilter::new` builds each leaf via `make_leaf(config.leaf_kind, ms, mc)`.
- [ ] 2.4 Replace the throwaway `ExtendedKalmanFilter` in `update_step` with `make_leaf(self.leaf_kind, x_common, p_common)`; thread `leaf_kind` onto `ImmFilter`.
- [ ] 2.5 `cargo test -p thresh-filter imm` — existing IMM tests pass **unedited**.

## 3. CKF-leaf parity test

- [ ] 3.1 Add `imm_ckf_leaf_matches_ekf_leaf`: same `ImmConfig` except `leaf_kind`, same fixed-seed synthetic trajectory; assert combined-state RMSE within a tight statistical tolerance and mode probabilities sum to one each step.
- [ ] 3.2 Add `imm_ukf_leaf_runs` smoke test (bank stepped with `Ukf` leaf produces finite output, probabilities sum to one).

## 4. Documentation

- [ ] 4.1 Update `crates/thresh-filter/README.md`: IMM leaf is now selectable (EKF/UKF/CKF) via `ImmConfig::leaf_kind`.
- [ ] 4.2 Cross-reference: mark the descoped Phase 3 of `cubature-kalman-filter` as delivered here when archiving that change.

## 5. Wrap-up

- [ ] 5.1 `cargo test -p thresh-filter` passes.
- [ ] 5.2 `cargo clippy --workspace --all-targets -- -D warnings` clean.
- [ ] 5.3 `cargo fmt --all -- --check` clean.
- [ ] 5.4 `openspec validate imm-pluggable-leaf-filter --strict --no-interactive` passes.
- [ ] 5.5 Open the PR against `develop`.
