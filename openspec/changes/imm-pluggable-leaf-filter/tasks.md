# Tasks — IMM Pluggable Leaf Filter

> Single-PR change. Depends on `cubature-kalman-filter` having landed. Phases are sequential. Backward-compatibility gate: the existing `imm::tests` module stays green with **no behavioural or assertion edits** — only the single mechanical field-access update of design.md Decision 3 is permitted (the original "zero edits" wording was found self-contradictory during apply; see Decision 3).

## 1. LeafFilter trait

- [x] 1.1 Define `pub trait LeafFilter: Send + Sync` **in `imm.rs`** (trait + `ImmLeafKind` + `make_leaf` are ~100 lines total — co-locating them avoids a new module and extra `pub use` re-exports; revisit a dedicated `leaf.rs` only if later leaf utilities materialize) with `predict(&mut self, &dyn MotionModel, f64)`, `update_linear(&mut self, &DVector<f64>, &DMatrix<f64>, &DMatrix<f64>)`, `x(&self) -> &DVector<f64>`, `p(&self) -> &DMatrix<f64>`, `set_x(&mut self, DVector<f64>)`, `set_p(&mut self, DMatrix<f64>)`.
- [x] 1.2 `impl LeafFilter for ExtendedKalmanFilter`.
- [x] 1.3 `impl LeafFilter for UnscentedKalmanFilter` (construct with `UkfParams::default()` where the bank instantiates it).
- [x] 1.4 `impl LeafFilter for CubatureKalmanFilter`.
- [x] 1.5 `enum ImmLeafKind { Ekf, Ukf, Ckf }` with `#[default] Ekf`; a constructor helper `make_leaf(kind, x, p) -> Box<dyn LeafFilter>`.

## 2. IMM bank refactor

- [x] 2.1 Replace `ModelConditionedFilter.ekf: ExtendedKalmanFilter` with `leaf: Box<dyn LeafFilter>`; update all field accesses (`f.ekf.x` → `f.leaf.x()`, writes → `set_x` / `set_p`).
- [x] 2.2 _Amended (design.md Decision 3): the selector is **not** a field on `ImmConfig`._ Add `pub leaf_kind: ImmLeafKind` to **`ImmFilter`** (owned by the instance, `#[default] Ekf`). `ImmConfig` and the `cv_ca` / `cv_ctrv` / `cv_ca_ctrv_ct` constructors are left unchanged, so `thresh-tracker` and the in-test struct literals do not need edits.
- [x] 2.3 _Amended:_ add `ImmFilter::with_leaf_kind(config, kind, &x, &p)` which builds each leaf via `make_leaf(kind, ms, mc)`; `ImmFilter::new` delegates to it with `ImmLeafKind::Ekf` (signature unchanged).
- [x] 2.4 Replace the throwaway `ExtendedKalmanFilter` in `update_step` with `make_leaf(self.leaf_kind, x_common, p_common)`; `leaf_kind` is threaded onto `ImmFilter` (read once per call into a local to keep it disjoint from the `&mut self.filters` loop borrow).
- [x] 2.5 `cargo test -p thresh-filter imm` — all pre-existing IMM tests pass with no behavioural/assertion edits; the only edit is the mechanical `f.ekf.x/p` → `f.leaf.x()/p()` access change in `interaction_uniform_identical_states_unchanged` (Decision 3). `Ekf` default ⇒ bit-identical numerics.

## 3. CKF-leaf parity test

- [x] 3.1 Added `imm_ckf_leaf_matches_ekf_leaf`: `ImmFilter::new` (Ekf) vs `ImmFilter::with_leaf_kind(.., Ckf, ..)` on the same fixed-seed synthetic trajectory; asserts max combined-state difference `< 1e-6` (CV+CA are linear, so the leaf kinds are theoretically identical here) and mode probabilities sum to one each step.
- [x] 3.2 Added `imm_ukf_leaf_runs` smoke test (bank stepped with `Ukf` leaf produces finite output, probabilities sum to one). Also added `leaf_filter_trait_object_matches_inherent` covering the spec's "all three filters satisfy the trait" scenario (each concrete filter through `&mut dyn LeafFilter` matches its inherent methods).

## 4. Documentation

- [x] 4.1 Updated `crates/thresh-filter/README.md`: IMM leaf is selectable (EKF/UKF/CKF) via `ImmFilter::with_leaf_kind`; `ImmFilter::new` defaults to EKF. _(Reflects the Decision-3 amendment: selection is at construction, not `ImmConfig::leaf_kind`.)_
- [x] 4.2 Cross-reference: the descoped Phase 3 of `cubature-kalman-filter` (CKF-as-IMM-leaf, that change's design.md Decision 7) is delivered here. Noted in the archived change's tasks.md.

## 5. Wrap-up

- [x] 5.1 `cargo test -p thresh-filter` passes (40 tests).
- [x] 5.2 `cargo clippy --workspace --all-targets -- -D warnings` clean.
- [x] 5.3 `cargo fmt --all -- --check` clean.
- [x] 5.4 `openspec validate imm-pluggable-leaf-filter --strict --no-interactive` passes.
- [x] 5.5 Open the PR against `develop`.
