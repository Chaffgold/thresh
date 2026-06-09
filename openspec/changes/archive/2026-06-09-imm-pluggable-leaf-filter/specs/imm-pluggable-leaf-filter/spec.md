## Capability: IMM Pluggable Leaf Filter

### Overview

The IMM filter bank in `thresh-filter::imm` becomes filter-agnostic: each model-conditioned leaf is any nonlinear Kalman filter (EKF, UKF, or CKF) selected at IMM construction (`ImmFilter::with_leaf_kind`; `ImmFilter::new` defaults to EKF), rather than a hardcoded `ExtendedKalmanFilter`. This delivers the CKF-as-IMM-leaf integration that the `cubature-kalman-filter` change descoped (its design.md Decision 7). The selector lives at construction rather than on `ImmConfig` per design.md Decision 3 (a config field was found to contradict this change's own backward-compatibility gate during apply).

## ADDED Requirements

### Requirement: LeafFilter trait

`thresh-filter` MUST expose a `LeafFilter` trait capturing the predict/update contract plus state accessors **and mutators** — `x()`, `p()`, `set_x()`, `set_p()` — the IMM bank needs (the interaction step writes mixed state back into each leaf, so setters are mandatory, not optional). It MUST implement the trait for `ExtendedKalmanFilter`, `UnscentedKalmanFilter`, and `CubatureKalmanFilter`.

#### Scenario: All three filters satisfy the trait

**WHEN** the bank holds a `Box<dyn LeafFilter>` and calls `predict`, `update_linear`, and the state accessors

**THEN** an EKF, a UKF, and a CKF instance each behave correctly through the trait object with no behavioural change versus calling their inherent methods directly

**SHALL** be covered by a unit test exercising each concrete filter through `&dyn LeafFilter`.

### Requirement: Selectable IMM leaf kind with EKF default

The IMM bank MUST expose an `ImmLeafKind` (`Ekf` / `Ukf` / `Ckf`) that selects the leaf filter for every mode, chosen at construction via `ImmFilter::with_leaf_kind` and owned by the `ImmFilter` instance. `ImmFilter::new` MUST default to `Ekf`. `ImmConfig` and its `cv_ca` / `cv_ctrv` / `cv_ca_ctrv_ct` constructors MUST remain unchanged (see design.md Decision 3).

#### Scenario: Existing callers are byte-for-byte unchanged

**WHEN** an `ImmFilter` is built via `ImmFilter::new` from a config produced by `cv_ca`, `cv_ctrv`, or `cv_ca_ctrv_ct`

**THEN** the leaf kind is `Ekf` and the bank produces results identical to the pre-refactor implementation, with `ImmConfig` and `ImmFilter::new` source-compatible (so `thresh-tracker` is untouched)

**SHALL** be enforced by the existing `imm::tests` module passing with no behavioural or assertion edits — only the single mechanical `f.ekf.x/p` → `f.leaf.x()/p()` access update of design.md Decision 3 is permitted, which changes no assertion, input, or expected value.

### Requirement: CKF-leaf IMM behavioural parity

An IMM bank configured with CKF leaves MUST track a synthetic trajectory within statistical equivalence of the equivalent EKF-leaf bank, with valid mode probabilities throughout.

#### Scenario: CKF-leaf bank matches EKF-leaf bank on a fixed seed

**WHEN** two IMM banks are built from equivalent `ImmConfig`s — one via `ImmFilter::new` (Ekf), one via `ImmFilter::with_leaf_kind(.., Ckf, ..)` — and stepped over the same fixed-seed synthetic trajectory

**THEN** the combined-state RMSE difference between the two banks is within a tight tolerance and each bank's mode probabilities sum to one at every step

**SHALL** be a single deterministic test asserting both conditions.

### Requirement: No tracker-side selector and no new dependencies

This change MUST NOT add a `thresh-tracker` filter-kind selector (none exists; out of scope) and MUST NOT add any new external crate to `thresh-filter`.

#### Scenario: Dependency and scope boundary held

**WHEN** the change is built and its diff reviewed

**THEN** `Cargo.lock` gains no new third-party entries and `crates/thresh-tracker/` is untouched

**SHALL** be verified by inspecting the PR diff before merge.
