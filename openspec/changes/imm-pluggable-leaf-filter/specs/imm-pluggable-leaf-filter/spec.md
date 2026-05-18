## Capability: IMM Pluggable Leaf Filter

### Overview

The IMM filter bank in `thresh-filter::imm` becomes filter-agnostic: each model-conditioned leaf is any nonlinear Kalman filter (EKF, UKF, or CKF) selected via `ImmConfig`, rather than a hardcoded `ExtendedKalmanFilter`. This delivers the CKF-as-IMM-leaf integration that the `cubature-kalman-filter` change descoped (its design.md Decision 7).

## ADDED Requirements

### Requirement: LeafFilter trait

`thresh-filter` MUST expose a `LeafFilter` trait capturing the predict/update contract plus state accessors **and mutators** — `x()`, `p()`, `set_x()`, `set_p()` — the IMM bank needs (the interaction step writes mixed state back into each leaf, so setters are mandatory, not optional). It MUST implement the trait for `ExtendedKalmanFilter`, `UnscentedKalmanFilter`, and `CubatureKalmanFilter`.

#### Scenario: All three filters satisfy the trait

**WHEN** the bank holds a `Box<dyn LeafFilter>` and calls `predict`, `update_linear`, and the state accessors

**THEN** an EKF, a UKF, and a CKF instance each behave correctly through the trait object with no behavioural change versus calling their inherent methods directly

**SHALL** be covered by a unit test exercising each concrete filter through `&dyn LeafFilter`.

### Requirement: Selectable IMM leaf kind with EKF default

`ImmConfig` MUST carry an `ImmLeafKind` (`Ekf` / `Ukf` / `Ckf`) that selects the leaf filter for every mode in the bank, defaulting to `Ekf`.

#### Scenario: Existing configs are byte-for-byte unchanged

**WHEN** an `ImmConfig` is built via `cv_ca`, `cv_ctrv`, or `cv_ca_ctrv_ct` without specifying a leaf kind

**THEN** the leaf kind is `Ekf` and the bank produces results identical to the pre-refactor implementation

**SHALL** be enforced by the existing `imm::tests` module passing with **zero edits**.

### Requirement: CKF-leaf IMM behavioural parity

An IMM bank configured with CKF leaves MUST track a synthetic trajectory within statistical equivalence of the equivalent EKF-leaf bank, with valid mode probabilities throughout.

#### Scenario: CKF-leaf bank matches EKF-leaf bank on a fixed seed

**WHEN** two IMM banks share an `ImmConfig` differing only in `leaf_kind` (`Ekf` vs `Ckf`) and are stepped over the same fixed-seed synthetic trajectory

**THEN** the combined-state RMSE difference between the two banks is within a tight tolerance and each bank's mode probabilities sum to one at every step

**SHALL** be a single deterministic test asserting both conditions.

### Requirement: No tracker-side selector and no new dependencies

This change MUST NOT add a `thresh-tracker` filter-kind selector (none exists; out of scope) and MUST NOT add any new external crate to `thresh-filter`.

#### Scenario: Dependency and scope boundary held

**WHEN** the change is built and its diff reviewed

**THEN** `Cargo.lock` gains no new third-party entries and `crates/thresh-tracker/` is untouched

**SHALL** be verified by inspecting the PR diff before merge.
