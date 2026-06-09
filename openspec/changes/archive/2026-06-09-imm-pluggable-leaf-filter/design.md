# Design — IMM Pluggable Leaf Filter

## Context

`ImmFilter` maintains a bank of `ModelConditionedFilter`s. Today each one holds a concrete `ExtendedKalmanFilter`:

```rust
pub struct ModelConditionedFilter {
    pub ekf: ExtendedKalmanFilter,
    pub model: Box<dyn MotionModel>,
    pub mapping: Box<dyn StateMapping>,
}
```

`ImmFilter::update_step` also builds a throwaway `ExtendedKalmanFilter` in common 6D space to perform the measurement update before back-projecting through `StateMapping`. EKF is therefore wired in at two points, both concrete.

EKF, UKF (`UnscentedKalmanFilter`), and CKF (`CubatureKalmanFilter`) already expose the same predict/update surface:

- `new(x, p)` (UKF additionally takes `UkfParams`)
- `predict(&mut self, model: &dyn MotionModel, dt: f64)`
- `update_linear(&mut self, z, h, r)` and a closure `update`
- public `x: DVector<f64>`, `p: DMatrix<f64>`

So a trait over that surface is mechanical; the work is the IMM refactor and not regressing the existing IMM behaviour.

## Goals / Non-Goals

**Goals:**

- A `LeafFilter` trait so the IMM bank is filter-agnostic.
- `impl LeafFilter` for EKF, UKF, CKF.
- IMM bank usable with any leaf kind; `Ekf` remains the default so all existing `ImmFilter::new` callers (including `thresh-tracker`) behave identically.
- A fixed-seed parity test: CKF-leaf IMM ≈ EKF-leaf IMM on a synthetic trajectory; mode probabilities sum to one.

**Non-Goals:**

- Tracker-side filter selection (no such selector exists in `thresh-tracker`).
- Per-leaf heterogeneity (e.g. EKF for mode 0, CKF for mode 1) — the bank uses one leaf kind for all modes in this change; mixed banks can be a follow-up if a use case appears.
- Touching `StateMapping`, motion models, or the IMM math (interaction / mode-probability / combine steps are unchanged).

## Decisions

### 1. A `LeafFilter` trait, not generics

**Decision:** `pub trait LeafFilter: Send + Sync` with `predict`, `update_linear`, `x()`, `p()`, `set_x()`, `set_p()`. `ModelConditionedFilter` holds `Box<dyn LeafFilter>`.

**Rationale:** The bank stores a heterogeneous-by-config collection and is already trait-object-heavy (`Box<dyn MotionModel>`, `Box<dyn StateMapping>`). Generics would virally parameterize `ImmFilter`, `ImmConfig`, and every caller. A boxed trait keeps the public `ImmFilter` type unchanged.

**API impact:** `ExtendedKalmanFilter`, `UnscentedKalmanFilter`, and `CubatureKalmanFilter` currently expose `x` / `p` as public fields with no accessor methods. Each must gain four trivial wrappers (`x()`, `p()`, `set_x()`, `set_p()`) to implement the trait — a small but real public-API surface addition (additive, not breaking: the existing public fields stay). Tracked as a Risk below.

### 2. Leaf kind selected at IMM construction, default `Ekf`

**Decision:** `enum ImmLeafKind { Ekf, Ukf, Ckf }` (with `#[default] Ekf`) is supplied **at IMM construction**, not via `ImmConfig`. `ImmFilter::with_leaf_kind(config, kind, &x, &p)` selects the kind; `ImmFilter::new(config, &x, &p)` delegates to it with `ImmLeafKind::Ekf`. The kind is owned by the `ImmFilter` instance as a `pub leaf_kind: ImmLeafKind` field; `update_step` reads `self.leaf_kind` when instantiating the temporary common-space leaf, so it is fixed at construction rather than re-read per call. `ImmConfig` and its `cv_ca` / `cv_ctrv` / `cv_ca_ctrv_ct` constructors are **unchanged**.

**Rationale:** The original proposal placed `leaf_kind` on `ImmConfig`. During apply this was found to contradict the change's own primary backward-compatibility gate — see Decision 3. Selecting the kind at construction keeps `ImmConfig` and `ImmFilter::new` byte-for-byte source-compatible (so `thresh-tracker` and all struct-literal call sites are untouched) while still letting the parity test flip a single knob (`with_leaf_kind`'s `kind` argument).

### 3. Why the selector is not a field on `ImmConfig` (contradiction found during apply)

**Decision:** Reject the `pub leaf_kind` field on `ImmConfig` from the original proposal; use the construction-time selector in Decision 2 instead. Amend tasks/spec wording from "existing `imm::tests` pass with **zero edits**" to "no behavioural or assertion edits; one mechanical field-access update is permitted."

**Why (discovered while implementing, mirrors the `cubature-kalman-filter` Decision 7 pattern):** The proposal asserted a `pub leaf_kind` field on `ImmConfig` could be added with *zero* edits to `imm::tests`. That is false in Rust:

- `imm::tests` constructs `ImmConfig` via **exhaustive struct literals** in four places (`config_validation_*` ×3, plus the two single-model configs in `cv_ctrv_mode_switching_on_maneuver`). Adding any required public field makes those literals fail to compile (`E0063`). There is no same-crate escape (privacy and `#[non_exhaustive]` do not exempt in-crate literals; the tests are a child module of `imm`).
- Renaming `ModelConditionedFilter.ekf` → `leaf` (required by Decision 1) breaks the one test that reaches in as `f.ekf.x` / `f.ekf.p` (`interaction_uniform_identical_states_unchanged`).

So "`leaf_kind` on `ImmConfig`" **and** "zero test edits" are mutually exclusive as written, and a config field would *also* force edits in `thresh-tracker` is avoided only because it uses a `cv_*` constructor — but the struct-literal tests still break. The construction-time selector satisfies the dominant constraint (no behavioural regression; `thresh-tracker` and `ImmConfig` untouched) at the cost of one **mechanical** edit: `f.ekf.x/p` → `f.leaf.x()/p()` in `interaction_uniform_identical_states_unchanged`. That edit changes no assertion, no input, and no expected value; with the `Ekf` default the numerics are bit-identical. All other existing `imm::tests` are genuinely unedited.

### 4. UKF leaf uses default `UkfParams`

**Decision:** When the leaf kind is `Ukf`, construct with `UkfParams::default()`. No UKF tuning is surfaced through `ImmConfig` in this change.

**Rationale:** Keeps the selector a simple enum. UKF tuning inside IMM is a separate concern; CKF (the motivating leaf) is parameter-free anyway.

### 5. The internal common-space update leaf follows the configured kind

**Decision:** The temporary `ExtendedKalmanFilter` in `update_step` becomes a `LeafFilter` of the configured kind, constructed per call from the common-space `(x, p)`.

**Rationale:** For a *linear* common-space `H` all three filters are equivalent to the standard KF, so this does not change EKF-leaf results (covered by the parity test), but it keeps the leaf kind consistent end-to-end.

## Risks

- **Regressing the existing IMM suite.** Mitigation: `Ekf` default + run the full `imm::tests` module with no behavioural or assertion changes — only the single mechanical `f.ekf.x/p` → `f.leaf.x()/p()` access update of Decision 3 is permitted; every assertion, input, and expected value stays identical and the `Ekf` default keeps the numerics bit-for-bit.
- **Trait-object dispatch cost.** Negligible relative to the per-step matrix algebra; not optimized here.
- **`set_x` / `set_p` widening the trait.** The interaction step writes the mixed state back into each leaf (`f.leaf.set_x(ms)`). Setters are required; kept minimal.

## Open Questions

- Should `ImmLeafKind` eventually be per-mode rather than per-bank? Deferred (Non-Goal) until a concrete need appears.
- Worth a `criterion` benchmark of EKF vs UKF vs CKF IMM banks? Yes, but in a separate evaluation change once this lands.
