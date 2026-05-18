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
- IMM bank usable with any leaf kind; `Ekf` remains the default so all existing `ImmConfig` constructors behave identically.
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

### 2. Leaf kind selected on `ImmConfig`, default `Ekf`

**Decision:** Add `pub leaf_kind: ImmLeafKind` (`enum ImmLeafKind { Ekf, Ukf, Ckf }`) to `ImmConfig`, defaulting to `Ekf`. Existing constructors (`cv_ca`, `cv_ctrv`, `cv_ca_ctrv_ct`) set `Ekf` so behaviour is byte-for-byte unchanged.

**Rationale:** Backward compatibility is the dominant risk. A defaulted field localizes the change and lets the parity test flip a single knob.

**Ownership:** `ImmFilter::new` copies `config.leaf_kind` into a new `ImmFilter` field at construction time; `update_step` reads `self.leaf_kind` (not the config) when instantiating the temporary common-space leaf, so the kind is owned by the filter instance rather than re-read from the config on every call.

### 3. UKF leaf uses default `UkfParams`

**Decision:** When the leaf kind is `Ukf`, construct with `UkfParams::default()`. No UKF tuning is surfaced through `ImmConfig` in this change.

**Rationale:** Keeps the selector a simple enum. UKF tuning inside IMM is a separate concern; CKF (the motivating leaf) is parameter-free anyway.

### 4. The internal common-space update leaf follows the configured kind

**Decision:** The temporary `ExtendedKalmanFilter` in `update_step` becomes a `LeafFilter` of the configured kind, constructed per call from the common-space `(x, p)`.

**Rationale:** For a *linear* common-space `H` all three filters are equivalent to the standard KF, so this does not change EKF-leaf results (covered by the parity test), but it keeps the leaf kind consistent end-to-end.

## Risks

- **Regressing the existing IMM suite.** Mitigation: `Ekf` default + run the full `imm::tests` module unchanged; they must stay green with zero edits.
- **Trait-object dispatch cost.** Negligible relative to the per-step matrix algebra; not optimized here.
- **`set_x` / `set_p` widening the trait.** The interaction step writes back into each leaf (`f.ekf.x = ms`). Setters are required; kept minimal.

## Open Questions

- Should `ImmLeafKind` eventually be per-mode rather than per-bank? Deferred (Non-Goal) until a concrete need appears.
- Worth a `criterion` benchmark of EKF vs UKF vs CKF IMM banks? Yes, but in a separate evaluation change once this lands.
