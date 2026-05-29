# IMM Pluggable Leaf Filter

## What

Generalize the IMM filter bank in `thresh-filter::imm` so each model-conditioned leaf can be any nonlinear Kalman filter (EKF, UKF, or CKF) instead of the hardcoded `ExtendedKalmanFilter`. Then wire `CubatureKalmanFilter` in as a selectable IMM leaf — the integration that the `cubature-kalman-filter` change descoped (see that change's design.md Decision 7).

## Why

`cubature-kalman-filter` shipped the CKF as a `thresh-filter`-level filter. Its design.md originally assumed CKF could join the IMM bank by copying *"the existing `StateMapping` impl on `UnscentedKalmanFilter`"*. The Task 3 survey found that assumption is false:

- `StateMapping` is implemented **per motion model** (`CvMapping`, `CaMapping`, `CtrvMapping`, `CtMapping`) — it maps state *layouts* to a common 6D space. It has nothing to do with filter type.
- `ModelConditionedFilter` **hardcodes `ExtendedKalmanFilter`**, and `ImmFilter::update_step` constructs a temporary `ExtendedKalmanFilter` internally. There is no leaf-filter trait.

So letting the IMM bank use a UKF or CKF leaf is a real `imm.rs` refactor with regression risk against the existing IMM test suite — too large to ride along inside a single-PR filter addition. This change owns that refactor and the resulting CKF-leaf behavioural-parity test.

## How

- Introduce a `LeafFilter` trait in `thresh-filter` capturing the predict/update contract the IMM bank needs: `predict(&mut self, &dyn MotionModel, dt)`, `update_linear(&mut self, z, h, r)`, and accessors for `x` / `p`.
- Implement `LeafFilter` for `ExtendedKalmanFilter`, `UnscentedKalmanFilter`, and `CubatureKalmanFilter` (the three already share this API surface).
- Replace `ModelConditionedFilter`'s concrete `ekf: ExtendedKalmanFilter` field with `Box<dyn LeafFilter>`, and the internal temporary EKF in `update_step` with a leaf chosen by the bank's configured filter kind.
- Add an IMM-level leaf-kind selector (`ImmLeafKind { Ekf, Ukf, Ckf }`, default `Ekf` for backward compatibility) chosen at construction via `ImmFilter::with_leaf_kind`, with `ImmFilter::new` defaulting to `Ekf`. _(Amended from "on `ImmConfig`": placing it on `ImmConfig` was found during apply to contradict this change's own zero-edit backward-compatibility gate — see design.md Decision 3. `ImmConfig` is left unchanged.)_
- Behavioural-parity test: an IMM bank configured with CKF leaves tracks a fixed-seed synthetic trajectory within statistical equivalence of the EKF-leaf bank, and mode probabilities still sum to one.

## Out of scope

- Tracker-side filter-kind selector. `thresh-tracker` exposes no such enum (confirmed by the `cubature-kalman-filter` survey, Decision 8); not introduced here.
- Square-root / higher-order filter variants.
- A CKF-vs-UKF-vs-EKF IMM accuracy benchmark on `thresh-eval` scenarios — a separate evaluation change once the leaf is pluggable.
- Changes to `StateMapping` or the motion-model set; they are orthogonal and stay as-is.

## Affected crates and paths

- `crates/thresh-filter/src/imm.rs` — `LeafFilter` trait, generalized `ModelConditionedFilter`, leaf-kind selector, refactored `update_step`.
- `crates/thresh-filter/src/ekf.rs`, `ukf.rs`, `ckf.rs` — `impl LeafFilter` blocks (no behavioural change to the filters themselves).
- `crates/thresh-filter/README.md` — note IMM leaf selectability once it lands.

## Dependencies

- Depends on `cubature-kalman-filter` having landed (the `CubatureKalmanFilter` type and its `predict`/`update_linear` API).
- No new external crates.
