# thresh-filter

Kalman filter family with configurable motion models for the thresh tracking
stack.

## Filters

| Filter | Type | Use it for |
|---|---|---|
| `kf` — Kalman Filter | Linear | Linear-Gaussian models; cheapest. |
| `ekf` — Extended KF | Jacobian linearization | Mildly nonlinear models; default IMM leaf. |
| `ukf` — Unscented KF | `2n + 1` weighted sigma points | Strongly nonlinear models; tunable via `UkfParams` (`alpha`, `beta`, `kappa`). |
| `ckf` — Cubature KF | `2n` equally-weighted cubature points | Strongly nonlinear models where robustness matters more than tuning flexibility. Parameter-free. |

All filters share the same surface: `new(x, p)` (UKF also takes `UkfParams`),
`predict(&dyn MotionModel, dt)`, `update_linear(z, h, r)` / closure `update`,
and public `x` / `p`.

Motion models (`models/`): CV, CA, CTRV, CT. The IMM filter (`imm`) blends a
bank of model-conditioned filters by Bayesian mode probability.

## CKF vs UKF — when to choose which

Both are third-order-accurate deterministic-sample nonlinear filters and are
empirically interchangeable on linear and mildly nonlinear problems. Reach for
the **UKF** when you need the extra spread/prior flexibility its `(alpha,
beta, kappa)` knobs give and you are willing to tune them. Reach for the
**CKF** when you want a *parameter-free* filter that cannot mis-tune: its `2n`
points are equally and positively weighted, so the predicted covariance stays
positive-definite under the standard rule — useful for thresh's corner cases
(ballistic re-entry's wide dynamic range, hard-maneuvering UAVs, OTHR
ionospheric refraction) where a poorly-chosen UKF `kappa` can drive the
covariance indefinite. When in doubt on aerospace targets, prefer CKF.

CKF reference: I. Arasaratnam and S. Haykin, "Cubature Kalman Filters", *IEEE
Transactions on Automatic Control*, vol. 54, no. 6, pp. 1254–1269, June 2009.

## IMM leaf filter

The IMM bank's model-conditioned leaf is selectable — EKF, UKF, or CKF — via
the `ImmLeafKind` chosen at construction:

```rust
use thresh_filter::imm::{ImmConfig, ImmFilter, ImmLeafKind};

// Default leaf is EKF — behaviour identical to the pre-pluggable bank.
let ekf_bank = ImmFilter::new(ImmConfig::cv_ca(5.0, 1.0), &x0, &p0);

// Or pick the leaf kind explicitly (every mode uses the same kind).
let ckf_bank =
    ImmFilter::with_leaf_kind(ImmConfig::cv_ca(5.0, 1.0), ImmLeafKind::Ckf, &x0, &p0);
```

The kind is owned by the `ImmFilter` instance and also drives the
common-space measurement-update leaf. `ImmFilter::new` defaults to
`ImmLeafKind::Ekf`, so existing callers and the `thresh-tracker` IMM path
are unaffected.
