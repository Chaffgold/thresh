//! Adaptive Dormand–Prince RK5(4) integration on the time-aware
//! acceleration seam, with PI step control and grid-exact output sampling.
//!
//! The 7-stage first-same-as-last (FSAL) embedded pair of Dormand & Prince
//! (1980) driven by the PI ("Lund-stabilized") step-size controller of
//! Hairer's `dopri5`, plus **grid-exact output sampling by step clamping**:
//! when the controller's next natural step would cross a caller-requested
//! sample time, the step is shortened to land on it exactly, so every
//! returned sample is an integration-accurate state at the requested offset
//! — no interpolant (design Decision 6 of `orbit-propagation-fidelity`).
//! The dopri5 dense-output quartic (Shampine 1986) is the recorded upgrade
//! path if clamping-induced step churn ever shows up in profiling.
//!
//! This module is the first consumer of the **time-aware closure seam**
//! (design Decision 1): accelerations are
//! `Fn(f64, &Vector3<f64>, &Vector3<f64>) -> Vector3<f64>` — seconds past
//! the arc's epoch, position (m), velocity (m/s) → acceleration (m/s²) —
//! with the arc's absolute [`crate::time::Epoch`] bound at closure
//! construction (see [`TimeAwareAccel`]). The fixed-step RK4 in
//! [`integrate`](super::integrate) and its position/velocity-only seam are
//! untouched, as is every existing consumer of them.
//!
//! Discontinuous right-hand sides (e.g. the cylindrical Earth-shadow step
//! of the SRP force) are handled by the controller's natural step
//! rejection, bounded below by the configurable step floor
//! ([`DpConfig::min_step_s`]); a conical-penumbra shadow model is the
//! recorded upgrade path on the force side.
//!
//! # Coefficient provenance
//!
//! The tableau was fetched this session (2026-07-16) from two authoritative
//! reproductions of Dormand & Prince (1980), "A family of embedded
//! Runge-Kutta formulae", J. Comput. Appl. Math. 6(1):19–26, Table 2
//! (RK5(4)7M), and cross-checked entry by entry:
//!
//! - Hairer's `dopri5.f` (`unige.ch/~hairer/prog/nonstiff/dopri5.f`,
//!   `CDOPRI` block) — also the source of the PI controller constants;
//! - scipy v1.14.1 `scipy/integrate/_ivp/rk.py` (`class RK45`).
//!
//! The error row follows `dopri5.f`'s sign convention `E = b − b̂`
//! (5th-order minus embedded 4th-order weights); scipy stores the exact
//! negation. The embedded 4th-order weights `b̂` were additionally
//! cross-checked against the Butcher tableau on the Dormand–Prince
//! Wikipedia page (fetched this session; see the
//! `error_row_matches_published_fourth_order_weights` test). Checksum
//! tripwire tests pin the transcription per the `frames/nutation1980.rs`
//! pattern.

use nalgebra::Vector3;
use serde::{Deserialize, Serialize};

use crate::orbital::state::{ElementError, Frame, OrbitalElements, OrbitalState};

// ---------------------------------------------------------------------------
// The time-aware acceleration seam (design Decision 1)
// ---------------------------------------------------------------------------

/// Time-aware acceleration closure seam:
/// `accel(t_s, &position_m, &velocity_m_s) -> acceleration_m_s2`, where
/// `t_s` is **seconds past the arc's epoch**. The arc's absolute
/// [`crate::time::Epoch`] is bound at closure construction (design
/// Decision 1 of `orbit-propagation-fidelity`), so force legs that need
/// absolute time (Earth-fixed rotations, Sun/Moon positions) evaluate at
/// `epoch + t_s` inside the closure.
///
/// Blanket-implemented for every matching `Fn`, so any closure of the right
/// shape is already a `TimeAwareAccel`; the `ForceModelConfig` builder
/// produces closures matching this seam. The existing time-free seam
/// `Fn(&Vector3, &Vector3) -> Vector3` consumed by
/// [`rk4_step`](super::integrate::rk4_step) is unaffected.
pub trait TimeAwareAccel: Fn(f64, &Vector3<f64>, &Vector3<f64>) -> Vector3<f64> {}

impl<F> TimeAwareAccel for F where F: Fn(f64, &Vector3<f64>, &Vector3<f64>) -> Vector3<f64> {}

// ---------------------------------------------------------------------------
// Dormand–Prince RK5(4) tableau (fetched; see module-level provenance)
// ---------------------------------------------------------------------------
//
// Values transcribed from the fetched `dopri5.f` `CDOPRI` block and scipy
// v1.14.1 `rk.py` `RK45` (identical rationals in both). Stage 7 is the FSAL
// stage: its coupling row equals the solution weights `DP_B`, it is
// evaluated at the accepted 5th-order solution, and it becomes `k₁` of the
// next step.

/// Stage time fractions c₁..c₇ (c₇ = 1 is the FSAL stage at the step end).
const DP_C: [f64; 7] = [0.0, 0.2, 0.3, 0.8, 8.0 / 9.0, 1.0, 1.0];

/// Coupling row a₂ⱼ.
const DP_A2: [f64; 1] = [0.2];
/// Coupling row a₃ⱼ.
const DP_A3: [f64; 2] = [3.0 / 40.0, 9.0 / 40.0];
/// Coupling row a₄ⱼ.
const DP_A4: [f64; 3] = [44.0 / 45.0, -56.0 / 15.0, 32.0 / 9.0];
/// Coupling row a₅ⱼ.
const DP_A5: [f64; 4] = [
    19372.0 / 6561.0,
    -25360.0 / 2187.0,
    64448.0 / 6561.0,
    -212.0 / 729.0,
];
/// Coupling row a₆ⱼ.
const DP_A6: [f64; 5] = [
    9017.0 / 3168.0,
    -355.0 / 33.0,
    46732.0 / 5247.0,
    49.0 / 176.0,
    -5103.0 / 18656.0,
];
/// Coupling rows a₂ⱼ..a₆ⱼ, table-driven for the stage loop.
const DP_A: [&[f64]; 5] = [&DP_A2, &DP_A3, &DP_A4, &DP_A5, &DP_A6];

/// 5th-order solution weights b₁..b₆ (b₇ = 0), also the FSAL coupling row
/// a₇ⱼ in `dopri5.f` (A71..A76).
const DP_B: [f64; 6] = [
    35.0 / 384.0,
    0.0,
    500.0 / 1113.0,
    125.0 / 192.0,
    -2187.0 / 6784.0,
    11.0 / 84.0,
];

/// Error-estimate weights e₁..e₇ = b − b̂ (5th-order minus embedded
/// 4th-order weights), `dopri5.f` sign convention (E1..E7).
const DP_E: [f64; 7] = [
    71.0 / 57600.0,
    0.0,
    -71.0 / 16695.0,
    71.0 / 1920.0,
    -17253.0 / 339200.0,
    22.0 / 525.0,
    -1.0 / 40.0,
];

// ---------------------------------------------------------------------------
// PI step-control constants (fetched from `dopri5.f`; see provenance)
// ---------------------------------------------------------------------------

/// Safety factor on the predicted step (`SAFE` in `dopri5.f`).
const SAFETY: f64 = 0.9;
/// Minimum step-scale per accepted step, `FAC1`: `h_new/h ≥ 0.2`.
const STEP_SCALE_MIN: f64 = 0.2;
/// Maximum step-scale per accepted step, `FAC2`: `h_new/h ≤ 10`.
const STEP_SCALE_MAX: f64 = 10.0;
/// Lund-stabilization exponent (`BETA`, dopri5 default 0.04).
const PI_BETA: f64 = 0.04;
/// Error exponent `EXPO1 = 0.2 − 0.75·BETA` (= 0.17 at the default BETA;
/// 0.2 = 1/(order 4 error estimator + 1)).
const PI_EXPO1: f64 = 0.2 - 0.75 * PI_BETA;
/// Initial value and floor for the previous-error memory (`FACOLD`).
const PI_ERR_FLOOR: f64 = 1.0e-4;

// ---------------------------------------------------------------------------
// Configuration and results
// ---------------------------------------------------------------------------

/// Adaptive-step configuration for the Dormand–Prince integrator.
///
/// Tolerances follow the `dopri5` scaled-norm convention: a step is
/// accepted when the RMS over the 6 state components of
/// `error_i / (abs_tol + rel_tol·max(|y_i|, |y_new_i|))` is ≤ 1. The same
/// scalar pair applies to position (m) and velocity (m/s) components, like
/// scipy's scalar `atol`/`rtol`.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct DpConfig {
    /// Absolute tolerance (m for position components, m/s for velocity
    /// components). Default `1e-9`.
    pub abs_tol: f64,
    /// Relative tolerance (dimensionless). Default `1e-9`.
    pub rel_tol: f64,
    /// First trial step (s), clamped into `[min_step_s, max_step_s]`.
    /// Default 10 s; the controller recovers from a poor guess within a few
    /// steps.
    pub initial_step_s: f64,
    /// Step floor (s). A trial step at or below the floor is **accepted
    /// unconditionally**, error estimate notwithstanding — this bounds the
    /// step churn a discontinuous force (shadow-boundary crossing) can
    /// cause (design Decision 6 risk note). Grid clamping may still take
    /// steps shorter than the floor to land exactly on a requested sample
    /// time. Default 1 ms.
    pub min_step_s: f64,
    /// Step ceiling (s). Default 300 s.
    pub max_step_s: f64,
}

impl Default for DpConfig {
    fn default() -> Self {
        Self {
            abs_tol: 1.0e-9,
            rel_tol: 1.0e-9,
            initial_step_s: 10.0,
            min_step_s: 1.0e-3,
            max_step_s: 300.0,
        }
    }
}

/// One grid-exact output sample: the integration state at exactly the
/// requested time offset.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct DpSample {
    /// Seconds past the arc start — **bitwise** the requested offset.
    pub offset_s: f64,
    /// Position (m).
    pub position: Vector3<f64>,
    /// Velocity (m/s).
    pub velocity: Vector3<f64>,
}

/// Result of an adaptive propagation: the requested samples plus the
/// accepted-step record (the churn diagnostic the design's eclipse risk
/// note calls for) and the rejection count.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DpSolution {
    /// One sample per requested offset, in request order.
    pub samples: Vec<DpSample>,
    /// End time (seconds past the arc start) of every accepted step, in
    /// order. Every requested offset > 0 appears here exactly once: samples
    /// are produced by steps ending on them, never by interpolation.
    pub accepted_step_offsets: Vec<f64>,
    /// Number of rejected trial steps.
    pub rejected_steps: usize,
}

// ---------------------------------------------------------------------------
// Internals: stage evaluation, error norm, PI controller
// ---------------------------------------------------------------------------

/// Derivative of the second-order state `(r, v)`: `dr = v` at the stage
/// point, `dv` = acceleration there.
#[derive(Debug, Clone, Copy)]
struct Deriv {
    dr: Vector3<f64>,
    dv: Vector3<f64>,
}

/// State advanced by `h·Σ wⱼ·kⱼ` over the leading `weights.len()` stages.
fn weighted_state(
    position: &Vector3<f64>,
    velocity: &Vector3<f64>,
    h: f64,
    weights: &[f64],
    k: &[Deriv; 7],
) -> (Vector3<f64>, Vector3<f64>) {
    let mut dr = Vector3::zeros();
    let mut dv = Vector3::zeros();
    for (w, ki) in weights.iter().zip(k.iter()) {
        dr += *w * ki.dr;
        dv += *w * ki.dv;
    }
    (position + h * dr, velocity + h * dv)
}

/// Scaled RMS error norm of the embedded difference `h·Σ eⱼ·kⱼ` (dopri5's
/// `ERR`): accept when ≤ 1.
fn error_norm(
    k: &[Deriv; 7],
    h: f64,
    before: (&Vector3<f64>, &Vector3<f64>),
    after: (&Vector3<f64>, &Vector3<f64>),
    config: &DpConfig,
) -> f64 {
    let mut err_r = Vector3::zeros();
    let mut err_v = Vector3::zeros();
    for (w, ki) in DP_E.iter().zip(k.iter()) {
        err_r += *w * ki.dr;
        err_v += *w * ki.dv;
    }
    let (r0, v0) = before;
    let (r1, v1) = after;
    let mut sum = 0.0;
    for i in 0..3 {
        let sc_r = config.abs_tol + config.rel_tol * r0[i].abs().max(r1[i].abs());
        let sc_v = config.abs_tol + config.rel_tol * v0[i].abs().max(v1[i].abs());
        sum += (h * err_r[i] / sc_r).powi(2) + (h * err_v[i] / sc_v).powi(2);
    }
    (sum / 6.0).sqrt()
}

/// PI ("Lund-stabilized") step-size controller, transcribed from the
/// fetched `dopri5.f` (see module provenance): `h_new = h / fac` with
/// `fac = clamp(err^EXPO1 / facold^BETA / SAFE, 1/FAC2, 1/FAC1)`, previous
/// accepted error memory `facold = max(err, 1e-4)`, no growth on the first
/// accepted step after a rejection, and shrink-only rejection scaling.
#[derive(Debug, Clone)]
struct PiController {
    facold: f64,
    just_rejected: bool,
}

impl PiController {
    fn new() -> Self {
        Self {
            facold: PI_ERR_FLOOR,
            just_rejected: false,
        }
    }

    /// Next step size after an accepted step of size `h` with error `err`.
    fn accept(&mut self, h: f64, err: f64) -> f64 {
        let fac11 = err.powf(PI_EXPO1);
        let fac = (fac11 / self.facold.powf(PI_BETA) / SAFETY)
            .clamp(1.0 / STEP_SCALE_MAX, 1.0 / STEP_SCALE_MIN);
        let mut h_new = h / fac;
        if self.just_rejected {
            // dopri5: HNEW = MIN(HNEW, H) right after a rejection.
            h_new = h_new.min(h);
        }
        self.facold = err.max(PI_ERR_FLOOR);
        self.just_rejected = false;
        h_new
    }

    /// Next (smaller) step size after a rejected step of size `h` with
    /// error `err` (> 1). dopri5: `HNEW = H / MIN(1/FAC1, err^EXPO1/SAFE)`.
    fn reject(&mut self, h: f64, err: f64) -> f64 {
        self.just_rejected = true;
        h / (err.powf(PI_EXPO1) / SAFETY).min(1.0 / STEP_SCALE_MIN)
    }
}

// ---------------------------------------------------------------------------
// Stepping loop
// ---------------------------------------------------------------------------

/// Mutable integration state threaded through the phase helpers.
struct StepCtx<'a, F: TimeAwareAccel> {
    config: &'a DpConfig,
    accel: &'a F,
    /// Seconds past the arc start.
    t: f64,
    position: Vector3<f64>,
    velocity: Vector3<f64>,
    /// FSAL derivative at `(t, position, velocity)`.
    k1: Deriv,
    /// Controller's current natural step proposal (s).
    h: f64,
    controller: PiController,
    accepted_step_offsets: Vec<f64>,
    rejected_steps: usize,
}

/// One evaluated trial step of size `h` from the current context state.
struct Trial {
    k: [Deriv; 7],
    position: Vector3<f64>,
    velocity: Vector3<f64>,
    err: f64,
}

/// Evaluate the seven FSAL stages and the embedded error estimate for a
/// trial step of size `h`, without committing anything.
fn trial_step<F: TimeAwareAccel>(ctx: &StepCtx<'_, F>, h: f64) -> Trial {
    let mut k = [ctx.k1; 7];
    for (row_idx, row) in DP_A.iter().enumerate() {
        let (r_s, v_s) = weighted_state(&ctx.position, &ctx.velocity, h, row, &k);
        let a_s = (ctx.accel)(ctx.t + DP_C[row_idx + 1] * h, &r_s, &v_s);
        k[row_idx + 1] = Deriv { dr: v_s, dv: a_s };
    }
    // 5th-order solution, then the FSAL stage evaluated exactly there.
    let (r_new, v_new) = weighted_state(&ctx.position, &ctx.velocity, h, &DP_B, &k);
    let a_new = (ctx.accel)(ctx.t + h, &r_new, &v_new);
    k[6] = Deriv {
        dr: v_new,
        dv: a_new,
    };
    let err = error_norm(
        &k,
        h,
        (&ctx.position, &ctx.velocity),
        (&r_new, &v_new),
        ctx.config,
    );
    Trial {
        k,
        position: r_new,
        velocity: v_new,
        err,
    }
}

/// Commit an accepted trial: advance time, adopt the 5th-order state, reuse
/// the FSAL stage as the next `k₁`, and ask the controller for the next
/// natural step.
///
/// Precondition: `landing` is `Some(target)` exactly when `h_try` was
/// clamped to `target − t` by [`advance_to_target`]; assigning `target`
/// directly (rather than `t + h_try`, its value up to one ulp) is what
/// makes grid sampling exact.
fn accept_trial<F: TimeAwareAccel>(
    ctx: &mut StepCtx<'_, F>,
    trial: &Trial,
    h_try: f64,
    landing: Option<f64>,
) {
    let new_t = landing.unwrap_or(ctx.t + h_try);
    assert!(
        new_t > ctx.t,
        "step {h_try} s is below f64 resolution at t = {} s; raise min_step_s",
        ctx.t
    );
    ctx.t = new_t;
    ctx.position = trial.position;
    ctx.velocity = trial.velocity;
    ctx.k1 = trial.k[6];
    ctx.accepted_step_offsets.push(ctx.t);
    ctx.h = ctx
        .controller
        .accept(h_try, trial.err)
        .clamp(ctx.config.min_step_s, ctx.config.max_step_s);
}

/// Advance the context to exactly `target` seconds past the arc start,
/// clamping the final step to land on it (grid-exact sampling, design
/// Decision 6). Steps at or below the floor are accepted unconditionally
/// (see [`DpConfig::min_step_s`]).
fn advance_to_target<F: TimeAwareAccel>(ctx: &mut StepCtx<'_, F>, target: f64) {
    while ctx.t < target {
        let remaining = target - ctx.t;
        let lands = ctx.h >= remaining;
        let h_try = if lands { remaining } else { ctx.h };
        let trial = trial_step(ctx, h_try);
        if trial.err <= 1.0 || h_try <= ctx.config.min_step_s {
            accept_trial(ctx, &trial, h_try, lands.then_some(target));
        } else {
            ctx.rejected_steps += 1;
            ctx.h = ctx
                .controller
                .reject(h_try, trial.err)
                .max(ctx.config.min_step_s);
        }
    }
}

/// Panic with a clear message on an invalid configuration.
fn validate_config(config: &DpConfig) {
    assert!(
        config.abs_tol > 0.0 && config.abs_tol.is_finite(),
        "abs_tol must be finite and > 0, got {}",
        config.abs_tol
    );
    assert!(
        config.rel_tol >= 0.0 && config.rel_tol.is_finite(),
        "rel_tol must be finite and >= 0, got {}",
        config.rel_tol
    );
    assert!(
        config.min_step_s > 0.0 && config.min_step_s.is_finite(),
        "min_step_s must be finite and > 0, got {}",
        config.min_step_s
    );
    assert!(
        config.max_step_s >= config.min_step_s && config.max_step_s.is_finite(),
        "max_step_s must be finite and >= min_step_s, got {}",
        config.max_step_s
    );
    assert!(
        config.initial_step_s > 0.0 && config.initial_step_s.is_finite(),
        "initial_step_s must be finite and > 0, got {}",
        config.initial_step_s
    );
}

/// Panic with a clear message on an invalid sample grid.
fn validate_grid(sample_offsets_s: &[f64]) {
    let mut previous = -f64::INFINITY;
    for &offset in sample_offsets_s {
        assert!(
            offset.is_finite() && offset >= 0.0,
            "sample offsets must be finite and >= 0, got {offset}"
        );
        assert!(
            offset > previous,
            "sample offsets must be strictly increasing, got {offset} after {previous}"
        );
        previous = offset;
    }
}

// ---------------------------------------------------------------------------
// Public entry points
// ---------------------------------------------------------------------------

/// Propagate `(position, velocity)` under the time-aware acceleration
/// closure, returning grid-exact samples at `sample_offsets_s` (seconds
/// past the arc start, strictly increasing, ≥ 0; an offset of exactly 0
/// returns the initial state).
///
/// Every sample is produced by an accepted integration step ending exactly
/// on the requested offset (step clamping, no interpolation), and
/// [`DpSample::offset_s`] carries the requested value bitwise. The
/// propagation is deterministic: identical inputs produce bitwise-identical
/// step sequences and samples.
///
/// Time discipline: `t = 0` is the arc's epoch; the closure receives
/// seconds past it (see [`TimeAwareAccel`]). For a frame- and
/// epoch-disciplined wrapper over [`OrbitalState`], use
/// [`propagate_orbital_state`].
///
/// # Panics
///
/// Panics when `config` is invalid (non-positive tolerances or steps,
/// `max_step_s < min_step_s`) or `sample_offsets_s` is not finite,
/// non-negative, and strictly increasing.
///
/// # Example
///
/// ```
/// use nalgebra::Vector3;
/// use thresh_core::orbital::dormand_prince::{DpConfig, propagate_on_grid};
/// use thresh_core::orbital::{GravityModel, two_body_acceleration};
///
/// let gravity = GravityModel::EARTH_WGS84;
/// let radius = gravity.equatorial_radius + 500_000.0;
/// let speed = (gravity.mu / radius).sqrt();
/// let solution = propagate_on_grid(
///     &Vector3::new(radius, 0.0, 0.0),
///     &Vector3::new(0.0, speed, 0.0),
///     &[60.0, 120.0],
///     &DpConfig::default(),
///     |_t: f64, r: &Vector3<f64>, _v: &Vector3<f64>| two_body_acceleration(r, &gravity),
/// );
/// assert_eq!(solution.samples[1].offset_s, 120.0);
/// ```
pub fn propagate_on_grid<F: TimeAwareAccel>(
    position: &Vector3<f64>,
    velocity: &Vector3<f64>,
    sample_offsets_s: &[f64],
    config: &DpConfig,
    accel: F,
) -> DpSolution {
    validate_config(config);
    validate_grid(sample_offsets_s);

    let k1 = Deriv {
        dr: *velocity,
        dv: accel(0.0, position, velocity),
    };
    let mut ctx = StepCtx {
        config,
        accel: &accel,
        t: 0.0,
        position: *position,
        velocity: *velocity,
        k1,
        h: config
            .initial_step_s
            .clamp(config.min_step_s, config.max_step_s),
        controller: PiController::new(),
        accepted_step_offsets: Vec::new(),
        rejected_steps: 0,
    };

    let mut samples = Vec::with_capacity(sample_offsets_s.len());
    for &offset in sample_offsets_s {
        advance_to_target(&mut ctx, offset);
        samples.push(DpSample {
            offset_s: offset,
            position: ctx.position,
            velocity: ctx.velocity,
        });
    }
    DpSolution {
        samples,
        accepted_step_offsets: ctx.accepted_step_offsets,
        rejected_steps: ctx.rejected_steps,
    }
}

/// Epoch- and frame-disciplined propagation entry point: propagate a
/// GCRF-tagged [`OrbitalState`] forward by `dt_s` seconds of true elapsed
/// (TAI) time under the time-aware acceleration closure.
///
/// The result is a `Cartesian`-represented state tagged [`Frame::Gcrf`]
/// with the epoch advanced **exactly** by `dt_s` (`state.epoch + dt_s`,
/// exact for nanosecond-representable `dt_s` per the
/// [`crate::time::Epoch`] arithmetic contract) and `mu` carried over
/// unchanged. The dynamics are entirely the closure's: `state.mu` rides
/// along for element conversions only, and the closure receives seconds
/// past `state.epoch` (bind the epoch at closure construction — design
/// Decision 1).
///
/// # Errors
///
/// - [`ElementError::FrameMismatch`] when `state` is not GCRF-tagged
///   (`expected` = [`Frame::Gcrf`], `found` = the state's tag). High-
///   fidelity propagation integrates in GCRF by contract; convert
///   explicitly first via [`OrbitalState::to_frame`] — never implicitly
///   here.
/// - [`ElementError::RequiresSgp4`] for `Tle`-represented states.
///
/// # Panics
///
/// Panics when `dt_s` is negative or non-finite, or `config` is invalid
/// (see [`propagate_on_grid`]).
pub fn propagate_orbital_state<F: TimeAwareAccel>(
    state: &OrbitalState,
    dt_s: f64,
    config: &DpConfig,
    accel: F,
) -> Result<OrbitalState, ElementError> {
    if state.frame != Frame::Gcrf {
        return Err(ElementError::FrameMismatch {
            expected: Frame::Gcrf,
            found: state.frame,
        });
    }
    let (position, velocity) = state.as_cartesian()?;
    let solution = propagate_on_grid(&position, &velocity, &[dt_s], config, accel);
    let sample = solution
        .samples
        .last()
        .expect("one requested offset yields one sample");
    Ok(OrbitalState {
        elements: OrbitalElements::Cartesian {
            position: sample.position,
            velocity: sample.velocity,
        },
        mu: state.mu,
        frame: Frame::Gcrf,
        epoch: state.epoch + dt_s,
    })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use std::f64::consts::TAU;

    use super::*;
    use crate::orbital::gravity::{GravityModel, j2_acceleration, two_body_acceleration};
    use crate::time::Epoch;

    const EARTH: GravityModel = GravityModel::EARTH_WGS84;

    /// Circular LEO at 500 km: `(position, velocity, period_s)`.
    fn circular_leo() -> (Vector3<f64>, Vector3<f64>, f64) {
        let radius = EARTH.equatorial_radius + 500_000.0;
        let speed = (EARTH.mu / radius).sqrt();
        let period = TAU * radius / speed;
        (
            Vector3::new(radius, 0.0, 0.0),
            Vector3::new(0.0, speed, 0.0),
            period,
        )
    }

    fn two_body(_t: f64, r: &Vector3<f64>, _v: &Vector3<f64>) -> Vector3<f64> {
        two_body_acceleration(r, &EARTH)
    }

    // ── Tableau transcription tripwires (module provenance) ─────────────

    /// Nutation-table-pattern checksums: exact sequential-sum values
    /// computed this session from the fetched rationals (Python, identical
    /// left-to-right IEEE-754 summation order as `iter().sum()`). Any
    /// dropped, duplicated, or mistyped coefficient moves at least one sum.
    #[test]
    fn tableau_checksums_match_fetched_source() {
        let c_sum: f64 = DP_C.iter().sum();
        let a_sum: f64 = DP_A.iter().flat_map(|row| row.iter()).sum();
        let b_sum: f64 = DP_B.iter().sum();
        let e_sum: f64 = DP_E.iter().sum();
        assert_eq!(c_sum, 4.188_888_888_888_889);
        assert_eq!(a_sum, 3.188_888_888_888_89);
        assert_eq!(b_sum, 0.999_999_999_999_999_8);
        assert_eq!(e_sum, 0.0);
    }

    /// Structural order conditions — independent of any transcription: each
    /// coupling row must sum to its stage time, and the 5th-order weights
    /// must satisfy the quadrature conditions Σ bⱼcⱼ^q = 1/(q+1), q ≤ 4.
    /// A single wrong digit anywhere in the tableau breaks one of these.
    #[test]
    fn tableau_satisfies_order_conditions() {
        for (row, c) in DP_A.iter().zip(&DP_C[1..6]) {
            let row_sum: f64 = row.iter().sum();
            assert!(
                (row_sum - c).abs() < 1e-15,
                "row sum {row_sum} must equal stage time {c}"
            );
        }
        for q in 0..5_i32 {
            let s: f64 = DP_B.iter().zip(&DP_C).map(|(b, c)| b * c.powi(q)).sum();
            let target = 1.0 / f64::from(q + 1);
            assert!(
                (s - target).abs() < 1e-15,
                "quadrature condition q={q}: {s} vs {target}"
            );
        }
    }

    /// Cross-source tripwire: the error row must equal `b − b̂` with the
    /// embedded 4th-order weights b̂ as published in the Dormand–Prince
    /// Butcher tableau (Wikipedia, fetched this session, agreeing with
    /// Dormand & Prince (1980) Table 2) — a third source independent of
    /// the `dopri5.f`/scipy pair the row was transcribed from.
    #[test]
    fn error_row_matches_published_fourth_order_weights() {
        const B_HAT: [f64; 7] = [
            5179.0 / 57600.0,
            0.0,
            7571.0 / 16695.0,
            393.0 / 640.0,
            -92097.0 / 339200.0,
            187.0 / 2100.0,
            1.0 / 40.0,
        ];
        for (j, (b_hat, e)) in B_HAT.iter().zip(DP_E.iter()).enumerate() {
            let b = DP_B.get(j).copied().unwrap_or(0.0);
            assert!(
                ((b - b_hat) - e).abs() < 1e-16,
                "e_{j} must equal b_{j} − b̂_{j}"
            );
        }
    }

    // ── PI controller unit behavior ──────────────────────────────────────

    #[test]
    fn pi_controller_scales_are_clamped_and_monotonic() {
        // Zero error (exact integration): maximum growth, h_new = 10·h.
        let mut c = PiController::new();
        assert_eq!(c.accept(10.0, 0.0), 100.0);

        // Larger error ⇒ smaller next step; err = 1 already shrinks.
        let mut c = PiController::new();
        let at_limit = c.accept(10.0, 1.0);
        let mut c = PiController::new();
        let over = c.accept(10.0, 0.5);
        assert!(at_limit < 10.0, "err at the limit must shrink: {at_limit}");
        assert!(over > at_limit, "smaller err must give a larger step");

        // Rejection always shrinks, never below the 0.2 scale clamp.
        let mut c = PiController::new();
        let rejected = c.reject(10.0, 1.0e9);
        assert_eq!(rejected, 2.0, "huge error clamps at h/5");
        // …and the first acceptance after a rejection never grows the step.
        let after = c.accept(2.0, 1.0e-12);
        assert!(after <= 2.0, "no growth right after a rejection: {after}");
    }

    // ── Time-awareness of the seam ───────────────────────────────────────

    /// A cubic-in-time right-hand side (`a_x = 6t` ⇒ `r_x = t³`) is
    /// integrated exactly by both embedded orders, so the result checks the
    /// `DP_C` stage times and `DP_B` weights jointly: a wrong stage time
    /// would break exactness.
    #[test]
    fn time_dependent_cubic_is_integrated_exactly() {
        let solution = propagate_on_grid(
            &Vector3::zeros(),
            &Vector3::zeros(),
            &[50.0],
            &DpConfig::default(),
            |t: f64, _r: &Vector3<f64>, _v: &Vector3<f64>| Vector3::new(6.0 * t, 0.0, 0.0),
        );
        let sample = &solution.samples[0];
        assert!(
            (sample.position.x - 50.0_f64.powi(3)).abs() < 1e-6,
            "r_x = t³ must be exact to rounding, got {}",
            sample.position.x
        );
        assert!(
            (sample.velocity.x - 3.0 * 50.0_f64.powi(2)).abs() < 1e-8,
            "v_x = 3t² must be exact to rounding, got {}",
            sample.velocity.x
        );
    }

    // ── Convergence (spec: "Convergence at the design order on a Kepler
    //    orbit") ─────────────────────────────────────────────────────────

    /// One circular-LEO period at two tolerance decades: the tighter
    /// tolerance must beat the looser one, and both must sit inside their
    /// tolerance-implied envelopes (measured on this implementation and
    /// recorded with ~30× margin; see the assertion messages).
    #[test]
    fn kepler_orbit_converges_across_two_tolerance_decades() {
        let (r0, v0, period) = circular_leo();
        let mut errors = Vec::new();
        for tol in [1.0e-7, 1.0e-9] {
            let config = DpConfig {
                abs_tol: tol,
                rel_tol: tol,
                ..DpConfig::default()
            };
            let solution = propagate_on_grid(&r0, &v0, &[period], &config, two_body);
            // After exactly one period the analytic solution is the
            // initial state.
            errors.push((solution.samples[0].position - r0).norm());
        }
        let (loose, tight) = (errors[0], errors[1]);
        assert!(
            tight < loose,
            "tighter tolerance must reduce global error: {tight:.3e} vs {loose:.3e}"
        );
        assert!(
            loose / tight > 10.0,
            "two tolerance decades must buy at least a decade of error: \
             {loose:.3e} / {tight:.3e}"
        );
        // Tolerance-implied envelopes: measured 6.6 m (1e-7) and 2.1e-2 m
        // (1e-9) on this implementation (ratio 322), asserted at ~8× margin.
        assert!(loose < 50.0, "1e-7 envelope exceeded: {loose:.3e} m");
        assert!(tight < 0.2, "1e-9 envelope exceeded: {tight:.3e} m");
    }

    // ── Determinism (spec: "Deterministic adaptive stepping") ────────────

    #[test]
    fn repeated_runs_are_bitwise_identical() {
        let (r0, v0, _) = circular_leo();
        let offsets = [123.456, 1000.0, 2345.678, 5000.0];
        let config = DpConfig::default();
        let j2 = |_t: f64, r: &Vector3<f64>, _v: &Vector3<f64>| j2_acceleration(r, &EARTH);

        let a = propagate_on_grid(&r0, &v0, &offsets, &config, j2);
        let b = propagate_on_grid(&r0, &v0, &offsets, &config, j2);

        assert_eq!(a.rejected_steps, b.rejected_steps);
        let steps_a: Vec<u64> = a
            .accepted_step_offsets
            .iter()
            .map(|t| t.to_bits())
            .collect();
        let steps_b: Vec<u64> = b
            .accepted_step_offsets
            .iter()
            .map(|t| t.to_bits())
            .collect();
        assert_eq!(
            steps_a, steps_b,
            "accepted-step sequence must be bitwise identical"
        );
        for (sa, sb) in a.samples.iter().zip(&b.samples) {
            for i in 0..3 {
                assert_eq!(sa.position[i].to_bits(), sb.position[i].to_bits());
                assert_eq!(sa.velocity[i].to_bits(), sb.velocity[i].to_bits());
            }
        }
    }

    // ── Grid-exact sampling (spec: "Samples land exactly on the requested
    //    grid") ──────────────────────────────────────────────────────────

    #[test]
    fn samples_land_exactly_on_the_requested_grid() {
        let (r0, v0, _) = circular_leo();
        // Deliberately awkward offsets: nothing the natural step sequence
        // would hit on its own.
        let offsets = [0.0, 97.3, 300.0, 543.21];
        let solution = propagate_on_grid(&r0, &v0, &offsets, &DpConfig::default(), two_body);

        assert_eq!(solution.samples.len(), offsets.len());
        for (sample, requested) in solution.samples.iter().zip(&offsets) {
            assert_eq!(
                sample.offset_s.to_bits(),
                requested.to_bits(),
                "sample must carry the requested offset bitwise"
            );
        }
        // Each nonzero sample time is the end of an accepted integration
        // step — produced by integration, not interpolation.
        for &requested in &offsets[1..] {
            assert!(
                solution.accepted_step_offsets.contains(&requested),
                "no accepted step ends at {requested}"
            );
        }
    }

    /// Extending the grid must not change the states delivered at the
    /// shared prefix: sampling is step clamping, so the step history up to
    /// a common sample time is identical (would fail under interpolation
    /// against a different step sequence).
    #[test]
    fn grid_prefix_yields_bitwise_identical_states() {
        let (r0, v0, _) = circular_leo();
        let config = DpConfig::default();
        let short = propagate_on_grid(&r0, &v0, &[40.0, 97.25], &config, two_body);
        let long = propagate_on_grid(&r0, &v0, &[40.0, 97.25, 800.0, 2000.0], &config, two_body);

        for (a, b) in short.samples.iter().zip(&long.samples) {
            assert_eq!(a.offset_s.to_bits(), b.offset_s.to_bits());
            for i in 0..3 {
                assert_eq!(a.position[i].to_bits(), b.position[i].to_bits());
                assert_eq!(a.velocity[i].to_bits(), b.velocity[i].to_bits());
            }
        }
    }

    // ── Step clamps ──────────────────────────────────────────────────────

    #[test]
    fn accepted_steps_respect_the_max_step_ceiling() {
        let (r0, v0, period) = circular_leo();
        let config = DpConfig {
            abs_tol: 1.0e-6,
            rel_tol: 1.0e-6,
            ..DpConfig::default()
        };
        let solution = propagate_on_grid(&r0, &v0, &[period], &config, two_body);
        let mut previous = 0.0;
        for &end in &solution.accepted_step_offsets {
            assert!(
                end - previous <= config.max_step_s + 1e-9,
                "step {} exceeds the ceiling",
                end - previous
            );
            previous = end;
        }
    }

    /// With tolerances no step size can satisfy, the floor keeps the
    /// integration progressing: steps at `min_step_s` are accepted
    /// unconditionally (the design's bounded-churn guarantee for
    /// discontinuous forces).
    #[test]
    fn min_step_floor_forces_progress() {
        let (r0, v0, _) = circular_leo();
        let config = DpConfig {
            abs_tol: 1.0e-16,
            rel_tol: 1.0e-16,
            min_step_s: 5.0,
            ..DpConfig::default()
        };
        let solution = propagate_on_grid(&r0, &v0, &[60.0], &config, two_body);
        assert!(solution.rejected_steps > 0, "controller must have engaged");
        assert!(solution.samples[0].position.norm().is_finite());
        assert_eq!(solution.samples[0].offset_s, 60.0);
    }

    // ── Energy drift (task 3.3) ──────────────────────────────────────────

    /// Energy drift over one eccentric-LEO period at the default 1e-9
    /// tolerances, measured at 1.59e-9 relative on this implementation;
    /// asserted with ~60× margin. (The fixed-step RK4 seam's equivalent
    /// test bounds its drift at 1e-8 over one circular LEO period at a
    /// 10 s step — see
    /// `integrate::tests::rk4_step_conserves_two_body_energy_over_one_orbit`.)
    #[test]
    fn energy_drift_over_one_leo_period_is_bounded() {
        let sma = EARTH.equatorial_radius + 600_000.0;
        let ecc = 0.02;
        let r_p = sma * (1.0 - ecc);
        let v_p = (EARTH.mu * (2.0 / r_p - 1.0 / sma)).sqrt();
        let r0 = Vector3::new(r_p, 0.0, 0.0);
        let v0 = Vector3::new(0.0, v_p, 0.0);
        let period = TAU * (sma.powi(3) / EARTH.mu).sqrt();

        let energy =
            |r: &Vector3<f64>, v: &Vector3<f64>| 0.5 * v.norm_squared() - EARTH.mu / r.norm();
        let initial = energy(&r0, &v0);

        let solution = propagate_on_grid(&r0, &v0, &[period], &DpConfig::default(), two_body);
        let sample = &solution.samples[0];
        let drift = (energy(&sample.position, &sample.velocity) - initial).abs() / initial.abs();
        assert!(
            drift < 1e-7,
            "relative energy drift {drift:.3e} over one period"
        );
    }

    // ── Epoch/frame discipline (spec: "Propagation preserves frame and
    //    epoch discipline") ──────────────────────────────────────────────

    fn gcrf_leo_state(epoch: Epoch) -> OrbitalState {
        let (position, velocity, _) = circular_leo();
        OrbitalState {
            elements: OrbitalElements::Cartesian { position, velocity },
            mu: EARTH.mu,
            frame: Frame::Gcrf,
            epoch,
        }
    }

    #[test]
    fn propagation_preserves_frame_and_epoch() {
        let epoch = Epoch::from_gregorian_utc(2026, 1, 1, 0, 0, 0, 0);
        let state = gcrf_leo_state(epoch);
        let dt_s = 600.0;
        let config = DpConfig::default();

        let out = propagate_orbital_state(&state, dt_s, &config, two_body).unwrap();

        // GCRF in, GCRF out; epoch advanced exactly by Δt; mu unchanged.
        assert_eq!(out.frame, Frame::Gcrf);
        assert_eq!(out.epoch, epoch + dt_s, "epoch must be exactly t₀ + Δt");
        assert_eq!(out.epoch - state.epoch, dt_s);
        assert_eq!(out.mu.to_bits(), state.mu.to_bits());

        // The elements are exactly the raw integrator's output (the wrapper
        // adds discipline, not different numbers).
        let (r0, v0) = state.as_cartesian().unwrap();
        let raw = propagate_on_grid(&r0, &v0, &[dt_s], &config, two_body);
        let (r_out, v_out) = out.as_cartesian().unwrap();
        for i in 0..3 {
            assert_eq!(r_out[i].to_bits(), raw.samples[0].position[i].to_bits());
            assert_eq!(v_out[i].to_bits(), raw.samples[0].velocity[i].to_bits());
        }
    }

    /// Non-GCRF input is a returned `FrameMismatch` naming both frames —
    /// never an implicit conversion — and TLE elements surface
    /// `RequiresSgp4` from the element conversion.
    #[test]
    fn propagation_rejects_non_gcrf_and_tle_states() {
        let epoch = Epoch::from_gregorian_utc(2026, 1, 1, 0, 0, 0, 0);
        let teme = OrbitalState {
            frame: Frame::Teme,
            ..gcrf_leo_state(epoch)
        };
        assert_eq!(
            propagate_orbital_state(&teme, 60.0, &DpConfig::default(), two_body),
            Err(ElementError::FrameMismatch {
                expected: Frame::Gcrf,
                found: Frame::Teme,
            })
        );

        let tle = OrbitalState {
            elements: OrbitalElements::Tle {
                line1: "1".to_string(),
                line2: "2".to_string(),
            },
            ..gcrf_leo_state(epoch)
        };
        assert_eq!(
            propagate_orbital_state(&tle, 60.0, &DpConfig::default(), two_body),
            Err(ElementError::RequiresSgp4)
        );
    }

    // ── Input validation ─────────────────────────────────────────────────

    #[test]
    #[should_panic(expected = "strictly increasing")]
    fn non_increasing_grid_is_rejected() {
        let (r0, v0, _) = circular_leo();
        let _ = propagate_on_grid(&r0, &v0, &[60.0, 60.0], &DpConfig::default(), two_body);
    }

    #[test]
    #[should_panic(expected = "finite and >= 0")]
    fn negative_offset_is_rejected() {
        let (r0, v0, _) = circular_leo();
        let _ = propagate_on_grid(&r0, &v0, &[-1.0], &DpConfig::default(), two_body);
    }

    #[test]
    #[should_panic(expected = "min_step_s")]
    fn zero_min_step_is_rejected() {
        let (r0, v0, _) = circular_leo();
        let config = DpConfig {
            min_step_s: 0.0,
            ..DpConfig::default()
        };
        let _ = propagate_on_grid(&r0, &v0, &[60.0], &config, two_body);
    }

    #[test]
    fn default_tolerances_are_1e9() {
        let config = DpConfig::default();
        assert_eq!(config.abs_tol, 1.0e-9);
        assert_eq!(config.rel_tol, 1.0e-9);
    }
}
