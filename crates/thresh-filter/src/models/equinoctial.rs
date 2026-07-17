//! Equinoctial-element orbital motion model for sigma-point filters
//! (element-space-orbital-filtering design Decisions 3, 5, 6).
//!
//! State vector `[a, h, k, p, q, λ]` — the stored direct/prograde equinoctial
//! set in `thresh_core::orbital::OrbitalElements::Equinoctial` field order
//! (`sma, h, k, p, q, mean_longitude`), metres and radians. The dynamics are
//! the Gauss variational equations of `thresh_core::orbital::gve` (two-body
//! secular `dλ/dt ⊇ √(µ/a³)` plus a J2 perturbation rotated into RSW),
//! sub-stepped with fixed RK4 inside `predict` — the same step convention as
//! the sibling Cartesian `KeplerJ2` model. Nonlinear: use the UKF/CKF
//! sigma-point machinery (or the EKF via the numeric Jacobian).
//!
//! # Unwrapped mean longitude (design Decision 3)
//!
//! The filter-state mean longitude `λ` is **continuous / unwrapped**: it grows
//! monotonically under the secular rate and is **never** reduced modulo 2π in
//! state space. Periodicity is owned entirely by the element→Cartesian
//! conversion (the trigonometric functions inside are 2π-periodic, and
//! `OrbitalState`'s `as_keplerian` reduces `mean_longitude` mod 2π when it
//! forms the anomaly). Because sigma points are generated around an unwrapped
//! mean and their spread is small relative to 2π, they stay contiguous, so
//! plain arithmetic sigma-point means and residuals remain correct with no
//! circular statistics and no UKF/CKF API change. `predict` therefore never
//! wraps `λ`, and the position measurement map (the only observation of `λ`)
//! goes through the periodic conversion. The `wrap_straddling_*` and
//! `long_horizon_*` tests prove this convention has no wrap artifact and does
//! not degrade as `λ` accumulates hundreds of radians (at `λ ~ 1e3 rad` the
//! f64 `sin`/`cos` argument-reduction loss is ~1e-13 rad — documented,
//! negligible for the demonstrated horizons).
//!
//! # Element-space process noise (design Decision 3 / task 3.4)
//!
//! The process noise follows the state-noise-compensation (SNC) framework of
//! N. Stacey & S. D'Amico, *"Analytical Process Noise Covariance Modeling
//! for Absolute and Relative Orbits"* (arXiv:2105.06516, Acta Astronautica
//! 2022, DOI 10.1016/j.actaastro.2022.01.020; PDF fetched 2026-07-16 from
//! <https://arxiv.org/pdf/2105.06516>). That paper models the unmodeled
//! acceleration `ε` as a zero-mean white Gaussian process with autocovariance
//! `E[ε(t)ε(τ)ᵀ] = Q̃ δ(t−τ)` (their Eq. 3), where `Q̃ ∈ R³ˣ³` is the
//! acceleration power spectral density, and gives the process-noise covariance
//! `Q_k = ∫ Φ(t_k,τ) Γ(τ) Q̃ Γ(τ)ᵀ Φ(t_k,τ)ᵀ dτ` (their Eq. 5) with the
//! process-noise mapping matrix `Γ = ∂ẋ/∂ε` (their Eq. 2). For the equinoctial
//! state `dxE/dt = G(xE)·d + [0,…,0,n]ᵀ`, `Γ` is exactly the Gauss variational
//! input matrix `G`. Over one sub-step, with `Φ ≈ I` and `Γ` frozen at the
//! reference elements, Eq. (5) reduces to the leading term
//! `Q ≈ Γ·(σ²I₃)·Γᵀ·Δt`, using the **inertial** acceleration PSD
//! `Q̃ᴵ = σ_accel²·I₃`. The sibling `KeplerJ2` uses the *same* physical PSD:
//! its continuous-white-noise-acceleration block is the closed-form of the
//! *same* Eq. (5) for the kinematic Cartesian state (the paper's Eq. 11), so
//! both models share one physical acceleration-PSD assumption — the fairness
//! requirement of the coast-gap demonstration (design Decision 5). Because
//! `Q̃ᴵ = σ²I₃` is isotropic, the inertial and RSW mappings coincide
//! (`Γᴵ Q̃ᴵ Γᴵᵀ = Γᴿ Q̃ᴿ Γᴿᵀ`), so the mapping is built from unit **inertial**
//! accelerations (no explicit RSW rotation needed). `Q` is state-independent
//! per `predict` interval (the `MotionModel` trait exposes only `dt`), so `G`
//! is evaluated once at the model's [`EquinoctialModel::reference_elements`] —
//! the standard nominal-trajectory SNC linearization.

use nalgebra::{DMatrix, DVector, Vector3};

use thresh_core::orbital::GravityModel;
use thresh_core::orbital::gve::{
    equinoctial_element_rates, j2_perturbation_acceleration, propagate_equinoctial,
};
use thresh_core::orbital::{Frame, OrbitalElements, OrbitalState};
use thresh_core::time::Epoch;

use crate::numeric::numeric_jacobian;
use crate::traits::MotionModel;

/// Fixed J2000.0 epoch for the throwaway [`OrbitalState`] built only to reuse
/// the crate's tested element→Cartesian conversions. Those conversions are
/// pure geometry — they read neither the epoch nor the frame tag — so this
/// value never affects any propagated element or mapped position (the same
/// convention as `thresh_core::orbital::gve`).
const CONVERSION_EPOCH_JD: f64 = 2_451_545.0;

/// Numeric-Jacobian per-column floor scales for the 6D element state
/// `[a, h, k, p, q, λ]`. `a` carries its own ~1e7 m magnitude, so its column
/// step tracks that; the dimensionless eccentricity/inclination elements and
/// the angle take a 1e-3 floor so a near-zero component (circular / equatorial
/// orbit, `λ ≈ 0`) still gets a meaningful finite-difference step. The
/// Jacobian exists only to satisfy [`MotionModel`] / feed an EKF — the
/// sigma-point filters never call it.
const ELEMENT_6D_SCALES: [f64; 6] = [1.0, 1e-3, 1e-3, 1e-3, 1e-3, 1e-3];

/// Equinoctial-element orbital motion model over the 6D state `[a, h, k, p,
/// q, λ]` (design Decisions 3 and 6).
///
/// `gravity` supplies `µ` (two-body secular rate and coefficient scaling) and
/// the J2 term (the perturbing acceleration) — parameterized, never a
/// hardcoded Earth constant, like `KeplerJ2`. `predict` sub-steps the interval
/// with fixed RK4 over the Gauss variational element rates (`max_step_s`
/// ceiling). The mean longitude is unwrapped (see the module docs).
pub struct EquinoctialModel {
    /// Central-body gravitational parameters (µ, J2, equatorial radius).
    pub gravity: GravityModel,
    /// RK4 sub-step ceiling (s); `predict` takes `ceil(dt / max_step_s)` equal
    /// sub-steps. Default 10.0 (matching `KeplerJ2`).
    pub max_step_s: f64,
    /// Isotropic unmodeled-acceleration white-noise PSD `σ_accel`
    /// (m/s²·Hz^-½), the same physical quantity as `KeplerJ2::sigma_accel`.
    /// Default 1e-3.
    pub sigma_accel: f64,
    /// Nominal elements at which the process-noise input matrix `G` is
    /// evaluated (the SNC nominal-trajectory linearization — see the module
    /// docs). Typically the filter's initial estimate.
    pub reference_elements: [f64; 6],
}

impl EquinoctialModel {
    /// Create a model for the given central body and nominal (reference)
    /// elements, with the defaults `max_step_s = 10.0` s and
    /// `sigma_accel = 1e-3` m/s²·Hz^-½.
    pub fn new(gravity: GravityModel, reference_elements: [f64; 6]) -> Self {
        Self {
            gravity,
            max_step_s: 10.0,
            sigma_accel: 1e-3,
            reference_elements,
        }
    }

    /// The J2-only inertial perturbation closure the GVE consume (design
    /// Decision 2): two-body is the analytic secular term inside the GVE and
    /// is never part of the perturbation input.
    fn j2_perturbation(&self) -> impl Fn(f64, &Vector3<f64>, &Vector3<f64>) -> Vector3<f64> {
        let gravity = self.gravity;
        move |_t, r, _v| j2_perturbation_acceleration(r, &gravity)
    }

    /// Map the element state to inertial Cartesian **position** (m) — the ECI
    /// position measurement map for position-measurement filters. Exactly the
    /// position component of the state's existing element→Cartesian conversion
    /// (spec: "Measurement mapping matches the conversion").
    pub fn position_measurement(&self, state: &DVector<f64>) -> DVector<f64> {
        equinoctial_position(&unpack(state), self.gravity.mu)
    }
}

/// Unpack the leading six components of `state` as `[a, h, k, p, q, λ]`.
fn unpack(state: &DVector<f64>) -> [f64; 6] {
    [state[0], state[1], state[2], state[3], state[4], state[5]]
}

/// Elements `[a, h, k, p, q, λ]` → inertial Cartesian position (m), via the
/// crate's tested element→Cartesian conversion (stack-only `OrbitalState`, no
/// heap element representation). Exposed so the coast-gap demonstration and
/// the UKF measurement closure share one mapping.
///
/// # Panics
///
/// Never in practice — the `Equinoctial` variant always converts (only the
/// `Tle` variant can fail, which this never constructs).
pub fn equinoctial_position(elements: &[f64; 6], mu: f64) -> DVector<f64> {
    let state = OrbitalState {
        elements: OrbitalElements::Equinoctial {
            sma: elements[0],
            h: elements[1],
            k: elements[2],
            p: elements[3],
            q: elements[4],
            mean_longitude: elements[5],
        },
        mu,
        frame: Frame::Teme,
        epoch: Epoch::from_jde_utc(CONVERSION_EPOCH_JD),
    };
    let (r, _) = state
        .as_cartesian()
        .expect("equinoctial elements always convert to Cartesian");
    DVector::from_row_slice(&[r.x, r.y, r.z])
}

/// Inertial process-noise mapping matrix `Γᴵ = ∂(element rates)/∂(inertial
/// perturbing acceleration)` — the 6×3 Gauss variational input matrix for an
/// acceleration modeled in the inertial frame (Stacey & D'Amico Eq. 2).
/// Column `j` is the element-rate response to a unit inertial acceleration
/// along axis `j`, isolated by subtracting the zero-perturbation base rate
/// (the analytic two-body `dλ/dt = n` secular term the GVE add unconditionally).
fn inertial_input_matrix(elements: &[f64; 6], mu: f64) -> DMatrix<f64> {
    let zero = |_t: f64, _r: &Vector3<f64>, _v: &Vector3<f64>| Vector3::zeros();
    let base = equinoctial_element_rates(elements, mu, 0.0, &zero);
    let mut g = DMatrix::zeros(6, 3);
    for axis in 0..3 {
        let unit = move |_t: f64, _r: &Vector3<f64>, _v: &Vector3<f64>| {
            let mut a = Vector3::zeros();
            a[axis] = 1.0;
            a
        };
        let rate = equinoctial_element_rates(elements, mu, 0.0, &unit);
        for row in 0..6 {
            g[(row, axis)] = rate[row] - base[row];
        }
    }
    g
}

impl MotionModel for EquinoctialModel {
    fn state_dim(&self) -> usize {
        6
    }

    fn predict(&self, state: &DVector<f64>, dt: f64) -> DVector<f64> {
        let elements = unpack(state);
        let out = propagate_equinoctial(
            elements,
            self.gravity.mu,
            0.0,
            dt,
            self.max_step_s,
            &self.j2_perturbation(),
        );
        DVector::from_row_slice(&out)
    }

    fn jacobian(&self, state: &DVector<f64>, dt: f64) -> DMatrix<f64> {
        numeric_jacobian(
            |x, step| self.predict(x, step),
            state,
            dt,
            &ELEMENT_6D_SCALES,
        )
    }

    fn process_noise(&self, dt: f64) -> DMatrix<f64> {
        // Eq. (5) of Stacey & D'Amico with Φ ≈ I over the interval and the
        // isotropic inertial acceleration PSD Q̃ᴵ = σ_accel²·I₃ (their Eq. 3):
        //   Q ≈ Γᴵ·(σ²I₃)·Γᴵᵀ·Δt = σ²·(Γᴵ Γᴵᵀ)·Δt.
        // Symmetric positive-semidefinite (Gram form; rank ≤ 3 — added to the
        // full-rank predicted covariance, so the sum stays PD). Γᴵ is frozen
        // at the reference elements (module docs). This is the same physical
        // PSD KeplerJ2 assumes (its Q is Eq. 5's closed form in Cartesian).
        let g = inertial_input_matrix(&self.reference_elements, self.gravity.mu);
        let scale = self.sigma_accel * self.sigma_accel * dt;
        (&g * g.transpose()) * scale
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ukf::{UkfParams, UnscentedKalmanFilter};
    use std::f64::consts::TAU;
    use thresh_core::orbital::{j2_acceleration, rk4_step};

    const EARTH: GravityModel = GravityModel::EARTH_WGS84;
    const EARTH_MU: f64 = GravityModel::EARTH_WGS84.mu;

    /// Equinoctial elements `[a, h, k, p, q, λ]` for a Keplerian state, via the
    /// crate's tested conversion.
    fn elements_from_keplerian(kep: [f64; 6]) -> [f64; 6] {
        let [sma, ecc, inc, raan, argp, true_anomaly] = kep;
        let state = OrbitalState {
            elements: OrbitalElements::Keplerian {
                sma,
                ecc,
                inc,
                raan,
                argp,
                true_anomaly,
            },
            mu: EARTH_MU,
            frame: Frame::Teme,
            epoch: Epoch::from_jde_utc(CONVERSION_EPOCH_JD),
        };
        let (a, h, k, p, q, l) = state.as_equinoctial().unwrap();
        [a, h, k, p, q, l]
    }

    /// A generic inclined, mildly eccentric LEO used across the tests.
    fn leo_elements() -> [f64; 6] {
        elements_from_keplerian([7_000_000.0, 0.01, 0.9, 1.2, 0.4, 0.0])
    }

    /// Inertial `(r, v)` of an element state.
    fn elements_to_state(elements: &[f64; 6]) -> (Vector3<f64>, Vector3<f64>) {
        OrbitalState {
            elements: OrbitalElements::Equinoctial {
                sma: elements[0],
                h: elements[1],
                k: elements[2],
                p: elements[3],
                q: elements[4],
                mean_longitude: elements[5],
            },
            mu: EARTH_MU,
            frame: Frame::Teme,
            epoch: Epoch::from_jde_utc(CONVERSION_EPOCH_JD),
        }
        .as_cartesian()
        .unwrap()
    }

    /// Cartesian two-body+J2 reference propagator (the independent formulation
    /// the GVE path is validated against): sub-stepped RK4 over the shared
    /// `j2_acceleration` composite. Returns the endpoint position (m).
    fn propagate_cartesian(
        r0: Vector3<f64>,
        v0: Vector3<f64>,
        dt: f64,
        max_step_s: f64,
    ) -> Vector3<f64> {
        let steps = (dt / max_step_s).ceil().max(1.0) as usize;
        let sub = dt / steps as f64;
        let accel = |p: &Vector3<f64>, _v: &Vector3<f64>| j2_acceleration(p, &EARTH);
        let (mut r, mut v) = (r0, v0);
        for _ in 0..steps {
            let (rr, vv) = rk4_step(&r, &v, sub, accel);
            r = rr;
            v = vv;
        }
        r
    }

    fn assert_symmetric_positive_definite(p: &DMatrix<f64>, label: &str) {
        let asym = (p - p.transpose()).amax();
        assert!(asym < 1e-6 * p.amax(), "{label}: asymmetry {asym:e}");
        assert!(
            p.clone().cholesky().is_some(),
            "{label}: covariance is not positive definite"
        );
    }

    /// PD diagonal covariance: (1 km)² on `a`, (5e-5)² on the dimensionless
    /// elements, `sigma_lambda²` on `λ`.
    fn element_covariance(sigma_lambda: f64) -> DMatrix<f64> {
        DMatrix::from_diagonal(&DVector::from_row_slice(&[
            1.0e6,
            2.5e-9,
            2.5e-9,
            2.5e-9,
            2.5e-9,
            sigma_lambda * sigma_lambda,
        ]))
    }

    // ── task 3.3: measurement mapping (spec: "Measurement mapping matches the
    //    conversion") ───────────────────────────────────────────────────────

    #[test]
    fn measurement_mapping_matches_conversion_bitwise() {
        let elements = leo_elements();
        let model = EquinoctialModel::new(EARTH, elements);
        let x = DVector::from_row_slice(&elements);
        let mapped = model.position_measurement(&x);

        // The exact element→Cartesian conversion this must equal.
        let (r, _) = elements_to_state(&elements);
        for i in 0..3 {
            assert_eq!(
                mapped[i].to_bits(),
                r[i].to_bits(),
                "measurement component {i} is not bitwise the conversion"
            );
        }
        // The free function and the method agree.
        let free = equinoctial_position(&elements, EARTH_MU);
        assert_eq!(free, mapped);
    }

    // ── task 3.4: process noise is the input-matrix-mapped acceleration PSD ──

    #[test]
    fn process_noise_is_symmetric_psd_and_scales_with_psd_and_dt() {
        let elements = leo_elements();
        let mut model = EquinoctialModel::new(EARTH, elements);
        model.sigma_accel = 1e-4;
        let q1 = model.process_noise(10.0);

        // Symmetric; positive-semidefinite (Gram form Γ Γᵀ).
        assert!((&q1 - q1.transpose()).amax() < 1e-30 * q1.amax().max(1.0));
        let min_eig = q1
            .clone()
            .symmetric_eigen()
            .eigenvalues
            .iter()
            .copied()
            .fold(f64::INFINITY, f64::min);
        assert!(
            min_eig > -1e-9 * q1.amax(),
            "Q not PSD, min eig {min_eig:e}"
        );

        // Linear in dt (Φ ≈ I leading term) and quadratic in σ_accel.
        let q2 = model.process_noise(20.0);
        assert!((&q2 - &q1 * 2.0).amax() < 1e-12 * q1.amax());
        model.sigma_accel = 2e-4; // ×2 → Q ×4
        let q4 = model.process_noise(10.0);
        assert!((&q4 - &q1 * 4.0).amax() < 1e-9 * q1.amax());

        // Rank ≤ 3 (three acceleration inputs) but added to a full-rank
        // predicted covariance keeps that sum PD — proven in the sigma-point
        // test below.
    }

    // ── task 3.1: sigma-point prediction (spec: "Sigma-point prediction
    //    through the element dynamics") ───────────────────────────────────────

    #[test]
    fn sigma_point_prediction_matches_direct_and_is_pd() {
        let elements = leo_elements();
        let model = EquinoctialModel::new(EARTH, elements);
        let x0 = DVector::from_row_slice(&elements);
        let p0 = element_covariance(1e-4);
        let dt = 600.0;

        // Direct propagation of the prior mean.
        let reference = model.predict(&x0, dt);

        let mut ukf = UnscentedKalmanFilter::new(x0, p0, UkfParams::default());
        ukf.predict(&model, dt);

        // Sigma-point mean matches direct propagation up to the (tiny)
        // nonlinearity across the sigma spread. MEASURED (cargo test): amax
        // gap ≈ 2.0e-3 (dominated by the a-component in metres over a 600 s
        // J2 arc); the dimensionless element components agree to ~1e-12.
        // Tolerance 1e-2.
        let gap = (&ukf.x - &reference).amax();
        assert!(gap < 1e-2, "sigma-point mean vs direct: amax gap {gap:e}");

        // Predicted covariance is symmetric positive definite (the rank-≤3
        // process noise added to the full-rank sigma-point moment).
        assert_symmetric_positive_definite(&ukf.p, "equinoctial UKF predict");
    }

    // ── task 3.2: wrap-straddle (spec: "Wrap-straddling sigma points produce
    //    no artifact") ────────────────────────────────────────────────────────

    #[test]
    fn wrap_straddling_sigma_points_produce_no_artifact() {
        // Mean λ sits just below 2π; a wide σ_λ with α = 1 spreads the sigma
        // points across the 2π boundary.
        let mut elements = leo_elements();
        elements[5] = TAU - 0.01;
        let model = EquinoctialModel::new(EARTH, elements);
        let params = UkfParams {
            alpha: 1.0,
            beta: 2.0,
            kappa: 0.0,
        };
        let p0 = element_covariance(0.03); // √6·0.03 ≈ 0.073 rad spread > 0.01
        let dt = 400.0;

        // Run A straddles 2π; run B is the same physical state shifted down by
        // exactly one revolution (λ near 0⁻). The dynamics are 2π-periodic in
        // λ, so the two runs must agree on the five slow elements and the
        // whole covariance, differing only by the constant 2π offset in the λ
        // mean — the hallmark of a wrap-artifact-free unwrapped convention.
        let mut ukf_a =
            UnscentedKalmanFilter::new(DVector::from_row_slice(&elements), p0.clone(), params);
        ukf_a.predict(&model, dt);

        let mut elements_b = elements;
        elements_b[5] -= TAU;
        let mut ukf_b =
            UnscentedKalmanFilter::new(DVector::from_row_slice(&elements_b), p0, params);
        ukf_b.predict(&model, dt);

        // Five slow elements agree (relative, since `a` is ~1e7).
        for i in 0..5 {
            let rel = (ukf_a.x[i] - ukf_b.x[i]).abs() / (1.0 + ukf_a.x[i].abs());
            assert!(
                rel < 1e-9,
                "element {i} differs across the wrap: rel {rel:e}"
            );
        }
        // λ differs by exactly one revolution (no wrap artifact in the mean).
        let lambda_gap = (ukf_a.x[5] - ukf_b.x[5]) - TAU;
        assert!(
            lambda_gap.abs() < 1e-6,
            "λ mean shift {} vs 2π",
            ukf_a.x[5] - ukf_b.x[5]
        );
        // The covariance is identical (no wrap artifact in the second moment).
        let cov_gap = (&ukf_a.p - &ukf_b.p).amax();
        assert!(
            cov_gap < 1e-9 * ukf_a.p.amax(),
            "covariance differs across the wrap by {cov_gap:e}"
        );
    }

    // ── task 3.2: long-horizon accumulation (spec: "Long-horizon accumulation
    //    stays accurate") ─────────────────────────────────────────────────────

    #[test]
    fn long_horizon_lambda_accumulation_stays_accurate() {
        let elements = leo_elements();
        let (r0, v0) = elements_to_state(&elements);
        let mut model = EquinoctialModel::new(EARTH, elements);
        // 5 s steps in both formulations so the residual reflects the
        // formulation agreement, not either integrator's coarse-step
        // truncation.
        model.max_step_s = 5.0;

        // ~50 revolutions: λ accumulates ~50·2π ≈ 314 rad, well into the
        // "hundreds of radians" regime.
        let period = TAU * (7_000_000_f64.powi(3) / EARTH_MU).sqrt();
        let arc = 50.0 * period;

        let el_end = model.predict(&DVector::from_row_slice(&elements), arc);
        // λ is unwrapped: it is NOT reduced mod 2π in state space.
        assert!(
            el_end[5] > 300.0,
            "λ must accumulate unwrapped, got {}",
            el_end[5]
        );

        // The converted Cartesian position still agrees with the independent
        // Cartesian two-body+J2 path at the cross-formulation tolerance.
        let r_gve = model.position_measurement(&el_end);
        let r_gve = Vector3::new(r_gve[0], r_gve[1], r_gve[2]);
        let r_cart = propagate_cartesian(r0, v0, arc, model.max_step_s);

        // MEASURED (cargo test): gap ≈ 0.109 m over 50 revs at 5 s steps
        // (≈ 2.81 m at 10 s steps — the ~26× shrink from halving the step
        // confirms the residual is ordinary RK4 truncation, dominated by the
        // Cartesian reference's non-analytic two-body integration, NOT a λ
        // artifact: the sin/cos argument-reduction loss at λ ~ 3e2 rad is
        // ~1e-14 rad, and nothing blows up as λ crosses hundreds of radians).
        // Tolerance 0.5 m (a wrap/precision artifact would drift kilometres).
        let gap = (r_gve - r_cart).norm();
        assert!(gap < 0.5, "long-horizon GVE vs Cartesian gap = {gap:.4e} m");
    }
}
