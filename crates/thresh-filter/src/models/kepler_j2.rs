//! Kepler + J2 orbital motion model.
//!
//! State: `[x, vx, y, vy, z, vz]` (6D, interleaved) — ECI position and
//! velocity in metres / m/s.
//! Nonlinear — requires EKF, UKF, or CKF.

use nalgebra::{DMatrix, DVector, Vector3};
use thresh_core::orbital::{GravityModel, j2_acceleration, rk4_step};

use crate::numeric::numeric_jacobian;
use crate::traits::MotionModel;

/// Numeric-Jacobian floor scales for the interleaved 6D layout: 1 m on
/// position columns, 1e-3 m/s on velocity columns (design Decision 3 of the
/// `orbital-ballistic-filter-models` change).
pub(crate) const INTERLEAVED_6D_SCALES: [f64; 6] = [1.0, 1e-3, 1.0, 1e-3, 1.0, 1e-3];

/// Unpack the interleaved `[x, vx, y, vy, z, vz]` prefix of `state` into
/// `(position, velocity)` vectors.
pub(crate) fn unpack_interleaved(state: &DVector<f64>) -> (Vector3<f64>, Vector3<f64>) {
    (
        Vector3::new(state[0], state[2], state[4]),
        Vector3::new(state[1], state[3], state[5]),
    )
}

/// Write `(position, velocity)` into the interleaved `[x, vx, y, vy, z, vz]`
/// prefix of `out`.
pub(crate) fn pack_interleaved(pos: &Vector3<f64>, vel: &Vector3<f64>, out: &mut DVector<f64>) {
    for i in 0..3 {
        out[2 * i] = pos[i];
        out[2 * i + 1] = vel[i];
    }
}

/// Number of equal RK4 sub-steps covering `dt`: `n = ceil(dt / max_step_s)`,
/// at least 1 (so `dt = 0` degenerates to a single exact identity step).
pub(crate) fn substep_count(dt: f64, max_step_s: f64) -> usize {
    (dt / max_step_s).ceil().max(1.0) as usize
}

/// Total two-body + J2 gravitational acceleration with the degenerate-radius
/// guard of design Decision 3: if `‖r‖ < equatorial_radius / 2` the state is
/// physically meaningless (filter divergence), so the radius used in the
/// force evaluation is clamped to that threshold — the force stays finite,
/// no NaN poisons the filter, and the covariance blow-up surfaces the
/// problem. Never panics.
///
/// Above the threshold this is exactly [`j2_acceleration`] (same code path),
/// preserving Decision 1's bitwise identity with the synth propagator.
pub(crate) fn clamped_gravity_acceleration(
    pos: &Vector3<f64>,
    gravity: &GravityModel,
) -> Vector3<f64> {
    let min_radius = gravity.equatorial_radius / 2.0;
    let r = pos.norm();
    if r >= min_radius {
        return j2_acceleration(pos, gravity);
    }
    // Degenerate: rescale the position to the clamp radius (arbitrary +x
    // direction if the radius is exactly zero, where no direction exists).
    let clamped = if r > 0.0 {
        pos * (min_radius / r)
    } else {
        Vector3::new(min_radius, 0.0, 0.0)
    };
    j2_acceleration(&clamped, gravity)
}

/// Fill the interleaved per-axis continuous white-noise-acceleration blocks
/// `σ²·[[dt³/3, dt²/2], [dt²/2, dt]]` into the leading 6×6 of `q`.
///
/// This is the CWNA discretization of Decisions 3/4 (see the process-noise
/// section of `docs/reference/orbital-ballistic-dynamics-reference.md`) —
/// deliberately the `dt³/3` block, not `ConstantVelocity`'s DWNA `dt⁴/4`.
pub(crate) fn fill_wna_blocks(q: &mut DMatrix<f64>, sigma_accel: f64, dt: f64) {
    let psd = sigma_accel * sigma_accel;
    let dt2 = dt * dt;
    let dt3 = dt2 * dt;
    for i in 0..3 {
        let p = 2 * i;
        q[(p, p)] = psd * dt3 / 3.0;
        q[(p, p + 1)] = psd * dt2 / 2.0;
        q[(p + 1, p)] = psd * dt2 / 2.0;
        q[(p + 1, p + 1)] = psd * dt;
    }
}

/// Kepler + J2 orbital motion model in 3D.
///
/// State vector: `[x, vx, y, vy, z, vz]` — interleaved like
/// [`crate::models::cv::ConstantVelocity`] and the IMM common space, **not**
/// the blocked `[r | v]` layout of astrodynamics texts (the permutation is
/// spelled out in `docs/reference/orbital-ballistic-dynamics-reference.md`).
///
/// Frame contract (documented, not enforced — the [`MotionModel`] trait is
/// frame-blind): the state is ECI under the repo's GMST-only convention
/// (`thresh_core::eci`), in metres and m/s.
///
/// `predict` sub-steps the interval through the shared closure-based RK4
/// integrator with `a(r)` = two-body + J2 from `thresh_core::orbital` —
/// byte-identical physics to the `thresh-synth` truth propagator (design
/// Decisions 1 and 3 of the `orbital-ballistic-filter-models` change). The
/// Jacobian is numeric: central differences of the exact discretized map.
pub struct KeplerJ2 {
    /// Central-body gravitational parameters (μ, J2, equatorial radius) —
    /// parameterized per Decision 1, never a hardcoded Earth constant.
    pub gravity: GravityModel,
    /// RK4 sub-step ceiling (s); `predict` takes `ceil(dt / max_step_s)`
    /// equal sub-steps. Default 10.0.
    pub max_step_s: f64,
    /// Unmodeled-acceleration white-noise PSD (m/s²·Hz^-½) absorbing drag,
    /// SRP, higher harmonics, and small maneuvers. Default 1e-3.
    pub sigma_accel: f64,
}

impl KeplerJ2 {
    /// Create a model for the given central body with the Decision 3
    /// defaults (`max_step_s = 10.0` s, `sigma_accel = 1e-3` m/s²).
    pub fn new(gravity: GravityModel) -> Self {
        Self {
            gravity,
            max_step_s: 10.0,
            sigma_accel: 1e-3,
        }
    }
}

impl MotionModel for KeplerJ2 {
    fn state_dim(&self) -> usize {
        6
    }

    fn predict(&self, state: &DVector<f64>, dt: f64) -> DVector<f64> {
        let (mut pos, mut vel) = unpack_interleaved(state);
        let n = substep_count(dt, self.max_step_s);
        let sub_dt = dt / n as f64;
        let accel =
            |p: &Vector3<f64>, _v: &Vector3<f64>| clamped_gravity_acceleration(p, &self.gravity);
        for _ in 0..n {
            let (p, v) = rk4_step(&pos, &vel, sub_dt, accel);
            pos = p;
            vel = v;
        }
        let mut next = DVector::zeros(6);
        pack_interleaved(&pos, &vel, &mut next);
        next
    }

    fn jacobian(&self, state: &DVector<f64>, dt: f64) -> DMatrix<f64> {
        numeric_jacobian(
            |x, step| self.predict(x, step),
            state,
            dt,
            &INTERLEAVED_6D_SCALES,
        )
    }

    fn process_noise(&self, dt: f64) -> DMatrix<f64> {
        let mut q = DMatrix::zeros(6, 6);
        fill_wna_blocks(&mut q, self.sigma_accel, dt);
        q
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ckf::CubatureKalmanFilter;
    use crate::ekf::ExtendedKalmanFilter;
    use crate::ukf::{UkfParams, UnscentedKalmanFilter};
    use nalgebra::Matrix3;
    use thresh_core::orbital::cartesian_to_keplerian;

    const EARTH: GravityModel = GravityModel::EARTH_WGS84;

    /// ISS-like LEO state: 400 km circular altitude, 51.6° inclination,
    /// interleaved `[x, vx, y, vy, z, vz]`.
    fn leo_state() -> DVector<f64> {
        let r = EARTH.equatorial_radius + 400_000.0;
        let v = (EARTH.mu / r).sqrt();
        let inc = 51.6_f64.to_radians();
        DVector::from_column_slice(&[r, 0.0, 0.0, v * inc.cos(), 0.0, v * inc.sin()])
    }

    fn assert_symmetric_positive_definite(p: &DMatrix<f64>, label: &str) {
        let asym = (p - p.transpose()).amax();
        assert!(asym < 1e-6 * p.amax(), "{label}: asymmetry {asym:e}");
        assert!(
            p.clone().cholesky().is_some(),
            "{label}: covariance is not positive definite"
        );
    }

    // ── Phase helpers ────────────────────────────────────────────────────

    #[test]
    fn unpack_pack_interleaved_round_trip() {
        let state = DVector::from_column_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let (pos, vel) = unpack_interleaved(&state);
        assert_eq!(pos, Vector3::new(1.0, 3.0, 5.0));
        assert_eq!(vel, Vector3::new(2.0, 4.0, 6.0));

        let mut out = DVector::zeros(6);
        pack_interleaved(&pos, &vel, &mut out);
        assert_eq!(out, state);
    }

    #[test]
    fn substep_count_ceils_and_floors_at_one() {
        assert_eq!(substep_count(60.0, 10.0), 6);
        assert_eq!(substep_count(60.1, 10.0), 7);
        assert_eq!(substep_count(5.0, 10.0), 1);
        assert_eq!(substep_count(0.0, 10.0), 1);
    }

    #[test]
    fn clamped_gravity_is_identical_above_threshold() {
        let pos = Vector3::new(6_778_137.0, 1000.0, -2000.0);
        let clamped = clamped_gravity_acceleration(&pos, &EARTH);
        let direct = j2_acceleration(&pos, &EARTH);
        // Same code path above the threshold: bitwise identical.
        for i in 0..3 {
            assert_eq!(clamped[i].to_bits(), direct[i].to_bits());
        }
    }

    #[test]
    fn clamped_gravity_stays_finite_below_threshold() {
        let min_radius = EARTH.equatorial_radius / 2.0;
        // Deep inside the clamp region, including the exact origin.
        for pos in [Vector3::new(100.0, -50.0, 25.0), Vector3::zeros()] {
            let acc = clamped_gravity_acceleration(&pos, &EARTH);
            assert!(acc.iter().all(|a| a.is_finite()), "NaN/inf at {pos:?}");
            // Magnitude is that of the force at the clamp radius, not larger.
            let bound = 1.01 * EARTH.mu / (min_radius * min_radius);
            assert!(acc.norm() <= bound, "unbounded force at {pos:?}");
        }
    }

    #[test]
    fn wna_blocks_match_closed_form() {
        let model = KeplerJ2::new(EARTH);
        let sigma = model.sigma_accel;
        let dt = 2.0;
        let q = model.process_noise(dt);
        let psd = sigma * sigma;
        for i in 0..3 {
            let p = 2 * i;
            assert!((q[(p, p)] - psd * dt.powi(3) / 3.0).abs() < 1e-18);
            assert!((q[(p, p + 1)] - psd * dt * dt / 2.0).abs() < 1e-18);
            assert!((q[(p + 1, p)] - q[(p, p + 1)]).abs() < 1e-18);
            assert!((q[(p + 1, p + 1)] - psd * dt).abs() < 1e-18);
        }
        // No cross-axis coupling.
        assert_eq!(q[(0, 2)], 0.0);
        assert_eq!(q[(1, 5)], 0.0);
        // The discretized CWNA block itself is positive definite.
        assert_symmetric_positive_definite(&q, "process noise");
    }

    // ── predict ──────────────────────────────────────────────────────────

    #[test]
    fn predict_single_substep_matches_shared_rk4() {
        let model = KeplerJ2::new(EARTH);
        let state = leo_state();
        let (pos, vel) = unpack_interleaved(&state);
        let (exp_pos, exp_vel) = rk4_step(&pos, &vel, 10.0, |p, _| j2_acceleration(p, &EARTH));

        let next = model.predict(&state, 10.0);
        let (got_pos, got_vel) = unpack_interleaved(&next);
        // Same closure, same integrator, one sub-step: bitwise identical.
        for i in 0..3 {
            assert_eq!(got_pos[i].to_bits(), exp_pos[i].to_bits());
            assert_eq!(got_vel[i].to_bits(), exp_vel[i].to_bits());
        }
    }

    #[test]
    fn predict_never_panics_on_degenerate_state() {
        let model = KeplerJ2::new(EARTH);
        // Physically meaningless state deep below the surface.
        let state = DVector::from_column_slice(&[1000.0, -50.0, 200.0, 10.0, -400.0, 3.0]);
        let next = model.predict(&state, 30.0);
        assert!(next.iter().all(|v| v.is_finite()));
    }

    // ── Task 3.3: prediction through all three nonlinear filters ─────────

    // Spec "Prediction through all three nonlinear filters": EKF, UKF, and
    // CKF each predict one step from the same LEO state — mean consistent
    // with Kepler+J2 propagation, covariance symmetric positive definite.
    #[test]
    fn ekf_ukf_ckf_one_step_from_leo_state() {
        let model = KeplerJ2::new(EARTH);
        let x0 = leo_state();
        let mut p0 = DMatrix::zeros(6, 6);
        for i in 0..3 {
            p0[(2 * i, 2 * i)] = 1_000.0 * 1_000.0; // (1 km)² position
            p0[(2 * i + 1, 2 * i + 1)] = 10.0 * 10.0; // (10 m/s)² velocity
        }
        let dt = 10.0;
        let reference = model.predict(&x0, dt);

        let mut ekf = ExtendedKalmanFilter::new(x0.clone(), p0.clone());
        ekf.predict(&model, dt);
        // EKF propagates the mean through the exact predict.
        assert!((&ekf.x - &reference).amax() < 1e-9);
        assert_symmetric_positive_definite(&ekf.p, "EKF");

        let mut ukf = UnscentedKalmanFilter::new(x0.clone(), p0.clone(), UkfParams::default());
        ukf.predict(&model, dt);
        // Sigma-point means differ from the propagated mean only by the
        // (tiny) nonlinearity of gravity across the sigma spread.
        assert!((&ukf.x - &reference).amax() < 1.0);
        assert_symmetric_positive_definite(&ukf.p, "UKF");

        let mut ckf = CubatureKalmanFilter::new(x0.clone(), p0.clone());
        ckf.predict(&model, dt);
        assert!((&ckf.x - &reference).amax() < 1.0);
        assert_symmetric_positive_definite(&ckf.p, "CKF");
    }

    // Spec "Non-Earth gravitational parameter": construction with lunar mu
    // uses the supplied mu throughout — a circular orbit sized from the
    // lunar mu stays circular, which Earth-mu leakage would destroy.
    #[test]
    fn lunar_mu_is_used_throughout() {
        let moon = GravityModel {
            mu: 4.904_869_5e12, // lunar GM (m³/s²)
            j2: 0.0,            // isolate the two-body term
            equatorial_radius: 1_738_100.0,
        };
        let r = moon.equatorial_radius + 100_000.0;
        let v = (moon.mu / r).sqrt();
        let x0 = DVector::from_column_slice(&[r, 0.0, 0.0, v, 0.0, 0.0]);

        let lunar = KeplerJ2::new(moon);
        let mut state = x0.clone();
        for _ in 0..30 {
            state = lunar.predict(&state, 60.0);
            let (pos, _) = unpack_interleaved(&state);
            assert!(
                (pos.norm() - r).abs() < 10.0,
                "circular lunar orbit drifted: {} m",
                (pos.norm() - r).abs()
            );
        }

        // Control: the same state under Earth's mu diverges immediately,
        // proving the assertion above actually discriminates on mu.
        let earth_model = KeplerJ2::new(EARTH);
        let earth_state = earth_model.predict(&x0, 60.0);
        let (earth_pos, _) = unpack_interleaved(&earth_state);
        assert!((earth_pos.norm() - r).abs() > 10_000.0);
    }

    // ── Task 3.4: Jacobian consistency vs the analytic linearization ─────

    /// Analytic two-body gravity gradient `G₂ᵦ = (μ/r⁵)·(3·r rᵀ − r²·I)`
    /// from the "Analytic Jacobian of the two-body + J2 field" section of
    /// `docs/reference/orbital-ballistic-dynamics-reference.md`.
    fn two_body_gradient(pos: &Vector3<f64>, gravity: &GravityModel) -> Matrix3<f64> {
        let r2 = pos.norm_squared();
        let r5 = r2 * r2 * r2.sqrt();
        (gravity.mu / r5) * (3.0 * pos * pos.transpose() - r2 * Matrix3::identity())
    }

    /// Analytic J2 gravity gradient `G_J2 = (3/2)·J2·μ·Re²/r⁵ · M` from the
    /// same reference-doc section (M entries as printed there).
    fn j2_gradient(pos: &Vector3<f64>, gravity: &GravityModel) -> Matrix3<f64> {
        let (x, y, z) = (pos.x, pos.y, pos.z);
        let r2 = pos.norm_squared();
        let u = z * z / r2;
        let c = 1.5 * gravity.j2 * gravity.mu * gravity.equatorial_radius.powi(2)
            / (r2 * r2 * r2.sqrt());

        let mut m = Matrix3::zeros();
        m[(0, 0)] = (5.0 * u - 1.0) + (5.0 * x * x / r2) * (1.0 - 7.0 * u);
        m[(1, 1)] = (5.0 * u - 1.0) + (5.0 * y * y / r2) * (1.0 - 7.0 * u);
        m[(2, 2)] = (5.0 * u - 3.0) + (5.0 * z * z / r2) * (5.0 - 7.0 * u);
        m[(0, 1)] = (5.0 * x * y / r2) * (1.0 - 7.0 * u);
        m[(1, 0)] = m[(0, 1)];
        m[(0, 2)] = (5.0 * x * z / r2) * (3.0 - 7.0 * u);
        m[(2, 0)] = m[(0, 2)];
        m[(1, 2)] = (5.0 * y * z / r2) * (3.0 - 7.0 * u);
        m[(2, 1)] = m[(1, 2)];
        c * m
    }

    /// Embed the 3×3 gravity gradient into the interleaved 6D continuous-time
    /// Jacobian: `A[2i, 2i+1] = 1`, `A[2i+1, 2j] = G_ij` (reference doc,
    /// "Embedding in the interleaved 6D state").
    fn interleaved_a_matrix(g: &Matrix3<f64>) -> DMatrix<f64> {
        let mut a = DMatrix::zeros(6, 6);
        for i in 0..3 {
            a[(2 * i, 2 * i + 1)] = 1.0;
            for j in 0..3 {
                a[(2 * i + 1, 2 * j)] = g[(i, j)];
            }
        }
        a
    }

    // Spec "Jacobian consistency": the numeric Jacobian agrees with
    // `Φ(dt) = I + A(x)·dt + O(dt²)` where A embeds the analytic two-body+J2
    // ∂a/∂r derived in docs/reference/orbital-ballistic-dynamics-reference.md,
    // section "Analytic Jacobian of the two-body + J2 field". Halving dt must
    // shrink the residual ~4× (the error really is second-order).
    #[test]
    fn numeric_jacobian_consistent_with_analytic_linearization() {
        let model = KeplerJ2::new(EARTH);
        // Generic LEO point with every position AND velocity component
        // nonzero: all G entries are exercised, and every numeric-Jacobian
        // column step scales with its component magnitude (a zero component
        // would fall to the small velocity floor, whose rounding noise
        // against ~7e6 m outputs swamps the O(dt²) residual under test).
        let pos = Vector3::new(4_000_000.0, 3_000_000.0, 4_500_000.0);
        let v_dir: Vector3<f64> = pos.cross(&Vector3::new(0.2, -0.3, 0.9)).normalize();
        let vel = (EARTH.mu / pos.norm()).sqrt() * v_dir;
        let mut x = DVector::zeros(6);
        pack_interleaved(&pos, &vel, &mut x);

        let g = two_body_gradient(&pos, &EARTH) + j2_gradient(&pos, &EARTH);
        // Free assertions from the reference doc: G is symmetric (Hessian of
        // the potential) and traceless (harmonic potential).
        assert!((g - g.transpose()).amax() < 1e-15 * g.amax());
        assert!(g.trace().abs() < 1e-9 * g.amax());

        let a = interleaved_a_matrix(&g);
        let identity = DMatrix::<f64>::identity(6, 6);

        let residual = |dt: f64| -> f64 {
            let f_num = model.jacobian(&x, dt);
            (&f_num - (&identity + &a * dt)).amax()
        };

        // Budget: the dropped O(dt²) term is ≈ ‖G‖·dt²/2 ≈ 1.3e-6 at dt = 1
        // (reference doc, "How the numeric Jacobian is cross-checked");
        // halving dt must shrink it ~4× (0.4 allows finite-difference noise).
        let res_full = residual(1.0);
        let res_half = residual(0.5);
        assert!(
            res_full < 5e-6,
            "first-order residual too large: {res_full:e}"
        );
        assert!(
            res_half < 0.4 * res_full,
            "residual is not O(dt²): res(1.0) = {res_full:e}, res(0.5) = {res_half:e}"
        );
    }

    // ── Task 3.5: golden tier 2 ──────────────────────────────────────────

    // Golden tier 2 (Decision 8): Vallado, "Fundamentals of Astrodynamics
    // and Applications", 4th ed., Example 2-4 (the Kepler problem), with J2
    // disabled. Documented tolerance: 10 m position / 0.01 m/s velocity —
    // the example's printed precision.
    #[test]
    fn vallado_kepler_propagation_example_2_4() {
        let two_body = GravityModel {
            j2: 0.0, // disable J2: pure Kepler propagation
            ..EARTH
        };
        let model = KeplerJ2::new(two_body);

        // Vallado Ex. 2-4 initial state (km/km·s⁻¹ in the book, SI here).
        let x0 = DVector::from_column_slice(&[
            1_131_340.0,
            -5_643.05,
            -2_282_343.0,
            4_303.33,
            6_672_423.0,
            2_428.79,
        ]);
        // Expected state after Δt = 40 min (book values, printed precision).
        let expected_pos = Vector3::new(-4_219_752.7, 4_363_029.2, -3_958_766.6);
        let expected_vel = Vector3::new(3_689.866, -1_916.735, -6_112.511);

        let next = model.predict(&x0, 40.0 * 60.0);
        let (pos, vel) = unpack_interleaved(&next);

        assert!(
            (pos - expected_pos).norm() <= 10.0,
            "position error {} m exceeds the 10 m tolerance",
            (pos - expected_pos).norm()
        );
        assert!(
            (vel - expected_vel).norm() <= 0.01,
            "velocity error {} m/s exceeds the 0.01 m/s tolerance",
            (vel - expected_vel).norm()
        );
    }

    // Spec "J2 secular drift fixture": multi-revolution LEO propagation
    // matches the analytic nodal-regression rate
    // dΩ/dt = −(3/2)·n·J2·(Re/p)²·cos i. Documented tolerance: 5% relative —
    // the drift is measured on the osculating RAAN, which carries
    // short-period J2 oscillations of order J2 (~1e-3 rad) on top of the
    // ~0.11 rad secular drift accumulated over 20 revolutions.
    #[test]
    fn j2_nodal_regression_matches_analytic_rate() {
        let sma = EARTH.equatorial_radius + 400_000.0;
        let ecc = 0.001;
        let inc = 51.6_f64.to_radians();
        let raan0 = 1.0;
        let (pos, vel) =
            thresh_core::orbital::keplerian_to_cartesian(sma, ecc, inc, raan0, 0.0, 0.0, EARTH.mu);
        let mut state = DVector::zeros(6);
        pack_interleaved(&pos, &vel, &mut state);

        let model = KeplerJ2::new(EARTH);
        let n_mean = (EARTH.mu / sma.powi(3)).sqrt();
        let period = std::f64::consts::TAU / n_mean;
        // 20 Keplerian revolutions in 60 s predict chunks.
        let chunks = (20.0 * period / 60.0).round();
        let total_time = chunks * 60.0;
        for _ in 0..chunks as usize {
            state = model.predict(&state, 60.0);
        }

        let (final_pos, final_vel) = unpack_interleaved(&state);
        let (_, _, _, raan_f, _, _) = cartesian_to_keplerian(&final_pos, &final_vel, EARTH.mu);

        let p_slr = sma * (1.0 - ecc * ecc);
        let analytic_rate =
            -1.5 * n_mean * EARTH.j2 * (EARTH.equatorial_radius / p_slr).powi(2) * inc.cos();
        let measured_drift = raan_f - raan0; // no 2π wrap: |drift| ≈ 0.11 rad
        let analytic_drift = analytic_rate * total_time;

        let rel_err = (measured_drift - analytic_drift).abs() / analytic_drift.abs();
        assert!(
            rel_err < 0.05,
            "nodal regression off by {:.2}%: measured {measured_drift:e}, analytic {analytic_drift:e}",
            rel_err * 100.0
        );
    }

    // Spec "Deterministic CI execution": the golden-vector tests run as
    // plain `cargo test` — this module needs no Python, no network, and no
    // non-default features by construction (plain #[cfg(test)] unit tests
    // over pure Rust math) — and repeated runs produce bit-identical
    // results. Asserted here for the tier-1/2 golden paths; task 8.2
    // extends the check to the tier-3 SGP4 fixtures.
    #[test]
    fn golden_propagation_is_bit_identical_across_runs() {
        let model = KeplerJ2::new(EARTH);
        let x0 = leo_state();

        let run = || {
            let mut state = x0.clone();
            for _ in 0..10 {
                state = model.predict(&state, 60.0);
            }
            let f = model.jacobian(&x0, 10.0);
            (state, f)
        };
        let (state_a, jac_a) = run();
        let (state_b, jac_b) = run();

        for i in 0..6 {
            assert_eq!(state_a[i].to_bits(), state_b[i].to_bits());
        }
        for (a, b) in jac_a.iter().zip(jac_b.iter()) {
            assert_eq!(a.to_bits(), b.to_bits());
        }
    }

    // Truth-identity test (Decision 1 enforcement, spec "Identical
    // accelerations from both consumers" at the propagation level):
    // KeplerJ2::predict over a 60 s LEO arc vs thresh_synth::propagate with
    // matched 10 s steps — same shared force math, same integrator, so the
    // agreement tolerance is 1e-6 m.
    #[test]
    fn predict_matches_synth_propagate_within_1e6_m() {
        let x0 = leo_state();
        let (pos0, vel0) = unpack_interleaved(&x0);

        let model = KeplerJ2::new(EARTH); // max_step_s = 10.0
        let predicted = model.predict(&x0, 60.0);
        let (pred_pos, pred_vel) = unpack_interleaved(&predicted);

        let initial = thresh_synth::orbital::OrbitalState::from_cartesian(
            [pos0.x, pos0.y, pos0.z],
            [vel0.x, vel0.y, vel0.z],
            2_451_545.0,
        );
        let config = thresh_synth::orbital::PropagatorConfig {
            include_j2: true,
            drag: None,
            dt_s: 10.0,
        };
        let truth = thresh_synth::orbital::propagate(&initial, 60.0, &config, 60.0);
        let last = truth.last().expect("propagate returns samples");

        for i in 0..3 {
            assert!(
                (pred_pos[i] - last.position[i]).abs() < 1e-6,
                "position component {i} differs by {} m",
                (pred_pos[i] - last.position[i]).abs()
            );
            assert!((pred_vel[i] - last.velocity[i]).abs() < 1e-9);
        }
    }
}
