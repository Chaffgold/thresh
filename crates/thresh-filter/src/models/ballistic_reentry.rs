//! Ballistic reentry motion model with estimated ballistic coefficient.
//!
//! State: `[x, vx, y, vy, z, vz, β]` (7D) — the interleaved 6D ECI layout of
//! [`crate::models::kepler_j2::KeplerJ2`] with the ballistic coefficient
//! β = m/(C_d·A) (kg/m²) appended.
//! Nonlinear — requires EKF, UKF, or CKF.

use nalgebra::{DMatrix, DVector, Vector3};
use thresh_core::orbital::{GravityModel, drag_acceleration, rk4_step};

use crate::models::kepler_j2::{
    clamped_gravity_acceleration, fill_wna_blocks, pack_interleaved, substep_count,
    unpack_interleaved,
};
use crate::numeric::numeric_jacobian;
use crate::traits::MotionModel;

/// Numeric-Jacobian floor scales for the 7D reentry layout: the interleaved
/// 6D floors (1 m position, 1e-3 m/s velocity) plus 100.0 on the β column —
/// `numeric_jacobian` steps `h_i = ε·max(|x_i|, s_i)`, so this yields
/// exactly the `max(|β|, 100.0)·ε` β step of design Decision 4.
const REENTRY_7D_SCALES: [f64; 7] = [1.0, 1e-3, 1.0, 1e-3, 1.0, 1e-3, 100.0];

/// Ballistic reentry model over a round rotating Earth.
///
/// State vector: `[x, vx, y, vy, z, vz, β]` — the first six components are
/// the interleaved ECI convention (GMST, metres / m/s, same frame contract
/// as [`crate::models::kepler_j2::KeplerJ2`]); β is the ballistic
/// coefficient in kg/m² with `β̇ = 0` dynamics (a random walk through the
/// process noise). Dynamics: `a = two_body+J2 + drag(r, v, 1/β)` from the
/// shared `thresh_core::orbital` force math, with the drag using the
/// co-rotating atmosphere relative velocity `v_rel = v − ω_⊕ × r` — the
/// rotating round Earth enters through the force model, not the frame
/// (design Decision 4 of the `orbital-ballistic-filter-models` change).
///
/// β is clamped to `beta_floor` inside `predict`, so measurement updates
/// that would drive β non-positive never produce undefined dynamics: the
/// effective (and re-emitted) β stays strictly positive.
pub struct BallisticReentry {
    /// Central-body gravitational parameters (μ, J2, equatorial radius).
    pub gravity: GravityModel,
    /// RK4 sub-step ceiling (s). Default 1.0 — reentry dynamics are fast
    /// and the atmosphere density varies steeply across a sub-step.
    pub max_step_s: f64,
    /// Unmodeled-acceleration white-noise PSD (m/s²·Hz^-½) absorbing lift
    /// and attitude effects. Default 5.0.
    pub sigma_accel: f64,
    /// β random-walk intensity (kg/m² per √s); keeps β uncertainty honest
    /// through the exo-atmospheric portion where β is unobservable.
    pub sigma_beta: f64,
    /// Lower clamp on the effective β (kg/m²). Default 10.0.
    pub beta_floor: f64,
}

impl BallisticReentry {
    /// Create a model with the Decision 4 defaults (`max_step_s = 1.0` s,
    /// `sigma_accel = 5.0` m/s², `beta_floor = 10.0` kg/m²).
    pub fn new(gravity: GravityModel, sigma_beta: f64) -> Self {
        Self {
            gravity,
            max_step_s: 1.0,
            sigma_accel: 5.0,
            sigma_beta,
            beta_floor: 10.0,
        }
    }

    /// Total acceleration `two_body+J2 + drag(r, v, 1/β)`. The gravity term
    /// carries the same degenerate-radius guard as `KeplerJ2`; the drag term
    /// vanishes outside the shared atmosphere table's 0–1000 km band, so one
    /// model covers midcourse and reentry without mode logic (Decision 6).
    fn acceleration(&self, pos: &Vector3<f64>, vel: &Vector3<f64>, beta: f64) -> Vector3<f64> {
        clamped_gravity_acceleration(pos, &self.gravity) + drag_acceleration(pos, vel, 1.0 / beta)
    }
}

impl MotionModel for BallisticReentry {
    fn state_dim(&self) -> usize {
        7
    }

    fn predict(&self, state: &DVector<f64>, dt: f64) -> DVector<f64> {
        let (mut pos, mut vel) = unpack_interleaved(state);
        // β̇ = 0; the floor keeps 1/β well-defined against adversarial
        // updates and is re-emitted so the filter state stays physical.
        let beta = state[6].max(self.beta_floor);
        let n = substep_count(dt, self.max_step_s);
        let sub_dt = dt / n as f64;
        let accel = |p: &Vector3<f64>, v: &Vector3<f64>| self.acceleration(p, v, beta);
        for _ in 0..n {
            let (p, v) = rk4_step(&pos, &vel, sub_dt, accel);
            pos = p;
            vel = v;
        }
        let mut next = DVector::zeros(7);
        pack_interleaved(&pos, &vel, &mut next);
        next[6] = beta;
        next
    }

    fn jacobian(&self, state: &DVector<f64>, dt: f64) -> DMatrix<f64> {
        numeric_jacobian(
            |x, step| self.predict(x, step),
            state,
            dt,
            &REENTRY_7D_SCALES,
        )
    }

    fn process_noise(&self, dt: f64) -> DMatrix<f64> {
        let mut q = DMatrix::zeros(7, 7);
        fill_wna_blocks(&mut q, self.sigma_accel, dt);
        // Independent β random walk (Decision 4).
        q[(6, 6)] = self.sigma_beta * self.sigma_beta * dt;
        q
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ckf::CubatureKalmanFilter;
    use crate::ekf::ExtendedKalmanFilter;
    use crate::models::kepler_j2::KeplerJ2;
    use crate::ukf::{UkfParams, UnscentedKalmanFilter};
    use thresh_core::eci::EARTH_ROTATION_RATE;
    use thresh_core::orbital::atmosphere_density;

    const EARTH: GravityModel = GravityModel::EARTH_WGS84;
    const BETA: f64 = 2_000.0;

    fn model() -> BallisticReentry {
        BallisticReentry::new(EARTH, 1.0)
    }

    /// Reentry-interface-like 7D state at the given altitude: mostly
    /// horizontal 7 km/s with a descending radial component.
    fn reentry_state(alt_m: f64) -> DVector<f64> {
        let r = EARTH.equatorial_radius + alt_m;
        DVector::from_column_slice(&[r, -2_000.0, 0.0, 6_700.0, 0.0, 500.0, BETA])
    }

    fn assert_symmetric_positive_definite(p: &DMatrix<f64>, label: &str) {
        let asym = (p - p.transpose()).amax();
        assert!(asym < 1e-6 * p.amax(), "{label}: asymmetry {asym:e}");
        assert!(
            p.clone().cholesky().is_some(),
            "{label}: covariance is not positive definite"
        );
    }

    // ── Process noise ────────────────────────────────────────────────────

    #[test]
    fn process_noise_has_wna_blocks_plus_beta_random_walk() {
        let m = model();
        let dt = 2.0;
        let q = m.process_noise(dt);
        let psd = m.sigma_accel * m.sigma_accel;
        // Leading 6×6: the shared CWNA blocks.
        assert!((q[(0, 0)] - psd * dt.powi(3) / 3.0).abs() < 1e-12);
        assert!((q[(1, 1)] - psd * dt).abs() < 1e-12);
        // β row/col: independent random walk, no cross-coupling.
        assert!((q[(6, 6)] - m.sigma_beta * m.sigma_beta * dt).abs() < 1e-15);
        for i in 0..6 {
            assert_eq!(q[(6, i)], 0.0);
            assert_eq!(q[(i, 6)], 0.0);
        }
        assert_symmetric_positive_definite(&q, "process noise");
    }

    // ── Task 3.7: dynamics tests ─────────────────────────────────────────

    // Spec "Drag deceleration matches the beta parameterization": inside the
    // atmosphere the drag magnitude equals ρ(h)·v_rel²/(2β), directed
    // opposite the air-relative velocity v_rel = v − ω_⊕ × r.
    #[test]
    fn drag_deceleration_matches_beta_parameterization() {
        let m = model();
        let alt = 60_000.0;
        let pos = Vector3::new(EARTH.equatorial_radius + alt, 0.0, 0.0);
        let vel = Vector3::new(-1_500.0, 2_500.0, -400.0);

        let drag = m.acceleration(&pos, &vel, BETA) - clamped_gravity_acceleration(&pos, &EARTH);

        let v_atm = Vector3::new(
            -EARTH_ROTATION_RATE * pos.y,
            EARTH_ROTATION_RATE * pos.x,
            0.0,
        );
        let v_rel = vel - v_atm;
        let expected_mag = atmosphere_density(alt) * v_rel.norm_squared() / (2.0 * BETA);

        assert!(
            (drag.norm() - expected_mag).abs() / expected_mag < 1e-12,
            "drag magnitude {} vs ρv²/2β = {expected_mag}",
            drag.norm()
        );
        let cosine = drag.dot(&v_rel) / (drag.norm() * v_rel.norm());
        assert!(
            cosine < -1.0 + 1e-12,
            "drag not anti-parallel to v_rel: cos = {cosine}"
        );
    }

    // Spec "Exoatmospheric limit": above the atmosphere ceiling the drag
    // term vanishes and prediction matches gravity-only propagation.
    #[test]
    fn exoatmospheric_prediction_matches_gravity_only() {
        let reentry = model(); // max_step_s = 1.0
        let mut gravity_only = KeplerJ2::new(EARTH);
        gravity_only.max_step_s = reentry.max_step_s; // matched sub-stepping

        // 1200 km altitude: above the 1000 km atmosphere table ceiling.
        let x7 = reentry_state(1_200_000.0);
        let x6 = DVector::from_column_slice(&x7.as_slice()[..6]);

        let dt = 30.0;
        let next7 = reentry.predict(&x7, dt);
        let next6 = gravity_only.predict(&x6, dt);

        for i in 0..6 {
            assert!(
                (next7[i] - next6[i]).abs() < 1e-9,
                "component {i} differs: {} vs {}",
                next7[i],
                next6[i]
            );
        }
        assert_eq!(next7[6], BETA, "β must be unchanged by predict");
    }

    // Spec "Prediction through all three nonlinear filters": EKF, UKF, and
    // CKF predict from the same reentry-interface state with SPD covariance.
    #[test]
    fn ekf_ukf_ckf_predict_from_reentry_interface_state() {
        let m = model();
        let x0 = reentry_state(120_000.0);
        let mut p0 = DMatrix::zeros(7, 7);
        for i in 0..3 {
            p0[(2 * i, 2 * i)] = 500.0 * 500.0; // position
            p0[(2 * i + 1, 2 * i + 1)] = 100.0 * 100.0; // velocity
        }
        p0[(6, 6)] = 500.0 * 500.0; // wide β prior
        let dt = 1.0;
        let reference = m.predict(&x0, dt);

        let mut ekf = ExtendedKalmanFilter::new(x0.clone(), p0.clone());
        ekf.predict(&m, dt);
        assert!((&ekf.x - &reference).amax() < 1e-9);
        assert_symmetric_positive_definite(&ekf.p, "EKF");

        let mut ukf = UnscentedKalmanFilter::new(x0.clone(), p0.clone(), UkfParams::default());
        ukf.predict(&m, dt);
        assert!((&ukf.x - &reference).amax() < 1.0);
        assert_symmetric_positive_definite(&ukf.p, "UKF");

        let mut ckf = CubatureKalmanFilter::new(x0.clone(), p0.clone());
        ckf.predict(&m, dt);
        assert!((&ckf.x - &reference).amax() < 1.0);
        assert_symmetric_positive_definite(&ckf.p, "CKF");
    }

    // Design risk "Numeric Jacobian conditioning near phase boundaries":
    // at a 100 km-altitude state (steep density gradient) the Jacobian must
    // stay finite and well-conditioned with the default 1 s sub-step.
    #[test]
    fn jacobian_well_conditioned_at_100km_altitude() {
        let m = model();
        let x = reentry_state(100_000.0);
        let f = m.jacobian(&x, 1.0);

        assert!(f.iter().all(|v| v.is_finite()), "Jacobian has NaN/inf");
        let svd = f.clone().svd(false, false);
        let s_max = svd.singular_values.max();
        let s_min = svd.singular_values.min();
        assert!(s_min > 0.0, "Jacobian is singular");
        let cond = s_max / s_min;
        assert!(cond < 1e3, "condition number {cond:e} too large");

        // β sensitivity direction check: raising β lowers drag deceleration,
        // so ∂v/∂β points along +v_rel (∂a_D/∂β = −a_D/β, reference doc).
        let (pos, vel) = unpack_interleaved(&x);
        let v_atm = Vector3::new(
            -EARTH_ROTATION_RATE * pos.y,
            EARTH_ROTATION_RATE * pos.x,
            0.0,
        );
        let v_rel = vel - v_atm;
        let dv_dbeta = Vector3::new(f[(1, 6)], f[(3, 6)], f[(5, 6)]);
        assert!(dv_dbeta.dot(&v_rel) > 0.0, "wrong ∂v/∂β sign");
    }

    // ── Reentry prediction vs truth generation ───────────────────────────

    /// Truth reentry arc from `thresh_synth::propagate` with drag matched to
    /// `BETA`: `cd·area/mass = 1·1/BETA = 1/β` exactly, so the truth and the
    /// filter model integrate the identical shared force closure.
    fn synth_reentry_truth(
        x0: &DVector<f64>,
        duration_s: f64,
        dt_s: f64,
    ) -> Vec<thresh_synth::orbital::OrbitalState> {
        let (pos, vel) = unpack_interleaved(x0);
        let initial = thresh_synth::orbital::OrbitalState::from_cartesian(
            [pos.x, pos.y, pos.z],
            [vel.x, vel.y, vel.z],
            2_451_545.0,
        );
        let config = thresh_synth::orbital::PropagatorConfig {
            include_j2: true,
            drag: Some(thresh_synth::orbital::DragConfig {
                cd: 1.0,
                area_m2: 1.0,
                mass_kg: BETA,
            }),
            dt_s,
        };
        thresh_synth::orbital::propagate(&initial, duration_s, &config, dt_s)
    }

    // Spec "Prediction matches truth with true beta": predicting across a
    // truth-generated reentry segment with the truth's initial state and
    // true β reproduces the truth trajectory. Both sides integrate the same
    // shared force closure with matched 1 s steps, so the documented
    // tolerance is 1e-6 m (Decision 1 physics identity; task 4.7 extends
    // this to the phased `BallisticProfile` generator once it exists).
    #[test]
    fn prediction_matches_truth_with_true_beta() {
        let m = model(); // max_step_s = 1.0
        let x0 = reentry_state(70_000.0);
        let duration = 60.0;

        let truth = synth_reentry_truth(&x0, duration, 1.0);
        let last = truth.last().expect("propagate returns samples");
        // The arc must actually be inside the sensible atmosphere.
        let final_alt = Vector3::from(last.position).norm() - EARTH.equatorial_radius;
        assert!(
            final_alt > 0.0 && final_alt < 100_000.0,
            "arc left the high-drag regime: final altitude {final_alt} m"
        );

        let predicted = m.predict(&x0, duration);
        let (pred_pos, pred_vel) = unpack_interleaved(&predicted);
        for i in 0..3 {
            assert!(
                (pred_pos[i] - last.position[i]).abs() < 1e-6,
                "position component {i} differs by {} m",
                (pred_pos[i] - last.position[i]).abs()
            );
            assert!((pred_vel[i] - last.velocity[i]).abs() < 1e-9);
        }
    }

    /// Shared β-convergence EKF harness for the truth-vs-filter tests: wide
    /// β prior, 1 Hz noise-free position measurements over `truth` (10 m
    /// measurement-noise model), returning the final `|β̂ − BETA|` error.
    fn run_beta_convergence_ekf(
        truth: &[thresh_synth::orbital::OrbitalState],
        x0: DVector<f64>,
    ) -> f64 {
        let m = model();
        let mut p0 = DMatrix::zeros(7, 7);
        for i in 0..3 {
            p0[(2 * i, 2 * i)] = 100.0 * 100.0;
            p0[(2 * i + 1, 2 * i + 1)] = 10.0 * 10.0;
        }
        p0[(6, 6)] = 1_000.0 * 1_000.0; // wide β prior
        let mut ekf = ExtendedKalmanFilter::new(x0, p0);

        let mut h = DMatrix::zeros(3, 7);
        for i in 0..3 {
            h[(i, 2 * i)] = 1.0;
        }
        let r = DMatrix::identity(3, 3) * 100.0;
        for sample in truth.iter().skip(1) {
            ekf.predict(&m, 1.0);
            let z = DVector::from_column_slice(&sample.position);
            ekf.update_linear(&z, &h, &r);
        }
        (ekf.x[6] - BETA).abs()
    }

    // Spec "Beta convergence during high-drag flight": an EKF tracking a
    // truth-generated reentry with position measurements, starting from a β
    // estimate biased 50% high, converges toward true β through the
    // high-drag portion and ends with substantially smaller error (task 4.8
    // re-runs this against the phased `BallisticProfile` generator — see
    // `beta_converges_against_phased_ballistic_truth` below).
    #[test]
    fn beta_converges_during_high_drag_flight() {
        let x0_truth = reentry_state(70_000.0);
        let truth = synth_reentry_truth(&x0_truth, 60.0, 1.0);

        // Filter starts at the true kinematic state but β biased +50%.
        let beta_bias = 0.5 * BETA;
        let mut x0 = x0_truth.clone();
        x0[6] = BETA + beta_bias;
        let final_error = run_beta_convergence_ekf(&truth, x0);

        // Observed final error ~0.01 kg/m² (noise-free measurements); the
        // 5% bound leaves orders-of-magnitude headroom for platform drift.
        assert!(
            final_error < 0.05 * beta_bias,
            "β did not converge: started {beta_bias} kg/m² off, ended {final_error} kg/m² off"
        );
    }

    // ── Tasks 4.7 / 4.8: validation against the phased truth generator ──

    /// MRBM-class phased ballistic truth (`thresh_synth::ballistic`, design
    /// Decision 5) at a 1 s grid, with true β = `BETA`. Returns the samples
    /// and the index of the first reentry-phase sample (post-burnout,
    /// descending below 100 km) — the same event the generator's own phase
    /// classifier uses.
    fn phased_ballistic_truth() -> (Vec<thresh_synth::orbital::OrbitalState>, usize) {
        let profile = thresh_synth::ballistic::BallisticProfile {
            launch_lat_rad: 0.0,
            launch_lon_rad: 0.0,
            launch_alt_m: 0.0,
            launch_azimuth_rad: 90.0_f64.to_radians(),
            thrust_accel: 50.0,
            burn_time_s: 65.0,
            pitch_over_s: 10.0,
            pitch_kick_rad: 0.30,
            beta: BETA,
            epoch_jd: 2_451_545.0,
        };
        let truth = thresh_synth::ballistic::generate(&profile, 1.0);
        let reentry_idx = truth
            .iter()
            .position(|s| {
                let pos = Vector3::from(s.position);
                let vel = Vector3::from(s.velocity);
                let elapsed = (s.epoch_jd - profile.epoch_jd) * 86_400.0;
                elapsed > profile.burn_time_s
                    && pos.dot(&vel) < 0.0
                    && pos.norm() - EARTH.equatorial_radius < 100_000.0
            })
            .expect("phased trajectory must reenter");
        (truth, reentry_idx)
    }

    /// 7D interleaved filter state from a synth Cartesian sample plus β.
    fn state7_from_sample(s: &thresh_synth::orbital::OrbitalState, beta: f64) -> DVector<f64> {
        let [x, y, z] = s.position;
        let [vx, vy, vz] = s.velocity;
        DVector::from_column_slice(&[x, vx, y, vy, z, vz, beta])
    }

    // Spec "Prediction matches truth with true beta", task 4.7: extends
    // `prediction_matches_truth_with_true_beta` (above) from `propagate`
    // truth to the phased `BallisticProfile` generator. The generator's
    // reentry closure is the same `gravity + drag(r, v, 1/β)` this model
    // integrates, with matched 1 s steps, so the documented tolerance stays
    // at float-noise level: 1e-6 m over a 40 s arc.
    #[test]
    fn prediction_matches_phased_ballistic_truth_with_true_beta() {
        let m = model(); // max_step_s = 1.0, matching the truth grid
        let (truth, reentry_idx) = phased_ballistic_truth();
        let span_s = 40usize;
        let window = &truth[reentry_idx..=reentry_idx + span_s];

        // Guard the phase assumption: the whole window is descending
        // reentry between the surface and the 100 km interface.
        for s in window {
            let pos = Vector3::from(s.position);
            let alt = pos.norm() - EARTH.equatorial_radius;
            assert!(alt > 0.0 && alt < 100_000.0, "window left reentry regime");
            assert!(pos.dot(&Vector3::from(s.velocity)) < 0.0);
        }

        let x0 = state7_from_sample(&window[0], BETA);
        let predicted = m.predict(&x0, span_s as f64);
        let (pred_pos, pred_vel) = unpack_interleaved(&predicted);
        let end = window.last().expect("non-empty window");
        for i in 0..3 {
            assert!(
                (pred_pos[i] - end.position[i]).abs() < 1e-6,
                "position component {i} differs by {} m",
                (pred_pos[i] - end.position[i]).abs()
            );
            assert!((pred_vel[i] - end.velocity[i]).abs() < 1e-9);
        }
        assert_eq!(predicted[6], BETA, "β must be unchanged by predict");
    }

    // Spec "Beta convergence during high-drag flight", task 4.8: the same
    // EKF harness as `beta_converges_during_high_drag_flight`, run against
    // the phased generator's reentry arc — position measurements only,
    // starting β biased +50% from truth.
    #[test]
    fn beta_converges_against_phased_ballistic_truth() {
        let (truth, reentry_idx) = phased_ballistic_truth();
        // Track the reentry arc down to the last above-surface sample.
        let end = truth
            .iter()
            .rposition(|s| Vector3::from(s.position).norm() > EARTH.equatorial_radius)
            .expect("samples above the surface");
        let arc = &truth[reentry_idx..=end];
        assert!(arc.len() > 30, "reentry arc too short: {} s", arc.len());

        let beta_bias = 0.5 * BETA;
        let x0 = state7_from_sample(&arc[0], BETA + beta_bias);
        let final_error = run_beta_convergence_ekf(arc, x0);

        // Observed final error ~0.2 kg/m² on this steeper, faster arc; the
        // 5% bound matches the propagate-truth twin test's headroom.
        assert!(
            final_error < 0.05 * beta_bias,
            "β did not converge: started {beta_bias} kg/m² off, ended {final_error} kg/m² off"
        );
    }

    // ── Task 3.8: β positivity via the floor ─────────────────────────────

    // Spec "Beta stays positive": a measurement update that drives the β
    // estimate negative leaves the effective β strictly positive — the
    // floor inside predict re-emits a physical β and the dynamics stay
    // well-defined.
    #[test]
    fn beta_floor_keeps_effective_beta_positive() {
        let m = model();
        let x0 = reentry_state(80_000.0);
        let p0 = DMatrix::identity(7, 7) * 1e4;
        let mut ekf = ExtendedKalmanFilter::new(x0, p0);
        ekf.predict(&m, 1.0);

        // Adversarial update: observe β directly with a wildly negative
        // value and near-zero noise.
        let mut h = DMatrix::zeros(1, 7);
        h[(0, 6)] = 1.0;
        let z = DVector::from_column_slice(&[-5_000.0]);
        let r = DMatrix::identity(1, 1) * 1e-6;
        ekf.update_linear(&z, &h, &r);
        assert!(ekf.x[6] < 0.0, "update should have driven β negative");

        // The next predict floors the effective β and keeps the state finite.
        ekf.predict(&m, 1.0);
        assert!(ekf.x[6] >= m.beta_floor);
        assert!(ekf.x[6] > 0.0, "effective β must be strictly positive");
        assert!(ekf.x.iter().all(|v| v.is_finite()));

        // The model itself also floors on a direct call.
        let poisoned = DVector::from_column_slice(&[
            EARTH.equatorial_radius + 80_000.0,
            -2_000.0,
            0.0,
            6_700.0,
            0.0,
            500.0,
            -123.0,
        ]);
        let next = m.predict(&poisoned, 1.0);
        assert_eq!(next[6], m.beta_floor);
    }
}
