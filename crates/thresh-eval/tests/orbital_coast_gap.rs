//! Coast-gap ANEES demonstration (element-space-orbital-filtering tasks 4.1–4.2,
//! design Decision 5).
//!
//! A seeded Monte-Carlo experiment that measures the change's payoff. Identical
//! deterministic truth arcs (Cartesian two-body+J2) are tracked with identical
//! ECI-position measurements under a **shared physical process-noise assumption**
//! (one isotropic acceleration PSD `σ_accel`, mapped into each filter's
//! coordinates — `KeplerJ2`'s Cartesian CWNA block and the equinoctial GVE
//! input-matrix map are the two forms of the *same* Stacey & D'Amico Eq. 5;
//! see `EquinoctialModel`'s docs). After a warmup, both filters coast through
//! measurement-free gaps of growing duration; at each gap end the
//! position-marginal ANEES (dof 3, in the common ECI position space) is measured
//! over `RUNS` seeded runs and judged against the two-sided 95% χ² band.
//!
//! # What the experiment isolates (design Decision 5)
//!
//! Decision 5's claim is specifically about **coast** — measurement-free
//! covariance propagation: element-space uncertainty stays Gaussian for many
//! revolutions while the Cartesian along-track uncertainty curves into a
//! "banana" that a linearized (EKF) covariance cannot represent. So the warmup
//! establishes one shared, consistent posterior (via the Cartesian EKF that is
//! also the baseline), and at gap start that posterior is expressed in each
//! filter's coordinates — kept as-is for the Cartesian EKF, and mapped to
//! equinoctial elements by an unscented transform for the equinoctial UKF (which
//! the diagnostics behind this test confirmed is consistent: element-space NEES
//! ≈ dof at gap start). Both then coast predict-only. This deliberately keeps the
//! comparison to the coast-phase covariance propagation the change is about, and
//! away from the (separate, known) trade-off that filtering ECI **position**
//! measurements — linear in Cartesian, nonlinear in elements — is statistically
//! harder in element space; that measurement-update behaviour is out of this
//! demonstration's scope. Truth is propagated by the very `KeplerJ2::predict`
//! the EKF uses (zero dynamics model error there); the equinoctial model matches
//! it to the sub-metre cross-formulation tolerance over the swept horizons,
//! negligible against the metre-to-kilometre coast covariance.
//!
//! # Result (the recorded sweep, RUNS = 50, seed = 7)
//!
//! The Cartesian EKF's along-track banana makes its position ANEES exceed the
//! band once per revolution (the banana's major axis rotating with the orbit),
//! while the equinoctial UKF stays inside the band throughout. Two clean
//! crossover gaps (Cartesian out, equinoctial in): **2.39 rev** (EKF 4.76,
//! band ≈ [2.36, 3.72], UKF 2.85) and **3.35 rev** (EKF 4.37, UKF 2.89). The
//! full sweep is printed by the test and asserted below.

use nalgebra::{DMatrix, DVector, Vector3};
use rand::prelude::*;

use thresh_eval::consistency::{ConsistencyAccumulator, ConsistencyVerdict, chi2, nees};
use thresh_filter::ekf::ExtendedKalmanFilter;
use thresh_filter::models::equinoctial::{EquinoctialModel, equinoctial_position};
use thresh_filter::models::kepler_j2::KeplerJ2;
use thresh_filter::traits::MotionModel;
use thresh_filter::ukf::{UkfParams, UnscentedKalmanFilter};

use thresh_core::orbital::{
    Frame, GravityModel, OrbitalElements, OrbitalState, keplerian_to_cartesian,
};
use thresh_core::time::Epoch;

// ── Scenario constants (the resolved sweep — recorded in design.md, task 4.1) ─

const EARTH: GravityModel = GravityModel::EARTH_WGS84;
const MU: f64 = GravityModel::EARTH_WGS84.mu;
/// Fixed J2000 epoch for the pure-geometry element conversions.
const CONVERSION_EPOCH_JD: f64 = 2_451_545.0;

/// Predict / measurement cadence (s).
const DT_STEP: f64 = 60.0;
/// RK4 sub-step ceiling shared by truth and both filters (s). At 15 s the
/// equinoctial GVE path matches the Cartesian truth to well under a metre over
/// the swept horizons — negligible against the coast covariance.
const MAX_STEP_S: f64 = 15.0;
/// Warmup measurement updates establishing the shared posterior.
const WARMUP_STEPS: usize = 5;
/// Monte-Carlo runs per gap (ANEES sample count at each checkpoint).
const RUNS: usize = 50;
/// Initial 1σ position uncertainty per axis (m).
const SIGMA_R: f64 = 3000.0;
/// Initial 1σ velocity uncertainty per axis (m/s) — large, so the along-track
/// banana forms within a few revolutions (the coast regime Decision 5 targets).
const SIGMA_V: f64 = 60.0;
/// ECI-position measurement 1σ per axis (m). Loose warmup keeps the shared
/// posterior large enough that the banana breaks the Cartesian EKF in a few
/// revolutions.
const SIGMA_Z: f64 = 200.0;
/// Shared isotropic acceleration white-noise PSD (m/s²·Hz^-½).
const SIGMA_ACCEL: f64 = 1e-6;
/// Fixed seed — the demonstration is deterministic, not statistical luck.
const SEED: u64 = 7;

/// Coast-gap checkpoints in predict steps (0 = gap start, at warmup end). The
/// LEO period is ≈ 92.7 steps of 60 s, so this sweeps 0 → ~4 revolutions at
/// roughly half-revolution spacing.
const CHECKPOINTS: [usize; 9] = [0, 46, 93, 139, 185, 232, 278, 325, 371];

// ── Truth orbit and small conversion helpers ─────────────────────────────────

/// Interleaved `[x, vx, y, vy, z, vz]` initial truth state: LEO, a = 7000 km,
/// e = 0.001, i = 51.6°.
fn truth0() -> DVector<f64> {
    let (r, v) =
        keplerian_to_cartesian(7_000_000.0, 0.001, 51.6_f64.to_radians(), 0.3, 0.4, 0.5, MU);
    DVector::from_row_slice(&[r.x, v.x, r.y, v.y, r.z, v.z])
}

fn to_posvel(x: &DVector<f64>) -> (Vector3<f64>, Vector3<f64>) {
    (
        Vector3::new(x[0], x[2], x[4]),
        Vector3::new(x[1], x[3], x[5]),
    )
}

/// Deterministic Cartesian two-body+J2 truth, propagated by the same
/// `KeplerJ2::predict` the Cartesian EKF uses (so that filter carries zero
/// dynamics model error). Returns state at every step time `i·DT_STEP`.
fn build_truth() -> Vec<(Vector3<f64>, Vector3<f64>)> {
    let model = KeplerJ2 {
        gravity: EARTH,
        max_step_s: MAX_STEP_S,
        sigma_accel: 0.0,
    };
    let total = WARMUP_STEPS + CHECKPOINTS[CHECKPOINTS.len() - 1];
    let mut x = truth0();
    let mut out = Vec::with_capacity(total + 1);
    out.push(to_posvel(&x));
    for _ in 0..total {
        x = model.predict(&x, DT_STEP);
        out.push(to_posvel(&x));
    }
    out
}

/// Interleaved Cartesian state → equinoctial element vector `[a, h, k, p, q, λ]`.
fn cartesian_to_equinoctial_vec(cart: &DVector<f64>) -> DVector<f64> {
    let (r, v) = to_posvel(cart);
    let state = OrbitalState {
        elements: OrbitalElements::Cartesian {
            position: r,
            velocity: v,
        },
        mu: MU,
        frame: Frame::Teme,
        epoch: Epoch::from_jde_utc(CONVERSION_EPOCH_JD),
    };
    let (a, h, k, p, q, l) = state.as_equinoctial().unwrap();
    DVector::from_row_slice(&[a, h, k, p, q, l])
}

/// Standard normal via Box–Muller on the seeded uniform stream.
fn gauss(rng: &mut StdRng) -> f64 {
    let u1: f64 = rng.random();
    let u2: f64 = rng.random();
    (-2.0 * (1.0 - u1).ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
}

// ── Unscented transform (Van der Merwe), used for the element filter's
//    initialization from the shared posterior and for the element→position
//    projection at NEES time ────────────────────────────────────────────────

/// Unscented transform of `N(mean, cov)` through `f`, returning the mapped mean
/// and covariance. `alpha = 1`, `beta = 2`, `kappa = 0`: exact for a linear
/// `f`, second-order for a nonlinear one.
fn unscented_transform<F>(
    mean: &DVector<f64>,
    cov: &DMatrix<f64>,
    f: F,
) -> (DVector<f64>, DMatrix<f64>)
where
    F: Fn(&DVector<f64>) -> DVector<f64>,
{
    let n = mean.len();
    let (alpha, beta, kappa) = (1.0_f64, 2.0_f64, 0.0_f64);
    let lambda = alpha * alpha * (n as f64 + kappa) - n as f64;
    let scale = n as f64 + lambda;
    let l = (cov * scale)
        .cholesky()
        .expect("UT covariance is positive definite")
        .l();

    let mut points = Vec::with_capacity(2 * n + 1);
    points.push(mean.clone());
    for i in 0..n {
        let col = l.column(i).clone_owned();
        points.push(mean + &col);
        points.push(mean - &col);
    }
    let mapped: Vec<DVector<f64>> = points.iter().map(&f).collect();

    let wm0 = lambda / scale;
    let wc0 = wm0 + (1.0 - alpha * alpha + beta);
    let w = 1.0 / (2.0 * scale);

    let mut out_mean = wm0 * &mapped[0];
    for point in &mapped[1..] {
        out_mean += w * point;
    }
    let d0 = &mapped[0] - &out_mean;
    let mut out_cov = wc0 * &d0 * d0.transpose();
    for point in &mapped[1..] {
        let d = point - &out_mean;
        out_cov += w * &d * d.transpose();
    }
    (out_mean, out_cov)
}

// ── Per-run sampling and the shared-warmup coast ─────────────────────────────

/// Diagonal interleaved initial Cartesian covariance `diag(σ_r², σ_v², …)`.
fn p0_cartesian() -> DMatrix<f64> {
    DMatrix::from_diagonal(&DVector::from_row_slice(&[
        SIGMA_R * SIGMA_R,
        SIGMA_V * SIGMA_V,
        SIGMA_R * SIGMA_R,
        SIGMA_V * SIGMA_V,
        SIGMA_R * SIGMA_R,
        SIGMA_V * SIGMA_V,
    ]))
}

/// Sample one run's initial Cartesian estimate `N(truth0, P0)` and its warmup
/// measurement sequence (`truth_pos + N(0, σ_z²I)`).
fn sample_run(
    rng: &mut StdRng,
    truth: &[(Vector3<f64>, Vector3<f64>)],
) -> (DVector<f64>, Vec<DVector<f64>>) {
    let (r0, v0) = truth[0];
    let x0 = DVector::from_row_slice(&[
        r0.x + SIGMA_R * gauss(rng),
        v0.x + SIGMA_V * gauss(rng),
        r0.y + SIGMA_R * gauss(rng),
        v0.y + SIGMA_V * gauss(rng),
        r0.z + SIGMA_R * gauss(rng),
        v0.z + SIGMA_V * gauss(rng),
    ]);
    let mut meas = Vec::with_capacity(WARMUP_STEPS);
    for k in 0..WARMUP_STEPS {
        let p = truth[k + 1].0;
        meas.push(DVector::from_row_slice(&[
            p.x + SIGMA_Z * gauss(rng),
            p.y + SIGMA_Z * gauss(rng),
            p.z + SIGMA_Z * gauss(rng),
        ]));
    }
    (x0, meas)
}

/// 3×6 position selector for the interleaved Cartesian state.
fn position_selector() -> DMatrix<f64> {
    DMatrix::from_row_slice(
        3,
        6,
        &[
            1.0, 0.0, 0.0, 0.0, 0.0, 0.0, //
            0.0, 0.0, 1.0, 0.0, 0.0, 0.0, //
            0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
        ],
    )
}

fn measurement_noise() -> DMatrix<f64> {
    DMatrix::identity(3, 3) * (SIGMA_Z * SIGMA_Z)
}

fn unpack6(state: &DVector<f64>) -> [f64; 6] {
    [state[0], state[1], state[2], state[3], state[4], state[5]]
}

/// Position-marginal NEES (dof 3) of the interleaved Cartesian EKF against a
/// truth position: error `truth − estimate`, marginal 3×3 position block of P.
fn cartesian_position_nees(ekf: &ExtendedKalmanFilter, truth_pos: &Vector3<f64>) -> f64 {
    let idx = [0usize, 2, 4];
    let error = DVector::from_row_slice(&[
        truth_pos.x - ekf.x[0],
        truth_pos.y - ekf.x[2],
        truth_pos.z - ekf.x[4],
    ]);
    let p_pos = DMatrix::from_fn(3, 3, |i, j| ekf.p[(idx[i], idx[j])]);
    nees(&error, &p_pos).expect("Cartesian position block invertible")
}

/// Position-marginal NEES (dof 3) of the element UKF against a truth position:
/// the element (mean, covariance) are projected into ECI position space by the
/// unscented transform, then scored — the same Gaussian projection the linear
/// Cartesian marginal block is for the EKF.
fn element_position_nees(ukf: &UnscentedKalmanFilter, truth_pos: &Vector3<f64>) -> f64 {
    let (pos_mean, p_pos) =
        unscented_transform(&ukf.x, &ukf.p, |e| equinoctial_position(&unpack6(e), MU));
    let error = DVector::from_row_slice(&[
        truth_pos.x - pos_mean[0],
        truth_pos.y - pos_mean[1],
        truth_pos.z - pos_mean[2],
    ]);
    nees(&error, &p_pos).expect("element position projection invertible")
}

/// One seeded run: shared Cartesian warmup, then coast both filters predict-only,
/// recording position NEES at each checkpoint. Returns
/// `(cartesian_nees_per_checkpoint, equinoctial_nees_per_checkpoint)`.
fn coast_run(
    rng: &mut StdRng,
    truth: &[(Vector3<f64>, Vector3<f64>)],
    reference_elements: [f64; 6],
) -> (Vec<f64>, Vec<f64>) {
    let (x0, meas) = sample_run(rng, truth);
    let kepler = KeplerJ2 {
        gravity: EARTH,
        max_step_s: MAX_STEP_S,
        sigma_accel: SIGMA_ACCEL,
    };
    let equinoctial = EquinoctialModel {
        gravity: EARTH,
        max_step_s: MAX_STEP_S,
        sigma_accel: SIGMA_ACCEL,
        reference_elements,
    };
    let h = position_selector();
    let r = measurement_noise();

    // Shared warmup: the Cartesian EKF establishes the posterior at gap start.
    let mut ekf = ExtendedKalmanFilter::new(x0, p0_cartesian());
    for z in &meas {
        ekf.predict(&kepler, DT_STEP);
        ekf.update_linear(z, &h, &r);
    }
    // The equinoctial UKF starts from that same posterior, mapped to elements.
    let (x0e, p0e) = unscented_transform(&ekf.x, &ekf.p, cartesian_to_equinoctial_vec);
    let mut ukf = UnscentedKalmanFilter::new(
        x0e,
        p0e,
        UkfParams {
            alpha: 1.0,
            beta: 2.0,
            kappa: 0.0,
        },
    );

    let mut cart_nees = Vec::with_capacity(CHECKPOINTS.len());
    let mut equi_nees = Vec::with_capacity(CHECKPOINTS.len());
    let max_coast = CHECKPOINTS[CHECKPOINTS.len() - 1];
    for c in 0..=max_coast {
        if c > 0 {
            ekf.predict(&kepler, DT_STEP);
            // Re-anchor the SNC input matrix at the current mean so Q's
            // element-space orientation tracks orbital phase through the
            // multi-revolution coast (see EquinoctialModel::re_anchored).
            let step_model = equinoctial.re_anchored(&ukf.x);
            ukf.predict(&step_model, DT_STEP);
        }
        if CHECKPOINTS.contains(&c) {
            let truth_pos = &truth[WARMUP_STEPS + c].0;
            cart_nees.push(cartesian_position_nees(&ekf, truth_pos));
            equi_nees.push(element_position_nees(&ukf, truth_pos));
        }
    }
    (cart_nees, equi_nees)
}

/// Sweep: accumulate per-checkpoint position ANEES for both filters over `RUNS`
/// seeded runs. Returns `(cartesian_accumulators, equinoctial_accumulators)`.
fn run_sweep(
    truth: &[(Vector3<f64>, Vector3<f64>)],
    reference_elements: [f64; 6],
) -> (Vec<ConsistencyAccumulator>, Vec<ConsistencyAccumulator>) {
    let mut rng = StdRng::seed_from_u64(SEED);
    let mut cart_accs: Vec<ConsistencyAccumulator> = CHECKPOINTS
        .iter()
        .map(|_| ConsistencyAccumulator::new(3))
        .collect();
    let mut equi_accs: Vec<ConsistencyAccumulator> = cart_accs.clone();
    for _ in 0..RUNS {
        let (cart_nees, equi_nees) = coast_run(&mut rng, truth, reference_elements);
        for i in 0..CHECKPOINTS.len() {
            cart_accs[i].push(cart_nees[i]);
            equi_accs[i].push(equi_nees[i]);
        }
    }
    (cart_accs, equi_accs)
}

fn reference_elements(truth: &[(Vector3<f64>, Vector3<f64>)]) -> [f64; 6] {
    let (r, v) = truth[0];
    let cart = DVector::from_row_slice(&[r.x, v.x, r.y, v.y, r.z, v.z]);
    let e = cartesian_to_equinoctial_vec(&cart);
    [e[0], e[1], e[2], e[3], e[4], e[5]]
}

// ── The demonstration test ──────────────────────────────────────────────────

#[test]
fn coast_gap_element_filter_outlasts_cartesian() {
    let truth = build_truth();
    let ref_elem = reference_elements(&truth);
    let (cart_accs, equi_accs) = run_sweep(&truth, ref_elem);
    let period = std::f64::consts::TAU * (7_000_000_f64.powi(3) / MU).sqrt();

    // Full sweep table (spec: "full sweep values recorded at the test").
    eprintln!(
        "coast-gap sweep: RUNS={RUNS}, dof=3, seed={SEED}, σ_r={SIGMA_R} m, σ_v={SIGMA_V} m/s, \
         σ_z={SIGMA_Z} m, σ_accel={SIGMA_ACCEL:e}, cadence={DT_STEP} s, warmup={WARMUP_STEPS}"
    );
    eprintln!(
        "{:>8} {:>6} {:>10} {:>14} {:>10} {:>14} {:>18}",
        "gap_s", "rev", "EKF_ANEES", "EKF_verdict", "UKF_ANEES", "UKF_verdict", "band"
    );
    let mut crossover = None;
    for i in 0..CHECKPOINTS.len() {
        let gap_s = CHECKPOINTS[i] as f64 * DT_STEP;
        let ekf_mean = cart_accs[i].mean().unwrap();
        let ukf_mean = equi_accs[i].mean().unwrap();
        let ekf_v = cart_accs[i].verdict(chi2::DEFAULT_ALPHA).unwrap();
        let ukf_v = equi_accs[i].verdict(chi2::DEFAULT_ALPHA).unwrap();
        let (lo, hi) = cart_accs[i].bounds(chi2::DEFAULT_ALPHA).unwrap();
        eprintln!(
            "{gap_s:>8.0} {:>6.2} {ekf_mean:>10.4} {:>14} {ukf_mean:>10.4} {:>14} {:>8}",
            gap_s / period,
            format!("{ekf_v:?}"),
            format!("{ukf_v:?}"),
            format!("[{lo:.3},{hi:.3}]"),
        );
        if crossover.is_none()
            && ekf_v != ConsistencyVerdict::Consistent
            && ukf_v == ConsistencyVerdict::Consistent
        {
            crossover = Some((gap_s / period, ekf_mean, ukf_mean, lo, hi));
        }
    }

    // Spec "Element filter outlasts the Cartesian filter through coast": a
    // recorded gap where the Cartesian ANEES is out of band while the element
    // ANEES is inside, with both values recorded.
    let (rev, ekf_mean, ukf_mean, lo, hi) =
        crossover.expect("no gap found where Cartesian is out of band and element is in band");
    eprintln!(
        "CROSSOVER at {rev:.2} rev: Cartesian ANEES {ekf_mean:.4} out of band [{lo:.4}, {hi:.4}] \
         (margin {:.4} above upper); equinoctial ANEES {ukf_mean:.4} inside (margin {:.4} above \
         lower, {:.4} below upper).",
        ekf_mean - hi,
        ukf_mean - lo,
        hi - ukf_mean
    );
    assert!(
        ekf_mean > hi,
        "Cartesian ANEES {ekf_mean} must exceed upper bound {hi}"
    );
    assert!(
        ukf_mean >= lo && ukf_mean <= hi,
        "equinoctial ANEES {ukf_mean} must be inside [{lo}, {hi}]"
    );

    // The stated result is stronger than one crossover: the equinoctial UKF
    // stays inside the band at EVERY swept gap — assert the whole sweep so a
    // regression at any other checkpoint cannot hide behind the crossover.
    for (i, acc) in equi_accs.iter().enumerate() {
        assert_eq!(
            acc.verdict(chi2::DEFAULT_ALPHA).unwrap(),
            ConsistencyVerdict::Consistent,
            "equinoctial ANEES must be in band at checkpoint {i} (gap {} s), got {:?}",
            CHECKPOINTS[i] as f64 * DT_STEP,
            acc.mean()
        );
    }
    // Both recorded Cartesian excursions (the once-per-revolution banana
    // rotation) are part of the recorded, bitwise-deterministic result.
    for &idx in &[5usize, 7usize] {
        assert_ne!(
            cart_accs[idx].verdict(chi2::DEFAULT_ALPHA).unwrap(),
            ConsistencyVerdict::Consistent,
            "Cartesian ANEES expected out of band at checkpoint {idx}"
        );
    }

    // Spec "Demonstration is deterministic": a second seeded sweep reproduces
    // every accumulated ANEES bit-for-bit.
    let (cart2, equi2) = run_sweep(&truth, ref_elem);
    for i in 0..CHECKPOINTS.len() {
        assert_eq!(
            cart_accs[i].sum.to_bits(),
            cart2[i].sum.to_bits(),
            "EKF sum {i}"
        );
        assert_eq!(
            equi_accs[i].sum.to_bits(),
            equi2[i].sum.to_bits(),
            "UKF sum {i}"
        );
        assert_eq!(cart_accs[i].n, cart2[i].n);
        assert_eq!(equi_accs[i].n, equi2[i].n);
    }
}
