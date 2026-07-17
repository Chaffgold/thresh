//! Gauss variational equations (GVE) for the stored direct/prograde
//! equinoctial element set `(a, h, k, p, q, λ)`.
//!
//! This module supplies the element-space dynamics counterpart to the
//! Cartesian [`crate::orbital::integrate`] tier: the per-element rate law
//! driven by a *perturbing* acceleration, plus a sub-stepped fixed-step RK4
//! loop over those rates (mirroring `integrate.rs`'s role for the second-order
//! Cartesian system). The stored set is nonsingular at zero eccentricity and
//! zero inclination; the retrograde geometry `i = π` is out of scope
//! (`tan(i/2)` — hence `p`, `q` — diverges there).
//!
//! # Source of the equations (provenance — never from memory)
//!
//! The element-rate equations are transcribed from **Appendix A of** N. Stacey
//! & S. D'Amico, *"Analytical Process Noise Covariance Modeling for Absolute and
//! Relative Orbits"* (arXiv:2105.06516, Acta Astronautica; PDF fetched
//! 2026-07-16 from <https://arxiv.org/pdf/2105.06516>), which states the Gauss
//! variational equations "formulated in equinoctial elements" for exactly this
//! set (their Eq. 42 defines `xE = (a, f, g, h, k, λ)`; Eq. 44–45 give
//! `dxE/dt = G(xE)·d + [0,…,0,n]ᵀ`; Eq. A.1 gives the scalar entries of `G`).
//! That appendix cites R. H. Battin, *An Introduction to the Mathematics and
//! Methods of Astrodynamics* (ref. 32) and E. A. Roth, *"The Gaussian form of
//! the variation-of-parameter equations formulated in equinoctial elements"*
//! (ref. 33), and explicitly notes that **Roth has a sign error on H̄** — the
//! form transcribed here is the appendix's corrected one, not Roth's.
//!
//! # Naming map: fetched source ⟷ this crate's stored set
//!
//! The fetched source names the eccentricity/inclination components `(f, g, h,
//! k)`; [`crate::orbital::OrbitalElements::Equinoctial`] stores `(h, k, p, q)`.
//! They are the same six numbers under this exact renaming:
//!
//! | fetched source                | this crate's stored field |
//! |-------------------------------|---------------------------|
//! | `f = e·cos(ω+Ω)`              | `k`                       |
//! | `g = e·sin(ω+Ω)`              | `h`                       |
//! | `h = tan(i/2)·cosΩ`          | `q`                       |
//! | `k = tan(i/2)·sinΩ`          | `p`                       |
//! | `a`, `λ = M+ω+Ω`             | `a`, `mean_longitude`     |
//!
//! Every equation below is written in **this crate's stored naming** with the
//! substitution already applied, and each helper's doc quotes the source form
//! it came from so the transcription is auditable line by line.
//!
//! # Perturbing-acceleration seam (design Decision 2)
//!
//! The GVE consume the *perturbing* acceleration only — the two-body term is
//! the analytic secular rate `dλ/dt ⊇ n(a) = √(µ/a³)`, never part of the
//! perturbation input. The evaluator takes a time-aware inertial closure
//! `Fn(t, &r, &v) -> a_pert` and rotates it into the RSW frame internally, so
//! any force the force stack can express can later drive the GVE. For this
//! change the perturbation is J2 alone; [`j2_perturbation_acceleration`]
//! builds it (see its note on the historical `j2_acceleration` composite).

use nalgebra::Vector3;

use crate::orbital::gravity::{GravityModel, j2_acceleration, two_body_acceleration};
use crate::orbital::state::{Frame, OrbitalElements, OrbitalState, keplerian_to_cartesian};
use crate::time::Epoch;

/// Arbitrary fixed Julian date (J2000.0) for the throwaway [`OrbitalState`]
/// built purely to reuse the crate's tested element conversions. The
/// equinoctial↔Keplerian↔Cartesian conversions are pure geometry — they read
/// neither the epoch nor the frame tag — so this value never affects any rate.
const CONVERSION_EPOCH_JD: f64 = 2_451_545.0;

// ---------------------------------------------------------------------------
// RSW (radial / transverse / normal) rotation
// ---------------------------------------------------------------------------

/// Right-handed RSW orbital basis built from an inertial position/velocity
/// pair: `R̂ = r̂` (radial), `Ŵ = (r×v)/‖r×v‖` (normal, along the
/// angular-momentum vector), `Ŝ = Ŵ × R̂` (transverse, completing the triad
/// toward the velocity).
///
/// This is the RTN frame of the fetched source (its Eq. B.2 rotation
/// `R_{I→R} = [r̂ᵀ; (n̂×r̂)ᵀ; n̂ᵀ]` with `n̂ = (r×v)/‖r×v‖`).
///
/// Domain: undefined only for the degenerate `‖r‖ = 0` or `‖r×v‖ = 0`
/// (rectilinear) states, which no bound orbit produces.
#[derive(Debug, Clone, Copy)]
pub struct RswBasis {
    /// Radial unit vector `R̂ = r̂`.
    pub radial: Vector3<f64>,
    /// Transverse unit vector `Ŝ = Ŵ × R̂`.
    pub transverse: Vector3<f64>,
    /// Normal unit vector `Ŵ = (r×v)/‖r×v‖`.
    pub normal: Vector3<f64>,
}

impl RswBasis {
    /// Construct the RSW basis from inertial position and velocity.
    pub fn from_state(r: &Vector3<f64>, v: &Vector3<f64>) -> Self {
        let radial = r.normalize();
        let normal = r.cross(v).normalize();
        let transverse = normal.cross(&radial);
        Self {
            radial,
            transverse,
            normal,
        }
    }

    /// Project an inertial acceleration onto the basis, returning the
    /// `(radial, transverse, normal)` = `(d_r, d_t, d_n)` components the GVE
    /// consume.
    pub fn project(&self, accel: &Vector3<f64>) -> (f64, f64, f64) {
        (
            accel.dot(&self.radial),
            accel.dot(&self.transverse),
            accel.dot(&self.normal),
        )
    }
}

// ---------------------------------------------------------------------------
// Perturbation seam
// ---------------------------------------------------------------------------

/// The **J2 perturbation acceleration alone** (the oblateness term, two-body
/// removed) — the perturbing acceleration the GVE consume for this change.
///
/// Note the seam: [`j2_acceleration`] returns two-body **plus** J2 combined
/// (documented historical behaviour in [`crate::orbital::gravity`], unlike
/// `j3_acceleration`/`j4_acceleration` which return their term alone), so the
/// pure J2 perturbation is that minus [`two_body_acceleration`]. This is the
/// same subtraction the gravity module's own tests use to isolate the
/// perturbation. Two-body must **not** appear here — it is the analytic
/// `dλ/dt ⊇ √(µ/a³)` secular term inside the GVE.
pub fn j2_perturbation_acceleration(pos: &Vector3<f64>, gravity: &GravityModel) -> Vector3<f64> {
    j2_acceleration(pos, gravity) - two_body_acceleration(pos, gravity)
}

// ---------------------------------------------------------------------------
// Auxiliary scalars for one rate evaluation
// ---------------------------------------------------------------------------

/// Everything a single element-rate evaluation needs at the current state:
/// the stored elements, the current inertial `(r, v)`, the true longitude's
/// sine/cosine, and the derived scalars shared across the coefficients.
///
/// Grouped into a struct rather than plumbed as arguments to keep each rate
/// helper's signature small (avoiding `clippy::too_many_arguments`) while
/// letting the top-level evaluator stay a flat sequence of phase calls.
struct GveAux {
    /// Semi-major axis `a` (m).
    a: f64,
    /// Stored `h = e·sin(ω+Ω)` (fetched source's `g`).
    h: f64,
    /// Stored `k = e·cos(ω+Ω)` (fetched source's `f`).
    k: f64,
    /// Stored `p = tan(i/2)·sinΩ` (fetched source's `k`).
    p: f64,
    /// Stored `q = tan(i/2)·cosΩ` (fetched source's `h`).
    q: f64,
    /// `sin l`, `cos l` of the true longitude `l = ν + ω + Ω`.
    sin_l: f64,
    cos_l: f64,
    /// `W = 1 + k·cos l + h·sin l = 1 + e·cos ν` (`= p_slr/r`). Positive for
    /// every bound orbit (`e < 1`); the singular denominator in most
    /// coefficients. Vanishes only at apoapsis of a parabola (`e = 1`).
    w: f64,
    /// `η = √(1 − e²)`, in `(0, 1]`; `1 + η ∈ [1, 2]` is never zero.
    eta: f64,
    /// Specific angular momentum `L = √(µ·p_slr) = ‖r×v‖`. Vanishes only as
    /// `e → 1` (`p_slr → 0`); the leading denominator of `Ā`, `B̄`, `K̄`.
    ang_mom: f64,
    /// Mean motion `n = √(µ/a³)`.
    n: f64,
    /// `√(p_slr/µ)`, the common leading factor of the eccentricity/inclination
    /// coefficients.
    sqrt_p_mu: f64,
    /// Gravitational parameter `µ` (m³/s²).
    mu: f64,
    /// Inertial position (m) at the current elements.
    r: Vector3<f64>,
    /// Inertial velocity (m/s) at the current elements.
    v: Vector3<f64>,
}

impl GveAux {
    /// Build the auxiliaries from the stored elements `[a, h, k, p, q, λ]`.
    ///
    /// The true longitude and `(r, v)` come from the crate's tested element
    /// conversions (equinoctial → Keplerian → Cartesian); `e`, `η`, `p_slr`,
    /// `L`, `n` come straight from the stored `a, h, k` (exact, no conversion).
    fn from_elements(elements: &[f64; 6], mu: f64) -> Self {
        let [a, h, k, p, q, mean_longitude] = *elements;
        let state = OrbitalState {
            elements: OrbitalElements::Equinoctial {
                sma: a,
                h,
                k,
                p,
                q,
                mean_longitude,
            },
            mu,
            frame: Frame::Teme,
            epoch: Epoch::from_jde_utc(CONVERSION_EPOCH_JD),
        };
        // Equinoctial → Keplerian is infallible (never the `Tle` variant), so
        // the true longitude l = ν + ω + Ω is always available.
        let (sma, ecc, inc, raan, argp, true_anomaly) = state
            .as_keplerian()
            .expect("equinoctial elements always convert to Keplerian");
        let true_longitude = true_anomaly + argp + raan;
        let (r, v) = keplerian_to_cartesian(sma, ecc, inc, raan, argp, true_anomaly, mu);

        let (sin_l, cos_l) = true_longitude.sin_cos();
        let ecc2 = h * h + k * k;
        let eta = (1.0 - ecc2).sqrt();
        let semi_latus_rectum = a * (1.0 - ecc2);
        let ang_mom = (mu * semi_latus_rectum).sqrt();
        let n = (mu / (a * a * a)).sqrt();
        let sqrt_p_mu = (semi_latus_rectum / mu).sqrt();
        let w = 1.0 + k * cos_l + h * sin_l;

        Self {
            a,
            h,
            k,
            p,
            q,
            sin_l,
            cos_l,
            w,
            eta,
            ang_mom,
            n,
            sqrt_p_mu,
            mu,
            r,
            v,
        }
    }

    /// `χ = p·cos l − q·sin l` — the fetched source's `(k·cos l − h·sin l)`
    /// cross-track factor, shared by `Ē`, `H̄`, and `M̄`.
    fn chi(&self) -> f64 {
        self.p * self.cos_l - self.q * self.sin_l
    }
}

// ---------------------------------------------------------------------------
// Per-element rate helpers (one phase each; fetched-source coefficient quoted)
// ---------------------------------------------------------------------------

/// `da/dt = Ā·d_r + B̄·d_t` with (source Eq. A.1, renamed `f→k, g→h`):
/// `Ā = (2a²/L)(k·sin l − h·cos l)`, `B̄ = 2a²·W/L`.
fn rate_a(aux: &GveAux, d_r: f64, d_t: f64) -> f64 {
    let two_a2_over_l = 2.0 * aux.a * aux.a / aux.ang_mom;
    let psi = aux.k * aux.sin_l - aux.h * aux.cos_l; // source (f sin l − g cos l)
    two_a2_over_l * (psi * d_r + aux.w * d_t)
}

/// `dh/dt` (the source's `dg/dt`) `= F̄·d_r + Ḡ·d_t + H̄·d_n` with
/// (Eq. A.1, renamed): `F̄ = −√(p/µ)·cos l`,
/// `Ḡ = √(p/µ)·(h + (1+W)·sin l)/W`, `H̄ = −√(p/µ)·k·χ/W`
/// (source `H̄ = √(p/µ)·f·(h·sin l − k·cos l)/W`, and `q·sin l − p·cos l = −χ`;
/// this is the Roth-sign-corrected form).
fn rate_h(aux: &GveAux, d_r: f64, d_t: f64, d_n: f64) -> f64 {
    let s = aux.sqrt_p_mu;
    let f_bar = -s * aux.cos_l;
    let g_bar = s * (aux.h + (1.0 + aux.w) * aux.sin_l) / aux.w;
    let h_bar = -s * aux.k * aux.chi() / aux.w;
    f_bar * d_r + g_bar * d_t + h_bar * d_n
}

/// `dk/dt` (the source's `df/dt`) `= C̄·d_r + D̄·d_t + Ē·d_n` with
/// (Eq. A.1, renamed): `C̄ = √(p/µ)·sin l`,
/// `D̄ = √(p/µ)·(k + (1+W)·cos l)/W`, `Ē = √(p/µ)·h·χ/W`
/// (source `Ē = √(p/µ)·g·(k·cos l − h·sin l)/W`, and `p·cos l − q·sin l = χ`).
fn rate_k(aux: &GveAux, d_r: f64, d_t: f64, d_n: f64) -> f64 {
    let s = aux.sqrt_p_mu;
    let c_bar = s * aux.sin_l;
    let d_bar = s * (aux.k + (1.0 + aux.w) * aux.cos_l) / aux.w;
    let e_bar = s * aux.h * aux.chi() / aux.w;
    c_bar * d_r + d_bar * d_t + e_bar * d_n
}

/// `dp/dt` (the source's `dk/dt`) `= J̄·d_n` with (Eq. A.1, renamed):
/// `J̄ = √(p/µ)·(1 + p² + q²)·sin l / (2W)` (source `(1 + h² + k²)` — the
/// inclination-element magnitude `1 + tan²(i/2) = sec²(i/2)`, never singular
/// for prograde orbits).
fn rate_p(aux: &GveAux, d_n: f64) -> f64 {
    let s2 = 1.0 + aux.p * aux.p + aux.q * aux.q;
    aux.sqrt_p_mu * s2 * aux.sin_l / (2.0 * aux.w) * d_n
}

/// `dq/dt` (the source's `dh/dt`) `= Ī·d_n` with (Eq. A.1, renamed):
/// `Ī = √(p/µ)·(1 + p² + q²)·cos l / (2W)`.
fn rate_q(aux: &GveAux, d_n: f64) -> f64 {
    let s2 = 1.0 + aux.p * aux.p + aux.q * aux.q;
    aux.sqrt_p_mu * s2 * aux.cos_l / (2.0 * aux.w) * d_n
}

/// `dλ/dt = K̄·d_r + L̄·d_t + M̄·d_n + n` with (Eq. A.1, renamed) the
/// two-body secular mean motion `n = √(µ/a³)` and
/// `K̄ = −√(p/µ)·((W−1)/(1+η) + 2η/W)`,
/// `L̄ = −L·(1+W)·(h·cos l − k·sin l)/(µ·W·(1+η))`
/// (source `(g·cos l − f·sin l)`), `M̄ = −L·χ/(µ·W)`
/// (source `−L·(k·cos l − h·sin l)/(µ·W)`).
fn rate_lambda(aux: &GveAux, d_r: f64, d_t: f64, d_n: f64) -> f64 {
    let s = aux.sqrt_p_mu;
    let k_bar = -s * ((aux.w - 1.0) / (1.0 + aux.eta) + 2.0 * aux.eta / aux.w);
    let l_bar = -aux.ang_mom * (1.0 + aux.w) * (aux.h * aux.cos_l - aux.k * aux.sin_l)
        / (aux.mu * aux.w * (1.0 + aux.eta));
    let m_bar = -aux.ang_mom * aux.chi() / (aux.mu * aux.w);
    k_bar * d_r + l_bar * d_t + m_bar * d_n + aux.n
}

// ---------------------------------------------------------------------------
// Element-rate evaluator and RK4 propagation
// ---------------------------------------------------------------------------

/// Evaluate `d[a, h, k, p, q, λ]/dt` at time `t` for the stored equinoctial
/// set under a time-aware inertial perturbing-acceleration closure.
///
/// The closure `perturbation(t, &r, &v)` returns the **perturbing** inertial
/// acceleration (two-body excluded — see [`j2_perturbation_acceleration`]);
/// it is rotated into RSW internally. With a zero closure, only `dλ/dt = n`
/// is nonzero (pure two-body secular motion).
///
/// Domain: valid for bound, prograde orbits (`0 ≤ e < 1`, `0 ≤ i < π`);
/// nonsingular at `e = 0` and `i = 0` (`W → 1`, `η → 1`, `1 + p² + q² → 1`).
pub fn equinoctial_element_rates<F>(
    elements: &[f64; 6],
    mu: f64,
    t: f64,
    perturbation: &F,
) -> [f64; 6]
where
    F: Fn(f64, &Vector3<f64>, &Vector3<f64>) -> Vector3<f64>,
{
    let aux = GveAux::from_elements(elements, mu);
    let a_pert = perturbation(t, &aux.r, &aux.v);
    let basis = RswBasis::from_state(&aux.r, &aux.v);
    let (d_r, d_t, d_n) = basis.project(&a_pert);
    [
        rate_a(&aux, d_r, d_t),
        rate_h(&aux, d_r, d_t, d_n),
        rate_k(&aux, d_r, d_t, d_n),
        rate_p(&aux, d_n),
        rate_q(&aux, d_n),
        rate_lambda(&aux, d_r, d_t, d_n),
    ]
}

/// Number of equal RK4 sub-steps covering `dt`: `ceil(dt / max_step_s)`, at
/// least 1 (so `dt = 0` degenerates to a single identity step). Mirrors the
/// `KeplerJ2` Cartesian model's sub-stepping so the two formulations share a
/// step convention.
fn substep_count(dt: f64, max_step_s: f64) -> usize {
    (dt / max_step_s).ceil().max(1.0) as usize
}

/// `x + a·y` element-wise for the 6-vector of elements (RK4 stage state).
fn axpy(x: &[f64; 6], a: f64, y: &[f64; 6]) -> [f64; 6] {
    let mut out = *x;
    for i in 0..6 {
        out[i] += a * y[i];
    }
    out
}

/// One classical fixed-step RK4 step of the first-order element ODE
/// `dx/dt = f(t, x)` over `dt`, starting at time `t`.
fn rk4_element_step<F>(x: &[f64; 6], mu: f64, t: f64, dt: f64, perturbation: &F) -> [f64; 6]
where
    F: Fn(f64, &Vector3<f64>, &Vector3<f64>) -> Vector3<f64>,
{
    let half = 0.5 * dt;
    let k1 = equinoctial_element_rates(x, mu, t, perturbation);
    let k2 = equinoctial_element_rates(&axpy(x, half, &k1), mu, t + half, perturbation);
    let k3 = equinoctial_element_rates(&axpy(x, half, &k2), mu, t + half, perturbation);
    let k4 = equinoctial_element_rates(&axpy(x, dt, &k3), mu, t + dt, perturbation);
    let mut out = *x;
    for i in 0..6 {
        out[i] += dt / 6.0 * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]);
    }
    out
}

/// Propagate the stored equinoctial elements over `dt` starting at time `t0`
/// with sub-stepped fixed RK4 (`ceil(dt / max_step_s)` equal sub-steps) over
/// the element rates, under the given inertial perturbation closure.
///
/// The mean longitude accumulates continuously (never reduced mod 2π here —
/// design Decision 3's convention; the element→Cartesian conversion owns
/// periodicity). Deterministic: identical inputs yield bitwise-identical
/// outputs.
pub fn propagate_equinoctial<F>(
    elements: [f64; 6],
    mu: f64,
    t0: f64,
    dt: f64,
    max_step_s: f64,
    perturbation: &F,
) -> [f64; 6]
where
    F: Fn(f64, &Vector3<f64>, &Vector3<f64>) -> Vector3<f64>,
{
    let steps = substep_count(dt, max_step_s);
    let sub_dt = dt / steps as f64;
    let mut x = elements;
    let mut t = t0;
    for _ in 0..steps {
        x = rk4_element_step(&x, mu, t, sub_dt, perturbation);
        t += sub_dt;
    }
    x
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::orbital::integrate::rk4_step;
    use std::f64::consts::TAU;

    const EARTH: GravityModel = GravityModel::EARTH_WGS84;
    const EARTH_MU: f64 = GravityModel::EARTH_WGS84.mu;

    /// The J2-only perturbation closure used throughout: pure oblateness,
    /// two-body excluded (design Decision 2). Time-independent for J2.
    fn j2_closure(
        gravity: GravityModel,
    ) -> impl Fn(f64, &Vector3<f64>, &Vector3<f64>) -> Vector3<f64> {
        move |_t, r, _v| j2_perturbation_acceleration(r, &gravity)
    }

    /// Zero perturbation closure (pure two-body).
    fn zero_closure() -> impl Fn(f64, &Vector3<f64>, &Vector3<f64>) -> Vector3<f64> {
        |_t, _r, _v| Vector3::zeros()
    }

    fn equinoctial_state(elements: &[f64; 6], mu: f64) -> OrbitalState {
        OrbitalState {
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
        }
    }

    /// Equinoctial elements `[a, h, k, p, q, λ]` for a Keplerian state.
    fn elements_from_keplerian(
        sma: f64,
        ecc: f64,
        inc: f64,
        raan: f64,
        argp: f64,
        true_anomaly: f64,
        mu: f64,
    ) -> [f64; 6] {
        let state = OrbitalState {
            elements: OrbitalElements::Keplerian {
                sma,
                ecc,
                inc,
                raan,
                argp,
                true_anomaly,
            },
            mu,
            frame: Frame::Teme,
            epoch: Epoch::from_jde_utc(CONVERSION_EPOCH_JD),
        };
        let (a, h, k, p, q, l) = state.as_equinoctial().unwrap();
        [a, h, k, p, q, l]
    }

    fn elements_to_position(elements: &[f64; 6], mu: f64) -> Vector3<f64> {
        equinoctial_state(elements, mu).as_cartesian().unwrap().0
    }

    /// Cartesian two-body+J2 reference propagator: sub-stepped RK4 over the
    /// shared `j2_acceleration` (two-body + J2 composite) closure — the
    /// independent formulation the GVE path is validated against.
    fn propagate_cartesian(
        r0: Vector3<f64>,
        v0: Vector3<f64>,
        gravity: GravityModel,
        dt: f64,
        max_step_s: f64,
    ) -> (Vector3<f64>, Vector3<f64>) {
        let steps = substep_count(dt, max_step_s);
        let sub_dt = dt / steps as f64;
        let accel = |p: &Vector3<f64>, _v: &Vector3<f64>| j2_acceleration(p, &gravity);
        let (mut r, mut v) = (r0, v0);
        for _ in 0..steps {
            let (rr, vv) = rk4_step(&r, &v, sub_dt, accel);
            r = rr;
            v = vv;
        }
        (r, v)
    }

    // ── RSW rotation (spec: "RSW rotation is orthonormal and right-handed") ──

    #[test]
    fn rsw_basis_is_orthonormal_and_right_handed() {
        // A generic inclined, eccentric state: all components nonzero.
        let elements = elements_from_keplerian(9_000_000.0, 0.3, 1.1, 0.7, 0.9, 1.3, EARTH_MU);
        let (r, v) = equinoctial_state(&elements, EARTH_MU)
            .as_cartesian()
            .unwrap();
        let basis = RswBasis::from_state(&r, &v);

        // Unit length.
        for u in [basis.radial, basis.transverse, basis.normal] {
            assert!((u.norm() - 1.0).abs() < 1e-15, "not unit: {}", u.norm());
        }
        // Mutually orthogonal.
        assert!(basis.radial.dot(&basis.transverse).abs() < 1e-15);
        assert!(basis.radial.dot(&basis.normal).abs() < 1e-15);
        assert!(basis.transverse.dot(&basis.normal).abs() < 1e-15);
        // W is parallel to the angular-momentum vector r×v.
        let h_vec = r.cross(&v);
        assert!((basis.normal - h_vec.normalize()).norm() < 1e-15);
        // Right-handed: R̂ × Ŝ = Ŵ.
        assert!((basis.radial.cross(&basis.transverse) - basis.normal).norm() < 1e-15);
        // Transverse has a positive velocity projection (points "forward").
        assert!(basis.transverse.dot(&v) > 0.0);
    }

    // ── Unperturbed motion (spec: "Unperturbed motion moves only λ") ────────

    #[test]
    fn unperturbed_motion_moves_only_mean_longitude() {
        let elements = elements_from_keplerian(7_500_000.0, 0.2, 0.6, 1.0, 0.4, 2.0, EARTH_MU);
        let rates = equinoctial_element_rates(&elements, EARTH_MU, 0.0, &zero_closure());

        // a, h, k, p, q rates are products with a zero perturbation ⇒ exactly 0.
        for (i, label) in ["a", "h", "k", "p", "q"].iter().enumerate() {
            assert_eq!(rates[i], 0.0, "d{label}/dt must be exactly zero");
        }
        // dλ/dt is exactly the two-body mean motion n = √(µ/a³).
        let n = (EARTH_MU / elements[0].powi(3)).sqrt();
        assert!(
            (rates[5] - n).abs() < 1e-15 * n,
            "dλ/dt = {} vs n = {n}",
            rates[5]
        );
    }

    // ── Structural spot values (spec: "Published spot values reproduced") ───
    //
    // The fetched source (arXiv:2105.06516, Appendix A) states the equinoctial
    // GVE symbolically — it carries NO worked numeric example — so, per the
    // spec's explicit allowance, structural identities derived from the fetched
    // equations are verified instead of a pinned numeric value. The chosen
    // identities are the classical Gauss-planetary reductions at zero
    // eccentricity, which the transcribed coefficients must reproduce exactly:
    //   • tangential accel on a circular orbit: da/dt = (2/n)·d_t
    //     (source B̄ = 2a²W/L → 2a²/√(µa) = 2/n at e=0, W=1).
    //   • radial accel on a circular orbit: dλ/dt − n = −(2/(n·a))·d_r
    //     (source K̄ = −√(p/µ)·((W−1)/(1+η) + 2η/W) → −2√(a/µ) = −2/(n·a)).
    #[test]
    fn circular_orbit_reductions_match_classical_gauss() {
        // Circular (e = 0 ⇒ h = k = 0), inclined orbit; λ arbitrary.
        let a = 8_000_000.0;
        let elements = elements_from_keplerian(a, 0.0, 0.7, 0.5, 0.0, 1.7, EARTH_MU);
        let aux = GveAux::from_elements(&elements, EARTH_MU);
        // e = 0 ⇒ W = 1, η = 1 exactly.
        assert!((aux.w - 1.0).abs() < 1e-12, "W = {}", aux.w);
        assert!((aux.eta - 1.0).abs() < 1e-12, "η = {}", aux.eta);

        let n = aux.n;
        // Pure tangential unit acceleration.
        let da_dt = rate_a(&aux, 0.0, 1.0);
        assert!(
            (da_dt - 2.0 / n).abs() < 1e-6 * (2.0 / n),
            "da/dt = {da_dt} vs 2/n = {}",
            2.0 / n
        );
        // Pure radial unit acceleration: dλ/dt − n = K̄ = −2/(n·a).
        let dlam_dt = rate_lambda(&aux, 1.0, 0.0, 0.0);
        let expected = -2.0 / (n * a);
        assert!(
            (dlam_dt - n - expected).abs() < 1e-6 * expected.abs(),
            "K̄ = {} vs −2/(na) = {expected}",
            dlam_dt - n
        );
    }

    // ── Near-singular domain (spec: "Near-circular near-equatorial finite") ─

    #[test]
    fn near_circular_near_equatorial_evaluation_is_finite() {
        // e = 1e-8, i = 1e-8: h,k ~ 1e-8, p,q ~ 5e-9.
        let elements = elements_from_keplerian(7_000_000.0, 1e-8, 1e-8, 0.0, 0.0, 0.9, EARTH_MU);
        assert!(elements[1].hypot(elements[2]) < 1e-7, "e not tiny");
        let rates = equinoctial_element_rates(&elements, EARTH_MU, 0.0, &j2_closure(EARTH));
        assert!(
            rates.iter().all(|r| r.is_finite()),
            "non-finite rate at near-singular state: {rates:?}"
        );
        // And the propagation stays finite over an arc.
        let end = propagate_equinoctial(elements, EARTH_MU, 0.0, 3600.0, 10.0, &j2_closure(EARTH));
        assert!(
            end.iter().all(|x| x.is_finite()),
            "non-finite propagation: {end:?}"
        );
    }

    // ── Deterministic propagation (spec: "Deterministic element propagation") ─

    #[test]
    fn element_propagation_is_bitwise_deterministic() {
        let elements = elements_from_keplerian(7_200_000.0, 0.05, 0.9, 1.2, 0.5, 0.3, EARTH_MU);
        let run =
            || propagate_equinoctial(elements, EARTH_MU, 0.0, 5400.0, 10.0, &j2_closure(EARTH));
        let a = run();
        let b = run();
        for i in 0..6 {
            assert_eq!(
                a[i].to_bits(),
                b[i].to_bits(),
                "element {i} not bit-identical"
            );
        }
    }

    // ── Step-halving convergence (spec: "Convergence under step halving") ───

    #[test]
    fn step_halving_shows_rk4_order() {
        // A J2-perturbed LEO arc (~one orbit). Reference at a very fine step;
        // compare the endpoint position error at step H and H/2.
        let elements = elements_from_keplerian(7_000_000.0, 0.02, 0.9, 1.0, 0.5, 0.0, EARTH_MU);
        let arc = 5000.0; // s (< one LEO period)
        let j2 = j2_closure(EARTH);

        let reference = propagate_equinoctial(elements, EARTH_MU, 0.0, arc, arc / 4096.0, &j2);
        let ref_pos = elements_to_position(&reference, EARTH_MU);

        let coarse_h = 40.0;
        let err = |h: f64| {
            let end = propagate_equinoctial(elements, EARTH_MU, 0.0, arc, h, &j2);
            (elements_to_position(&end, EARTH_MU) - ref_pos).norm()
        };
        let err_coarse = err(coarse_h);
        let err_fine = err(coarse_h / 2.0);
        let ratio = err_coarse / err_fine;

        // RK4 is 4th order ⇒ halving the step shrinks the error ≈ 2⁴ = 16×.
        // MEASURED (this arc/config, cargo test): err(40 s) ≈ 1.43e-3 m,
        // err(20 s) ≈ 8.85e-5 m, ratio ≈ 16.13. Band tolerates the
        // pre-asymptotic and round-off edges.
        assert!(
            (8.0..=32.0).contains(&ratio),
            "step-halving ratio {ratio:.2} (err_H = {err_coarse:.3e} m, err_H/2 = {err_fine:.3e} m) not ~16"
        );
    }

    // ── Cross-formulation equivalence (specs: LEO / eccentric both directions) ─
    //
    // Identical J2 physics through the GVE-in-elements path and the independent
    // Cartesian two-body+J2 RK4 path (they share only the J2/two-body force
    // math, no element-rate code), compared in inertial position at the arc
    // endpoint. "Both directions" is covered per regime: the same physical
    // initial state is authored once as Keplerian → equinoctial (element-born,
    // fed to the GVE path) and once as Keplerian → Cartesian (Cartesian-born,
    // fed to the RK4 path); the element endpoint is converted back to Cartesian
    // for the comparison, exercising the equinoctial→Cartesian direction at the
    // end as well. Tolerances are MEASURED at the chosen steps and documented
    // per test; they are tight enough that any term/sign error in the GVE
    // (which would perturb the J2 secular/periodic signature far more than RK4
    // truncation) fails the assertion.

    /// A cross-formulation equivalence case: a Keplerian initial state
    /// (`sma, ecc, inc, raan, argp, true_anomaly`) plus the arc and RK4 step.
    struct EquivCase {
        kep: [f64; 6],
        arc: f64,
        step: f64,
    }

    /// Endpoint inertial-position gap (m) between the GVE-in-elements path and
    /// the Cartesian two-body+J2 path for a case's initial state.
    fn gve_vs_cartesian_gap(case: &EquivCase) -> f64 {
        let [sma, ecc, inc, raan, argp, true_anomaly] = case.kep;
        let elements = elements_from_keplerian(sma, ecc, inc, raan, argp, true_anomaly, EARTH_MU);
        let (r0, v0) = equinoctial_state(&elements, EARTH_MU)
            .as_cartesian()
            .unwrap();

        let el_end = propagate_equinoctial(
            elements,
            EARTH_MU,
            0.0,
            case.arc,
            case.step,
            &j2_closure(EARTH),
        );
        let r_gve = elements_to_position(&el_end, EARTH_MU);

        let (r_cart, _) = propagate_cartesian(r0, v0, EARTH, case.arc, case.step);
        (r_gve - r_cart).norm()
    }

    #[test]
    fn leo_equivalence_both_directions() {
        // Near-circular LEO, ~3 revolutions (period ≈ 5560 s), 5 s steps.
        let period = TAU * (7_000_000_f64.powi(3) / EARTH_MU).sqrt();
        let gap = gve_vs_cartesian_gap(&EquivCase {
            kep: [7_000_000.0, 0.01, 0.9, 1.2, 0.4, 0.0],
            arc: 3.0 * period,
            step: 5.0,
        });
        // MEASURED (cargo test): gap ≈ 2.75e-3 m over 3 revs at 5 s steps
        // (dominated by the Cartesian RK4 two-body truncation — the GVE
        // two-body part is analytic). Tolerance 0.1 m.
        assert!(gap < 0.1, "LEO GVE vs Cartesian gap = {gap:.4e} m");
    }

    #[test]
    fn meo_equivalence() {
        // MEO (a ≈ 20 000 km, mild eccentricity), ~2 revolutions, 15 s steps.
        let period = TAU * (20_000_000_f64.powi(3) / EARTH_MU).sqrt();
        let gap = gve_vs_cartesian_gap(&EquivCase {
            kep: [20_000_000.0, 0.1, 0.7, 0.5, 1.1, 0.0],
            arc: 2.0 * period,
            step: 15.0,
        });
        // MEASURED (cargo test): gap ≈ 8.2e-4 m. Tolerance 0.1 m.
        assert!(gap < 0.1, "MEO GVE vs Cartesian gap = {gap:.4e} m");
    }

    #[test]
    fn eccentric_equivalence_through_perigee() {
        // Eccentric e = 0.6 (in the documented 0.3–0.7 band), a = 20 000 km ⇒
        // perigee ≈ 8000 km (above the surface), through ~3 perigee passages.
        // 2 s steps to resolve the fast perigee dynamics in both formulations.
        let period = TAU * (20_000_000_f64.powi(3) / EARTH_MU).sqrt();
        let gap = gve_vs_cartesian_gap(&EquivCase {
            kep: [20_000_000.0, 0.6, 0.5, 0.9, 0.3, 0.0],
            arc: 3.0 * period,
            step: 2.0,
        });
        // MEASURED (cargo test): gap ≈ 3.0e-4 m over 3 perigee passages at 2 s
        // steps — the fast-variable stress case; with 2 s steps both
        // formulations resolve perigee tightly, so they agree to sub-mm.
        // Tolerance 0.1 m (a wrong GVE term would drift kilometres — caught
        // with ~300× margin).
        assert!(gap < 0.1, "eccentric GVE vs Cartesian gap = {gap:.4e} m");
    }
}
