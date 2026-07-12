//! Frame-disciplined orbital state representation with on-demand element
//! conversion.
//!
//! [`OrbitalState`] borrows the shape of Stone Soup's MIT-licensed
//! `orbitalstate.py` — an element-representation enum plus a parameterized
//! gravitational constant — without borrowing its code path (Stone Soup's
//! orbital functionality is deprecated upstream), and improves on it with an
//! explicit [`OrbitalFrame`] tag (design Decision 2 of the
//! `orbital-ballistic-filter-models` change). Conversions are pure on-demand
//! methods, not memoized: reading back the constructed representation
//! performs no conversion at all.

use std::f64::consts::TAU;
use std::fmt;

use nalgebra::{DVector, Vector3};
use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// Errors from orbital element conversion and frame-disciplined operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ElementError {
    /// TLE mean elements cannot be converted analytically: they are defined
    /// by SGP4 theory, which lives behind `thresh-data`'s `orbital` feature
    /// (`tle_to_cartesian`) so this crate never grows the `sgp4` dependency.
    RequiresSgp4,
    /// Two states carried different [`OrbitalFrame`] tags where identical
    /// frames are required. Currently only debug-asserted in the arithmetic
    /// helpers; this variant is plumbed so the `astro-time-and-frames`
    /// change can promote the debug-asserts to hard errors without an API
    /// break.
    FrameMismatch {
        /// Frame of the state the operation was called on.
        expected: OrbitalFrame,
        /// Frame of the other state.
        found: OrbitalFrame,
    },
}

impl fmt::Display for ElementError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ElementError::RequiresSgp4 => write!(
                f,
                "TLE mean elements require SGP4 (see thresh-data's `orbital` feature)"
            ),
            ElementError::FrameMismatch { expected, found } => {
                write!(f, "frame mismatch: expected {expected:?}, found {found:?}")
            }
        }
    }
}

impl std::error::Error for ElementError {}

// ---------------------------------------------------------------------------
// Reference frame tag
// ---------------------------------------------------------------------------

/// Reference frame an [`OrbitalState`] is expressed in.
///
/// Under today's GMST-only rotation (`crate::eci`) the two variants are
/// numerically identical *by documented design*; the tag exists so
/// TLE-derived states are honestly labeled [`OrbitalFrame::Teme`] and the
/// later `astro-time-and-frames` change can make the distinction real
/// without an API break.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OrbitalFrame {
    /// Earth-centered inertial under the repo's GMST-only rotation
    /// convention (see `crate::eci`).
    EciGmst,
    /// True Equator, Mean Equinox — the frame of SGP4 output.
    Teme,
}

// ---------------------------------------------------------------------------
// Element representations
// ---------------------------------------------------------------------------

/// Orbital elements in one of four representations.
///
/// Conversion between representations is on demand through the
/// [`OrbitalState`] methods, which supply the gravitational parameter.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum OrbitalElements {
    /// Cartesian position (m) and velocity (m/s).
    Cartesian {
        /// Position (m).
        position: Vector3<f64>,
        /// Velocity (m/s).
        velocity: Vector3<f64>,
    },
    /// Classical Keplerian elements.
    Keplerian {
        /// Semi-major axis (m).
        sma: f64,
        /// Eccentricity (dimensionless).
        ecc: f64,
        /// Inclination (rad).
        inc: f64,
        /// Right ascension of the ascending node (rad).
        raan: f64,
        /// Argument of periapsis (rad).
        argp: f64,
        /// True anomaly (rad).
        true_anomaly: f64,
    },
    /// Equinoctial elements (direct/prograde set), nonsingular at zero
    /// eccentricity and zero inclination:
    /// `h = e·sin(ω+Ω)`, `k = e·cos(ω+Ω)`, `p = tan(i/2)·sin Ω`,
    /// `q = tan(i/2)·cos Ω`, `mean_longitude = M + ω + Ω`.
    Equinoctial {
        /// Semi-major axis (m).
        sma: f64,
        /// `e·sin(ω + Ω)` (dimensionless).
        h: f64,
        /// `e·cos(ω + Ω)` (dimensionless).
        k: f64,
        /// `tan(i/2)·sin Ω` (dimensionless); singular at i = π.
        p: f64,
        /// `tan(i/2)·cos Ω` (dimensionless); singular at i = π.
        q: f64,
        /// Mean longitude `M + ω + Ω` (rad).
        mean_longitude: f64,
    },
    /// Two-line element set (mean elements under SGP4 theory). Carried so
    /// ingest code can move a TLE-born state through frame-tagged plumbing;
    /// analytic conversion is impossible in this crate
    /// ([`ElementError::RequiresSgp4`]).
    Tle {
        /// TLE line 1.
        line1: String,
        /// TLE line 2.
        line2: String,
    },
}

// ---------------------------------------------------------------------------
// Orbital state
// ---------------------------------------------------------------------------

/// A frame-tagged orbital state with a required gravitational parameter.
///
/// `mu` is **required** — there is no Earth default, because a silent Earth
/// default would reintroduce the hardcoded-constant bug this type exists to
/// remove. Callers opt into Earth explicitly via
/// [`GravityModel::EARTH_WGS84`](super::GravityModel::EARTH_WGS84)`.mu`.
///
/// This is the ingest/initialization/analysis type; filters keep operating
/// on bare `DVector<f64>` via [`Self::to_filter_state`] /
/// [`Self::from_filter_state`]. It coexists with
/// `thresh_synth::OrbitalState`, the propagator's lightweight Cartesian
/// sample type (`From`/`TryFrom` conversions live in `thresh-synth`);
/// consolidating the two is owned by the `astro-time-and-frames` change.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OrbitalState {
    /// Element representation.
    pub elements: OrbitalElements,
    /// Gravitational parameter μ = GM (m³/s²) — required, no default.
    pub mu: f64,
    /// Reference frame tag.
    pub frame: OrbitalFrame,
    /// Epoch as Julian Date.
    pub epoch_jd: f64,
}

impl OrbitalState {
    /// Convert to Cartesian `(position, velocity)` in metres and m/s.
    ///
    /// Reading back a `Cartesian`-represented state performs no conversion.
    /// Returns [`ElementError::RequiresSgp4`] for the `Tle` variant.
    pub fn as_cartesian(&self) -> Result<(Vector3<f64>, Vector3<f64>), ElementError> {
        match &self.elements {
            OrbitalElements::Cartesian { position, velocity } => Ok((*position, *velocity)),
            OrbitalElements::Keplerian {
                sma,
                ecc,
                inc,
                raan,
                argp,
                true_anomaly,
            } => Ok(keplerian_to_cartesian(
                *sma,
                *ecc,
                *inc,
                *raan,
                *argp,
                *true_anomaly,
                self.mu,
            )),
            OrbitalElements::Equinoctial {
                sma,
                h,
                k,
                p,
                q,
                mean_longitude,
            } => {
                let (sma, ecc, inc, raan, argp, true_anomaly) =
                    equinoctial_to_keplerian(*sma, *h, *k, *p, *q, *mean_longitude);
                Ok(keplerian_to_cartesian(
                    sma,
                    ecc,
                    inc,
                    raan,
                    argp,
                    true_anomaly,
                    self.mu,
                ))
            }
            OrbitalElements::Tle { .. } => Err(ElementError::RequiresSgp4),
        }
    }

    /// Convert to classical Keplerian elements.
    ///
    /// Returns `(sma, ecc, inc, raan, argp, true_anomaly)` in metres and
    /// radians (matching the tuple order of `OrbitalElements::Keplerian`).
    /// Reading back a `Keplerian`-represented state performs no conversion.
    /// Returns [`ElementError::RequiresSgp4`] for the `Tle` variant.
    pub fn as_keplerian(&self) -> Result<(f64, f64, f64, f64, f64, f64), ElementError> {
        match &self.elements {
            OrbitalElements::Cartesian { position, velocity } => {
                Ok(cartesian_to_keplerian(position, velocity, self.mu))
            }
            OrbitalElements::Keplerian {
                sma,
                ecc,
                inc,
                raan,
                argp,
                true_anomaly,
            } => Ok((*sma, *ecc, *inc, *raan, *argp, *true_anomaly)),
            OrbitalElements::Equinoctial {
                sma,
                h,
                k,
                p,
                q,
                mean_longitude,
            } => Ok(equinoctial_to_keplerian(
                *sma,
                *h,
                *k,
                *p,
                *q,
                *mean_longitude,
            )),
            OrbitalElements::Tle { .. } => Err(ElementError::RequiresSgp4),
        }
    }

    /// Convert to equinoctial elements.
    ///
    /// Returns `(sma, h, k, p, q, mean_longitude)` (matching the field order
    /// of `OrbitalElements::Equinoctial`); singular at i = π (retrograde
    /// equatorial), where `tan(i/2)` diverges. Reading back an
    /// `Equinoctial`-represented state performs no conversion. Returns
    /// [`ElementError::RequiresSgp4`] for the `Tle` variant.
    pub fn as_equinoctial(&self) -> Result<(f64, f64, f64, f64, f64, f64), ElementError> {
        match &self.elements {
            OrbitalElements::Equinoctial {
                sma,
                h,
                k,
                p,
                q,
                mean_longitude,
            } => Ok((*sma, *h, *k, *p, *q, *mean_longitude)),
            OrbitalElements::Tle { .. } => Err(ElementError::RequiresSgp4),
            _ => {
                let (sma, ecc, inc, raan, argp, true_anomaly) = self.as_keplerian()?;
                Ok(keplerian_to_equinoctial(
                    sma,
                    ecc,
                    inc,
                    raan,
                    argp,
                    true_anomaly,
                ))
            }
        }
    }

    /// Re-express this state with `Cartesian` elements, preserving the
    /// frame tag, `mu`, and epoch.
    pub fn to_cartesian_state(&self) -> Result<Self, ElementError> {
        let (position, velocity) = self.as_cartesian()?;
        Ok(Self {
            elements: OrbitalElements::Cartesian { position, velocity },
            mu: self.mu,
            frame: self.frame,
            epoch_jd: self.epoch_jd,
        })
    }

    /// Re-express this state with `Keplerian` elements, preserving the
    /// frame tag, `mu`, and epoch.
    pub fn to_keplerian_state(&self) -> Result<Self, ElementError> {
        let (sma, ecc, inc, raan, argp, true_anomaly) = self.as_keplerian()?;
        Ok(Self {
            elements: OrbitalElements::Keplerian {
                sma,
                ecc,
                inc,
                raan,
                argp,
                true_anomaly,
            },
            mu: self.mu,
            frame: self.frame,
            epoch_jd: self.epoch_jd,
        })
    }

    /// Re-express this state with `Equinoctial` elements, preserving the
    /// frame tag, `mu`, and epoch.
    pub fn to_equinoctial_state(&self) -> Result<Self, ElementError> {
        let (sma, h, k, p, q, mean_longitude) = self.as_equinoctial()?;
        Ok(Self {
            elements: OrbitalElements::Equinoctial {
                sma,
                h,
                k,
                p,
                q,
                mean_longitude,
            },
            mu: self.mu,
            frame: self.frame,
            epoch_jd: self.epoch_jd,
        })
    }

    /// Position difference `self − other` (m), converting both states to
    /// Cartesian.
    ///
    /// Frame discipline: combining states with different [`OrbitalFrame`]
    /// tags is rejected with a debug-assert today; the
    /// [`ElementError::FrameMismatch`] variant is plumbed so the
    /// `astro-time-and-frames` change can promote this to a hard error.
    pub fn position_delta(&self, other: &Self) -> Result<Vector3<f64>, ElementError> {
        debug_assert_eq!(
            self.frame, other.frame,
            "mixed-frame operation: convert explicitly before combining states"
        );
        let (r_self, _) = self.as_cartesian()?;
        let (r_other, _) = other.as_cartesian()?;
        Ok(r_self - r_other)
    }

    /// Euclidean separation between two states (m). Same frame discipline
    /// as [`Self::position_delta`].
    pub fn separation(&self, other: &Self) -> Result<f64, ElementError> {
        Ok(self.position_delta(other)?.norm())
    }

    /// Emit the interleaved 6D filter state `[x, vx, y, vy, z, vz]`
    /// (metres, m/s) consumed by the `thresh-filter` motion models (design
    /// Decision 3 of `orbital-ballistic-filter-models`).
    pub fn to_filter_state(&self) -> Result<DVector<f64>, ElementError> {
        let (position, velocity) = self.as_cartesian()?;
        Ok(DVector::from_row_slice(&[
            position.x, velocity.x, position.y, velocity.y, position.z, velocity.z,
        ]))
    }

    /// Construct from an interleaved 6D filter state `[x, vx, y, vy, z, vz]`
    /// (metres, m/s), the inverse of [`Self::to_filter_state`].
    ///
    /// The result is tagged [`OrbitalFrame::EciGmst`] because filter states
    /// are ECI (GMST convention) by the motion-model frame contract of
    /// design Decision 3.
    ///
    /// # Panics
    ///
    /// Panics if `state` is not 6-dimensional.
    pub fn from_filter_state(state: &DVector<f64>, mu: f64, epoch_jd: f64) -> Self {
        assert_eq!(
            state.len(),
            6,
            "filter state must be the interleaved 6D [x, vx, y, vy, z, vz]"
        );
        Self {
            elements: OrbitalElements::Cartesian {
                position: Vector3::new(state[0], state[2], state[4]),
                velocity: Vector3::new(state[1], state[3], state[5]),
            },
            mu,
            frame: OrbitalFrame::EciGmst,
            epoch_jd,
        }
    }
}

// ---------------------------------------------------------------------------
// Keplerian ↔ Cartesian conversion math
// (moved from `thresh-synth/src/orbital.rs` with GM_EARTH replaced by `mu`)
// ---------------------------------------------------------------------------

/// Convert classical Keplerian elements to Cartesian `(position, velocity)`
/// in metres and m/s.
///
/// Angles in radians; `mu` is the gravitational parameter (m³/s²). This is
/// the former `thresh_synth::OrbitalState::from_keplerian` math (perifocal
/// state rotated by `R3(−Ω)·R1(−i)·R3(−ω)`), parameterized on `mu`.
pub fn keplerian_to_cartesian(
    sma: f64,
    ecc: f64,
    inc: f64,
    raan: f64,
    argp: f64,
    true_anomaly: f64,
    mu: f64,
) -> (Vector3<f64>, Vector3<f64>) {
    let nu = true_anomaly;
    let p = sma * (1.0 - ecc * ecc); // semi-latus rectum
    let r_mag = p / (1.0 + ecc * nu.cos());

    // Position and velocity in the perifocal frame (PQW)
    let r_pqw = [r_mag * nu.cos(), r_mag * nu.sin(), 0.0];
    let v_coeff = (mu / p).sqrt();
    let v_pqw = [v_coeff * (-nu.sin()), v_coeff * (ecc + nu.cos()), 0.0];

    // Rotation from PQW to ECI: R = R3(-Ω) R1(-i) R3(-ω)
    let (so, co) = raan.sin_cos();
    let (si, ci) = inc.sin_cos();
    let (sw, cw) = argp.sin_cos();

    // Rotation matrix columns
    let r11 = co * cw - so * sw * ci;
    let r12 = -(co * sw + so * cw * ci);
    let r21 = so * cw + co * sw * ci;
    let r22 = -(so * sw - co * cw * ci);
    let r31 = sw * si;
    let r32 = cw * si;

    let position = Vector3::new(
        r11 * r_pqw[0] + r12 * r_pqw[1],
        r21 * r_pqw[0] + r22 * r_pqw[1],
        r31 * r_pqw[0] + r32 * r_pqw[1],
    );
    let velocity = Vector3::new(
        r11 * v_pqw[0] + r12 * v_pqw[1],
        r21 * v_pqw[0] + r22 * v_pqw[1],
        r31 * v_pqw[0] + r32 * v_pqw[1],
    );

    (position, velocity)
}

/// Convert a Cartesian state (metres, m/s) to classical Keplerian elements.
///
/// Returns `(sma, ecc, inc, raan, argp, true_anomaly)` with angles in
/// radians. This is the former `thresh_synth::OrbitalState::to_keplerian`
/// math, parameterized on `mu`; degenerate geometries (equatorial and/or
/// circular orbits) return 0 for the undefined angles per the
/// `compute_raan` / `compute_argp` / `compute_true_anomaly` phase helpers.
pub fn cartesian_to_keplerian(
    position: &Vector3<f64>,
    velocity: &Vector3<f64>,
    mu: f64,
) -> (f64, f64, f64, f64, f64, f64) {
    let r = position;
    let v = velocity;
    let r_mag = (r.x * r.x + r.y * r.y + r.z * r.z).sqrt();
    let v2 = v.x * v.x + v.y * v.y + v.z * v.z;

    // Specific angular momentum h = r × v
    let h = Vector3::new(
        r.y * v.z - r.z * v.y,
        r.z * v.x - r.x * v.z,
        r.x * v.y - r.y * v.x,
    );
    let h_mag = (h.x * h.x + h.y * h.y + h.z * h.z).sqrt();

    // Node vector n = ẑ × h
    let n = Vector3::new(-h.y, h.x, 0.0);
    let n_mag = (n.x * n.x + n.y * n.y).sqrt();

    let rdotv = r.x * v.x + r.y * v.y + r.z * v.z;
    let e_vec = eccentricity_vector(r, v, r_mag, v2, rdotv, mu);
    let ecc = (e_vec.x * e_vec.x + e_vec.y * e_vec.y + e_vec.z * e_vec.z).sqrt();

    let sma = 1.0 / (2.0 / r_mag - v2 / mu);
    let inc = (h.z / h_mag).acos();
    let raan = compute_raan(&n, n_mag);
    let argp = compute_argp(&n, n_mag, &e_vec, ecc);
    let true_anomaly = compute_true_anomaly(&e_vec, ecc, r, r_mag, rdotv);

    (sma, ecc, inc, raan, argp, true_anomaly)
}

/// Compute the eccentricity vector from position, velocity, and derived scalars.
fn eccentricity_vector(
    r: &Vector3<f64>,
    v: &Vector3<f64>,
    r_mag: f64,
    v2: f64,
    rdotv: f64,
    mu: f64,
) -> Vector3<f64> {
    Vector3::new(
        (v2 - mu / r_mag) * r.x / mu - rdotv * v.x / mu,
        (v2 - mu / r_mag) * r.y / mu - rdotv * v.y / mu,
        (v2 - mu / r_mag) * r.z / mu - rdotv * v.z / mu,
    )
}

/// Compute right ascension of the ascending node from the node vector.
///
/// Returns 0 for (near-)equatorial orbits, where the node is undefined.
fn compute_raan(n: &Vector3<f64>, n_mag: f64) -> f64 {
    if n_mag <= 1e-12 {
        return 0.0;
    }
    let val = (n.x / n_mag).acos();
    if n.y >= 0.0 { val } else { TAU - val }
}

/// Compute argument of periapsis from node vector and eccentricity vector.
///
/// Returns 0 for (near-)equatorial or (near-)circular orbits, where it is
/// undefined.
fn compute_argp(n: &Vector3<f64>, n_mag: f64, e_vec: &Vector3<f64>, ecc: f64) -> f64 {
    if n_mag <= 1e-12 || ecc <= 1e-12 {
        return 0.0;
    }
    let ndote = (n.x * e_vec.x + n.y * e_vec.y) / (n_mag * ecc);
    let val = ndote.clamp(-1.0, 1.0).acos();
    if e_vec.z >= 0.0 { val } else { TAU - val }
}

/// Compute true anomaly from eccentricity vector and position.
///
/// Returns 0 for (near-)circular orbits, where periapsis is undefined.
fn compute_true_anomaly(
    e_vec: &Vector3<f64>,
    ecc: f64,
    r: &Vector3<f64>,
    r_mag: f64,
    rdotv: f64,
) -> f64 {
    if ecc <= 1e-12 {
        return 0.0;
    }
    let edotr = (e_vec.x * r.x + e_vec.y * r.y + e_vec.z * r.z) / (ecc * r_mag);
    let val = edotr.clamp(-1.0, 1.0).acos();
    if rdotv >= 0.0 { val } else { TAU - val }
}

// ---------------------------------------------------------------------------
// Keplerian ↔ Equinoctial conversion math
// ---------------------------------------------------------------------------

/// Convert classical Keplerian elements to the direct equinoctial set
/// `(sma, h, k, p, q, mean_longitude)`.
///
/// Singular at i = π (retrograde equatorial), where `tan(i/2)` diverges.
fn keplerian_to_equinoctial(
    sma: f64,
    ecc: f64,
    inc: f64,
    raan: f64,
    argp: f64,
    true_anomaly: f64,
) -> (f64, f64, f64, f64, f64, f64) {
    let lon_periapsis = argp + raan; // ω + Ω, longitude of periapsis
    let h = ecc * lon_periapsis.sin();
    let k = ecc * lon_periapsis.cos();
    let tan_half_inc = (0.5 * inc).tan();
    let p = tan_half_inc * raan.sin();
    let q = tan_half_inc * raan.cos();
    let mean_longitude = (true_to_mean_anomaly(true_anomaly, ecc) + lon_periapsis).rem_euclid(TAU);
    (sma, h, k, p, q, mean_longitude)
}

/// Convert the direct equinoctial set back to classical Keplerian elements
/// `(sma, ecc, inc, raan, argp, true_anomaly)`.
///
/// Degenerate geometries follow the [`cartesian_to_keplerian`] conventions:
/// `atan2(0, 0) = 0`, so circular orbits get `argp = 0` and equatorial
/// orbits get `raan = 0`.
fn equinoctial_to_keplerian(
    sma: f64,
    h: f64,
    k: f64,
    p: f64,
    q: f64,
    mean_longitude: f64,
) -> (f64, f64, f64, f64, f64, f64) {
    let ecc = (h * h + k * k).sqrt();
    let inc = 2.0 * (p * p + q * q).sqrt().atan();
    let raan = p.atan2(q).rem_euclid(TAU);
    let lon_periapsis = h.atan2(k); // ω + Ω
    let argp = (lon_periapsis - raan).rem_euclid(TAU);
    let mean_anomaly = (mean_longitude - lon_periapsis).rem_euclid(TAU);
    let true_anomaly = mean_to_true_anomaly(mean_anomaly, ecc);
    (sma, ecc, inc, raan, argp, true_anomaly)
}

/// Convert true anomaly to mean anomaly for an elliptical orbit (e < 1),
/// via the eccentric anomaly and Kepler's equation `M = E − e·sin E`.
/// Result normalized to [0, 2π).
fn true_to_mean_anomaly(true_anomaly: f64, ecc: f64) -> f64 {
    // E from ν: tan E = √(1−e²)·sin ν / (e + cos ν), quadrant-safe.
    let ecc_anomaly =
        ((1.0 - ecc * ecc).sqrt() * true_anomaly.sin()).atan2(ecc + true_anomaly.cos());
    (ecc_anomaly - ecc * ecc_anomaly.sin()).rem_euclid(TAU)
}

/// Convert mean anomaly to true anomaly for an elliptical orbit (e < 1) by
/// solving Kepler's equation. Result normalized to [0, 2π).
fn mean_to_true_anomaly(mean_anomaly: f64, ecc: f64) -> f64 {
    let ecc_anomaly = solve_kepler(mean_anomaly.rem_euclid(TAU), ecc);
    // ν from E: tan ν = √(1−e²)·sin E / (cos E − e), quadrant-safe.
    ((1.0 - ecc * ecc).sqrt() * ecc_anomaly.sin())
        .atan2(ecc_anomaly.cos() - ecc)
        .rem_euclid(TAU)
}

/// Solve Kepler's equation `M = E − e·sin E` for the eccentric anomaly `E`
/// by Newton iteration.
///
/// Converges quadratically for e < 1; the π start for high eccentricity and
/// the iteration cap guard the flat region near periapsis at extreme e.
fn solve_kepler(mean_anomaly: f64, ecc: f64) -> f64 {
    const MAX_ITERATIONS: usize = 50;
    const TOLERANCE: f64 = 1e-14;

    let mut ecc_anomaly = if ecc < 0.8 {
        mean_anomaly
    } else {
        std::f64::consts::PI
    };
    for _ in 0..MAX_ITERATIONS {
        let residual = ecc_anomaly - ecc * ecc_anomaly.sin() - mean_anomaly;
        if residual.abs() < TOLERANCE {
            break;
        }
        ecc_anomaly -= residual / (1.0 - ecc * ecc_anomaly.cos());
    }
    ecc_anomaly
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::orbital::GravityModel;

    const EARTH_MU: f64 = GravityModel::EARTH_WGS84.mu;
    /// Lunar GM (m³/s²), same value as the gravity-module tests.
    const LUNAR_MU: f64 = 4.904_869_5e12;

    /// A non-degenerate elliptical LEO used across the round-trip tests —
    /// the same elements as the pre-existing synth `keplerian_roundtrip`.
    fn leo_keplerian_state() -> OrbitalState {
        OrbitalState {
            elements: OrbitalElements::Keplerian {
                sma: 7_000_000.0,
                ecc: 0.01,
                inc: 0.9,
                raan: 1.2,
                argp: 0.5,
                true_anomaly: 0.8,
            },
            mu: EARTH_MU,
            frame: OrbitalFrame::EciGmst,
            epoch_jd: 2_451_545.0,
        }
    }

    // ── Keplerian round trip (spec: "Keplerian round trip") ─────────────

    #[test]
    fn keplerian_roundtrip() {
        // Tightness matches the pre-existing synth `keplerian_roundtrip`.
        let state = leo_keplerian_state();
        let (position, velocity) = state.as_cartesian().unwrap();
        let cart = OrbitalState {
            elements: OrbitalElements::Cartesian { position, velocity },
            ..state
        };
        let (sma, ecc, inc, raan, argp, ta) = cart.as_keplerian().unwrap();

        assert!((sma - 7_000_000.0).abs() < 1.0, "SMA: {sma:.1}");
        assert!((ecc - 0.01).abs() < 1e-10, "ecc: {ecc}");
        assert!((inc - 0.9).abs() < 1e-10, "inc: {inc}");
        assert!((raan - 1.2).abs() < 1e-10, "raan: {raan}");
        assert!((argp - 0.5).abs() < 1e-8, "argp: {argp}");
        assert!((ta - 0.8).abs() < 1e-8, "ta: {ta}");
    }

    // ── Conversion is deferred (spec: "Conversion is deferred…") ────────

    #[test]
    fn reading_back_constructed_representation_is_bitwise_identity() {
        // Conversions are pure on-demand methods (no eager or memoized
        // conversion at construction), so reading back the constructed
        // representation must return the stored values bitwise — a value
        // that had been pushed through any trigonometric round trip would
        // not survive this.
        let position = Vector3::new(6_524_834.0, 6_862_875.0, 6_448_296.0);
        let velocity = Vector3::new(4_901.327, 5_533.756, -1_976.341);
        let cart = OrbitalState {
            elements: OrbitalElements::Cartesian { position, velocity },
            mu: EARTH_MU,
            frame: OrbitalFrame::EciGmst,
            epoch_jd: 2_451_545.0,
        };
        let (r, v) = cart.as_cartesian().unwrap();
        for i in 0..3 {
            assert_eq!(r[i].to_bits(), position[i].to_bits());
            assert_eq!(v[i].to_bits(), velocity[i].to_bits());
        }

        let kep = leo_keplerian_state();
        let (sma, ecc, inc, raan, argp, ta) = kep.as_keplerian().unwrap();
        assert_eq!(sma.to_bits(), 7_000_000.0_f64.to_bits());
        assert_eq!(ecc.to_bits(), 0.01_f64.to_bits());
        assert_eq!(inc.to_bits(), 0.9_f64.to_bits());
        assert_eq!(raan.to_bits(), 1.2_f64.to_bits());
        assert_eq!(argp.to_bits(), 0.5_f64.to_bits());
        assert_eq!(ta.to_bits(), 0.8_f64.to_bits());
    }

    // ── Equinoctial conversions ──────────────────────────────────────────

    #[test]
    fn equinoctial_roundtrip() {
        let state = leo_keplerian_state();
        let equi = state.to_equinoctial_state().unwrap();
        let (sma, ecc, inc, raan, argp, ta) = equi.as_keplerian().unwrap();

        assert!((sma - 7_000_000.0).abs() < 1e-6, "SMA: {sma:.9}");
        assert!((ecc - 0.01).abs() < 1e-12, "ecc: {ecc}");
        assert!((inc - 0.9).abs() < 1e-12, "inc: {inc}");
        assert!((raan - 1.2).abs() < 1e-12, "raan: {raan}");
        assert!((argp - 0.5).abs() < 1e-10, "argp: {argp}");
        assert!((ta - 0.8).abs() < 1e-10, "ta: {ta}");
    }

    #[test]
    fn equinoctial_to_cartesian_matches_keplerian_path() {
        let kep = leo_keplerian_state();
        let equi = kep.to_equinoctial_state().unwrap();

        let (r_kep, v_kep) = kep.as_cartesian().unwrap();
        let (r_equi, v_equi) = equi.as_cartesian().unwrap();

        assert!((r_kep - r_equi).norm() < 1e-4, "position via equinoctial");
        assert!((v_kep - v_equi).norm() < 1e-7, "velocity via equinoctial");
    }

    // ── Anomaly phase helpers ────────────────────────────────────────────

    #[test]
    fn true_mean_anomaly_roundtrip() {
        for &ecc in &[0.0, 0.01, 0.4, 0.832_85] {
            for i in 0..12 {
                let nu = f64::from(i) * TAU / 12.0;
                let recovered = mean_to_true_anomaly(true_to_mean_anomaly(nu, ecc), ecc);
                let diff = (recovered - nu)
                    .rem_euclid(TAU)
                    .min((nu - recovered).rem_euclid(TAU));
                assert!(diff < 1e-12, "e={ecc} nu={nu}: recovered {recovered}");
            }
        }
    }

    #[test]
    fn solve_kepler_satisfies_keplers_equation_at_high_ecc() {
        let (mean_anomaly, ecc) = (0.3, 0.95);
        let e_anom = solve_kepler(mean_anomaly, ecc);
        assert!((e_anom - ecc * e_anom.sin() - mean_anomaly).abs() < 1e-13);
    }

    #[test]
    fn circular_orbit_mean_anomaly_equals_true_anomaly() {
        assert!((true_to_mean_anomaly(1.234, 0.0) - 1.234).abs() < 1e-15);
        assert!((mean_to_true_anomaly(1.234, 0.0) - 1.234).abs() < 1e-15);
    }

    // ── Frame discipline (spec: "Mixed-frame operation rejected") ───────

    #[test]
    #[should_panic(expected = "mixed-frame operation")]
    fn mixed_frame_position_delta_is_rejected() {
        let eci = leo_keplerian_state();
        let teme = OrbitalState {
            frame: OrbitalFrame::Teme,
            ..leo_keplerian_state()
        };
        let _ = eci.position_delta(&teme);
    }

    #[test]
    fn same_frame_arithmetic_works() {
        let a = leo_keplerian_state();
        let b = a.to_cartesian_state().unwrap();
        // Same physical state through two representations: zero separation
        // up to conversion round-off.
        assert!(a.separation(&b).unwrap() < 1e-4);
    }

    // ── Frame tag survival (spec: "Frame tag survives…") ────────────────

    #[test]
    fn frame_tag_survives_representation_conversion() {
        let teme = OrbitalState {
            frame: OrbitalFrame::Teme,
            ..leo_keplerian_state()
        };
        assert_eq!(teme.to_cartesian_state().unwrap().frame, OrbitalFrame::Teme);
        assert_eq!(teme.to_keplerian_state().unwrap().frame, OrbitalFrame::Teme);
        assert_eq!(
            teme.to_equinoctial_state().unwrap().frame,
            OrbitalFrame::Teme
        );
    }

    // ── Parameterized mu (spec: "Conversions honor the supplied mu") ────

    #[test]
    fn conversions_honor_supplied_mu() {
        // Same Cartesian elements, Earth vs lunar mu: the derived
        // semi-major axes must each satisfy the vis-viva relation
        // a = 1 / (2/r − v²/μ) for their own mu.
        let position = Vector3::new(7_000_000.0, 0.0, 0.0);
        let velocity = Vector3::new(0.0, 1_000.0, 0.0);
        let r = position.norm();
        let v2 = velocity.norm_squared();

        let mut smas = Vec::new();
        for mu in [EARTH_MU, LUNAR_MU] {
            let state = OrbitalState {
                elements: OrbitalElements::Cartesian { position, velocity },
                mu,
                frame: OrbitalFrame::EciGmst,
                epoch_jd: 2_451_545.0,
            };
            let (sma, ..) = state.as_keplerian().unwrap();
            let vis_viva = 1.0 / (2.0 / r - v2 / mu);
            assert!(
                (sma - vis_viva).abs() / vis_viva.abs() < 1e-12,
                "mu={mu:e}: sma {sma} vs vis-viva {vis_viva}"
            );
            smas.push(sma);
        }
        assert!(
            (smas[0] - smas[1]).abs() > 1.0e6,
            "Earth vs lunar mu must give different sma: {smas:?}"
        );
    }

    // ── Golden tier 1 (task 2.5, design Decision 8) ──────────────────────
    //
    // Attribution: element-conversion vectors from Vallado, "Fundamentals of
    // Astrodynamics and Applications", 4th ed., Example 2-5 (rv2coe) and
    // Example 2-6 (coe2rv) — the canonical worked examples also exercised by
    // Stone Soup's MIT-licensed orbital-state conversion tests, ported here
    // as inline Rust constants.
    //
    // Tolerances, recorded alongside the vectors:
    //   - Against the book's *printed* values: angles are printed to
    //     0.01–0.001°, so the assertion tolerance is 2e-4 rad; sma to
    //     1e-6 relative; ecc to 1e-6. Example 2-6's inputs are themselves
    //     printed rounded, so its r/v outputs carry ~25 m / ~0.02 m/s of
    //     input-rounding error (asserted at 30 m / 0.02 m/s).
    //   - Self-consistency through our own conversion math (round trip):
    //     1e-6 relative on sma, 1e-9 rad on angles.

    /// Vallado 4th ed. Example 2-5 input: r (m), ECI.
    const VALLADO_2_5_POSITION_M: [f64; 3] = [6_524.834e3, 6_862.875e3, 6_448.296e3];
    /// Vallado 4th ed. Example 2-5 input: v (m/s), ECI.
    const VALLADO_2_5_VELOCITY_M_S: [f64; 3] = [4.901_327e3, 5.533_756e3, -1.976_341e3];
    /// Example 2-5 printed elements: a (m), e, and angles (deg).
    const VALLADO_2_5_SMA_M: f64 = 36_127.343e3;
    const VALLADO_2_5_ECC: f64 = 0.832_853;
    const VALLADO_2_5_INC_DEG: f64 = 87.870;
    const VALLADO_2_5_RAAN_DEG: f64 = 227.898;
    const VALLADO_2_5_ARGP_DEG: f64 = 53.38;
    const VALLADO_2_5_TRUE_ANOMALY_DEG: f64 = 92.335;

    /// Vallado 4th ed. Example 2-6 input: p = 11067.790 km, e = 0.83285,
    /// i = 87.87°, Ω = 227.89°, ω = 53.38°, ν = 92.335° (as printed).
    const VALLADO_2_6_SEMI_LATUS_M: f64 = 11_067.790e3;
    const VALLADO_2_6_ECC: f64 = 0.832_85;
    const VALLADO_2_6_INC_DEG: f64 = 87.87;
    const VALLADO_2_6_RAAN_DEG: f64 = 227.89;
    const VALLADO_2_6_ARGP_DEG: f64 = 53.38;
    const VALLADO_2_6_TRUE_ANOMALY_DEG: f64 = 92.335;
    /// Example 2-6 printed output: r (m) and v (m/s).
    const VALLADO_2_6_POSITION_M: [f64; 3] = [6_525.344e3, 6_861.535e3, 6_449.125e3];
    const VALLADO_2_6_VELOCITY_M_S: [f64; 3] = [4.902_276e3, 5.533_124e3, -1.975_709e3];

    /// Printed-precision tolerance for angles (2–3 decimal degrees).
    const ANGLE_PRINT_TOL_RAD: f64 = 2e-4;

    fn vallado_2_5_cartesian_state() -> OrbitalState {
        OrbitalState {
            elements: OrbitalElements::Cartesian {
                position: Vector3::from_row_slice(&VALLADO_2_5_POSITION_M),
                velocity: Vector3::from_row_slice(&VALLADO_2_5_VELOCITY_M_S),
            },
            mu: EARTH_MU,
            frame: OrbitalFrame::EciGmst,
            epoch_jd: 2_451_545.0,
        }
    }

    #[test]
    fn vallado_example_2_5_rv2coe() {
        let (sma, ecc, inc, raan, argp, ta) = vallado_2_5_cartesian_state().as_keplerian().unwrap();

        assert!(
            (sma - VALLADO_2_5_SMA_M).abs() / VALLADO_2_5_SMA_M < 1e-6,
            "sma {sma:.1} vs printed {VALLADO_2_5_SMA_M:.1}"
        );
        assert!((ecc - VALLADO_2_5_ECC).abs() < 1e-6, "ecc {ecc}");
        for (label, got, printed_deg) in [
            ("inc", inc, VALLADO_2_5_INC_DEG),
            ("raan", raan, VALLADO_2_5_RAAN_DEG),
            ("argp", argp, VALLADO_2_5_ARGP_DEG),
            ("true anomaly", ta, VALLADO_2_5_TRUE_ANOMALY_DEG),
        ] {
            assert!(
                (got - printed_deg.to_radians()).abs() < ANGLE_PRINT_TOL_RAD,
                "{label}: {:.6}° vs printed {printed_deg}°",
                got.to_degrees()
            );
        }
    }

    #[test]
    fn vallado_example_2_6_coe2rv() {
        let ecc = VALLADO_2_6_ECC;
        let sma = VALLADO_2_6_SEMI_LATUS_M / (1.0 - ecc * ecc);
        let state = OrbitalState {
            elements: OrbitalElements::Keplerian {
                sma,
                ecc,
                inc: VALLADO_2_6_INC_DEG.to_radians(),
                raan: VALLADO_2_6_RAAN_DEG.to_radians(),
                argp: VALLADO_2_6_ARGP_DEG.to_radians(),
                true_anomaly: VALLADO_2_6_TRUE_ANOMALY_DEG.to_radians(),
            },
            mu: EARTH_MU,
            frame: OrbitalFrame::EciGmst,
            epoch_jd: 2_451_545.0,
        };
        let (r, v) = state.as_cartesian().unwrap();

        // 30 m / 0.02 m/s: dominated by the rounding of the book's printed
        // input elements, not by our conversion (see tolerance note above).
        let r_expected = Vector3::from_row_slice(&VALLADO_2_6_POSITION_M);
        let v_expected = Vector3::from_row_slice(&VALLADO_2_6_VELOCITY_M_S);
        assert!(
            (r - r_expected).norm() < 30.0,
            "position error {:.3} m",
            (r - r_expected).norm()
        );
        assert!(
            (v - v_expected).norm() < 0.02,
            "velocity error {:.6} m/s",
            (v - v_expected).norm()
        );
    }

    #[test]
    fn vallado_vectors_roundtrip_at_recorded_tolerances() {
        // Self-consistency at the task-2.5 tolerances: 1e-6 relative on
        // sma, 1e-9 rad on angles.
        let state = vallado_2_5_cartesian_state();
        let first = state.as_keplerian().unwrap();
        let second = state
            .to_keplerian_state()
            .unwrap()
            .to_cartesian_state()
            .unwrap()
            .as_keplerian()
            .unwrap();

        assert!((second.0 - first.0).abs() / first.0 < 1e-6, "sma");
        assert!((second.1 - first.1).abs() < 1e-9, "ecc");
        assert!((second.2 - first.2).abs() < 1e-9, "inc");
        assert!((second.3 - first.3).abs() < 1e-9, "raan");
        assert!((second.4 - first.4).abs() < 1e-9, "argp");
        assert!((second.5 - first.5).abs() < 1e-9, "true anomaly");
    }

    // ── Filter-state mapping (task 2.6) ──────────────────────────────────

    #[test]
    fn filter_state_is_interleaved() {
        let state = OrbitalState {
            elements: OrbitalElements::Cartesian {
                position: Vector3::new(1.0, 2.0, 3.0),
                velocity: Vector3::new(4.0, 5.0, 6.0),
            },
            mu: EARTH_MU,
            frame: OrbitalFrame::EciGmst,
            epoch_jd: 2_451_545.0,
        };
        let x = state.to_filter_state().unwrap();
        assert_eq!(x.as_slice(), &[1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }

    #[test]
    fn filter_state_roundtrip() {
        let state = leo_keplerian_state();
        let x = state.to_filter_state().unwrap();
        let back = OrbitalState::from_filter_state(&x, state.mu, state.epoch_jd);

        assert_eq!(back.frame, OrbitalFrame::EciGmst);
        assert_eq!(back.mu, state.mu);
        let (r0, v0) = state.as_cartesian().unwrap();
        let (r1, v1) = back.as_cartesian().unwrap();
        for i in 0..3 {
            assert_eq!(r0[i].to_bits(), r1[i].to_bits());
            assert_eq!(v0[i].to_bits(), v1[i].to_bits());
        }
    }

    #[test]
    #[should_panic(expected = "interleaved 6D")]
    fn from_filter_state_rejects_wrong_dimension() {
        let x = DVector::from_row_slice(&[1.0, 2.0, 3.0]);
        let _ = OrbitalState::from_filter_state(&x, EARTH_MU, 2_451_545.0);
    }

    // ── TLE variant (task 2.7, core side) ────────────────────────────────

    #[test]
    fn tle_variant_requires_sgp4() {
        let state = OrbitalState {
            elements: OrbitalElements::Tle {
                line1: "1 25544U 98067A   24001.00000000  .00016717  00000-0  10270-3 0  9026"
                    .to_string(),
                line2: "2 25544  51.6400 208.9163 0006703  30.1579 330.0018 15.49560455    18"
                    .to_string(),
            },
            mu: EARTH_MU,
            frame: OrbitalFrame::Teme,
            epoch_jd: 2_460_310.5,
        };
        assert_eq!(state.as_cartesian(), Err(ElementError::RequiresSgp4));
        assert_eq!(state.as_keplerian(), Err(ElementError::RequiresSgp4));
        assert_eq!(state.as_equinoctial(), Err(ElementError::RequiresSgp4));
        assert_eq!(state.to_filter_state(), Err(ElementError::RequiresSgp4));
    }
}
