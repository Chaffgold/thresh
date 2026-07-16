//! Reference-frame transforms via the IAU-76/FK5 reduction.
//!
//! The chain (Vallado's classical reduction) is
//!
//! ```text
//! GCRF --P(zeta,theta,z)--> MOD --N(dpsi,deps)--> TOD --R3(GAST)--> PEF --W(xp,yp)--> ITRF
//!                                                  |
//!                            TEME = R3(Eqe) * TOD  +-- TEME --R3(GMST)--> PEF
//! ```
//!
//! with precession **P** (IAU-76), nutation **N** (full 106-term IAU-1980
//! series), sidereal rotation by GAST = GMST(IAU-1982) + the IAU-1994
//! equation of the equinoxes (Eqe), and optional polar motion **W**. TEME —
//! the frame SGP4 emits — joins at the true-equator/mean-equinox point:
//! `r_tod = R3(−Eqe) · r_teme` (Vallado, AIAA 2006-6753 Appendix C, Eq. C-1)
//! and TEME → PEF is the pure `R3(GMST)` spin, so both junctions are
//! mutually consistent by construction.
//!
//! Rotations come from a [`FrameProvider`] — the seam that lets a later
//! (e.g. CIO-based or ANISE-backed) implementation slot in without touching
//! call sites. [`Iau76Fk5Provider`] is the default. Free functions
//! ([`teme_to_gcrf`], [`gcrf_to_itrf`], …) wrap the default provider for the
//! common path.
//!
//! # Conventions
//!
//! * Rotation matrices are frame rotations acting on column vectors:
//!   `v_to = R * v_from`. Positions are metres, velocities m/s, angles
//!   radians.
//! * Epochs are [`Epoch`] values; precession/nutation legs evaluate at the
//!   epoch's **TT**, sidereal legs at **UT1 = UTC + ΔUT1** (zero ΔUT1
//!   default ⇒ UT1 ≈ UTC; see [`Iau76Fk5Provider`] for the accuracy floor).
//! * GCRF is treated as the IAU-76/FK5 J2000 mean equator/equinox frame:
//!   the ~23 mas frame bias between them is below this reduction's accuracy
//!   floor and is not modeled.
//!
//! # Documented approximations
//!
//! * Inertial ↔ inertial rotations (GCRF/MOD/TOD/TEME) carry
//!   [`FrameRotation::r_dot`] = 0: precession/nutation rates are
//!   ~10⁻¹² rad/s, below anything the velocity or covariance transforms can
//!   observe at tracking accuracy.
//! * Inertial ↔ PEF/ITRF rotations carry the Earth-rotation rate term
//!   (ω⊕ ≈ 7.292115e-5 rad/s, [`crate::eci::EARTH_ROTATION_RATE`]) in
//!   `r_dot`; the far smaller GAST *acceleration* and polar-motion rates are
//!   neglected.
//! * Polar motion uses the FK5 convention s′ = 0 (see
//!   [`legs::polar_motion_matrix`]).
//!
//! # Example
//!
//! ```
//! use nalgebra::Vector3;
//! use thresh_core::frames::teme_to_gcrf;
//! use thresh_core::time::Epoch;
//!
//! let epoch = Epoch::from_gregorian_utc(2024, 1, 1, 0, 0, 0, 0);
//! let r_teme = Vector3::new(7_000_000.0, 0.0, 0.0); // metres, TEME
//! let v_teme = Vector3::new(0.0, 7_500.0, 0.0); // m/s, TEME
//! let (r_gcrf, _v_gcrf) = teme_to_gcrf(&r_teme, &v_teme, &epoch);
//! // TEME and GCRF differ by accumulated precession/nutation:
//! // kilometre-scale at LEO radius for a modern epoch.
//! assert!((r_gcrf - r_teme).norm() > 1_000.0);
//! ```

pub mod legs;
mod nutation1980;

use std::f64::consts::TAU;
use std::fmt;

use nalgebra::{Matrix3, Matrix6, Vector3};
use serde::{Deserialize, Serialize};

use crate::eci::{EARTH_ROTATION_RATE, rotation_z};
use crate::time::Epoch;

/// A reference frame of the IAU-76/FK5 reduction chain.
///
/// All variants share the geocentre as origin; they differ in axis
/// orientation (and, for [`Frame::Pef`]/[`Frame::Itrf`], in co-rotating with
/// the Earth).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "UPPERCASE")]
pub enum Frame {
    /// Geocentric Celestial Reference Frame — the inertial anchor of the
    /// chain, treated as the J2000 mean equator/mean equinox frame (the
    /// ~23 mas frame bias is not modeled by IAU-76/FK5).
    Gcrf,
    /// Mean equator and mean equinox of date (GCRF + IAU-76 precession).
    Mod,
    /// True equator and true equinox of date (MOD + IAU-1980 nutation).
    Tod,
    /// True equator, mean equinox — the frame SGP4/TLE ephemerides are
    /// expressed in; differs from TOD by `R3` of the equation of the
    /// equinoxes.
    Teme,
    /// Pseudo-Earth-Fixed: TOD spun by GAST — Earth-fixed except for polar
    /// motion.
    Pef,
    /// International Terrestrial Reference Frame: PEF corrected for polar
    /// motion (identical to PEF under the provider's zero-EOP default).
    Itrf,
}

impl Frame {
    /// Every frame in chain order, for iteration in tests and tooling.
    pub const ALL: [Frame; 6] = [
        Frame::Gcrf,
        Frame::Mod,
        Frame::Tod,
        Frame::Teme,
        Frame::Pef,
        Frame::Itrf,
    ];

    /// True for the Earth-co-rotating frames ([`Frame::Pef`],
    /// [`Frame::Itrf`]), whose rotations from inertial frames carry a
    /// non-zero [`FrameRotation::r_dot`].
    pub fn is_earth_fixed(self) -> bool {
        matches!(self, Frame::Pef | Frame::Itrf)
    }
}

impl fmt::Display for Frame {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = match self {
            Frame::Gcrf => "GCRF",
            Frame::Mod => "MOD",
            Frame::Tod => "TOD",
            Frame::Teme => "TEME",
            Frame::Pef => "PEF",
            Frame::Itrf => "ITRF",
        };
        f.write_str(name)
    }
}

/// A frame-to-frame rotation and its time derivative, evaluated at one
/// epoch.
///
/// Maps column vectors **from** the source frame **to** the target frame:
/// `r_to = r * r_from`, `v_to = r * v_from + r_dot * r_from` (metres, m/s).
/// For inertial → Earth-fixed pairs `r_dot` carries the ω⊕ Earth-rotation
/// term; for inertial ↔ inertial pairs it is zero (documented approximation,
/// see the module docs).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FrameRotation {
    /// Rotation matrix (source frame → target frame).
    pub r: Matrix3<f64>,
    /// Time derivative of `r` (rad/s scale), used by the velocity and
    /// covariance transforms.
    pub r_dot: Matrix3<f64>,
}

impl FrameRotation {
    /// The inverse rotation (target frame → source frame): transposes both
    /// `r` and `r_dot` (valid because `r` is orthonormal).
    pub fn inverse(&self) -> FrameRotation {
        FrameRotation {
            r: self.r.transpose(),
            r_dot: self.r_dot.transpose(),
        }
    }

    /// Rotates a position (metres) from the source frame to the target
    /// frame.
    pub fn rotate_position(&self, position: &Vector3<f64>) -> Vector3<f64> {
        self.r * position
    }

    /// Rotates a position/velocity state (metres, m/s) from the source
    /// frame to the target frame: `v_to = r·v + r_dot·p` — the `r_dot` term
    /// is what subtracts (or restores) the ω⊕ × r co-rotation velocity on
    /// inertial ↔ Earth-fixed legs.
    pub fn rotate_state(
        &self,
        position: &Vector3<f64>,
        velocity: &Vector3<f64>,
    ) -> (Vector3<f64>, Vector3<f64>) {
        (self.r * position, self.r * velocity + self.r_dot * position)
    }

    /// The 6×6 state-transform Jacobian `[[R, 0], [Ṙ, R]]` (position block
    /// first, velocity block second), for covariance propagation.
    pub fn jacobian(&self) -> Matrix6<f64> {
        let mut j = Matrix6::zeros();
        j.fixed_view_mut::<3, 3>(0, 0).copy_from(&self.r);
        j.fixed_view_mut::<3, 3>(3, 0).copy_from(&self.r_dot);
        j.fixed_view_mut::<3, 3>(3, 3).copy_from(&self.r);
        j
    }

    /// Maps a 6×6 position/velocity covariance (m², m²/s, m²/s² blocks)
    /// through [`FrameRotation::jacobian`]: `P_to = J · P_from · Jᵀ`,
    /// re-symmetrized against floating-point drift.
    ///
    /// On inertial ↔ ITRF legs the `Ṙ` block couples position uncertainty
    /// into the velocity block; on inertial ↔ inertial legs `Ṙ = 0` and this
    /// reduces to the block rotation `R·P·Rᵀ` per block.
    pub fn rotate_covariance(&self, covariance: &Matrix6<f64>) -> Matrix6<f64> {
        let j = self.jacobian();
        let p = j * covariance * j.transpose();
        (p + p.transpose()) * 0.5
    }

    /// Maps a 3×3 position-only covariance (m²) as `R·P·Rᵀ`,
    /// re-symmetrized — the measurement-space form (e.g. for the ENU chain).
    pub fn rotate_position_covariance(&self, covariance: &Matrix3<f64>) -> Matrix3<f64> {
        let p = self.r * covariance * self.r.transpose();
        (p + p.transpose()) * 0.5
    }
}

/// Error from a [`FrameProvider`].
///
/// Serde derives exist so error-carrying types that embed this (e.g.
/// `thresh_core::orbital::ElementError`) keep their serialized form.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum FrameError {
    /// The provider cannot produce a rotation between these two frames
    /// (never returned by [`Iau76Fk5Provider`], which supports every pair).
    UnsupportedPair {
        /// Requested source frame.
        from: Frame,
        /// Requested target frame.
        to: Frame,
    },
}

impl fmt::Display for FrameError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            FrameError::UnsupportedPair { from, to } => {
                write!(f, "provider has no rotation from {from} to {to}")
            }
        }
    }
}

impl std::error::Error for FrameError {}

/// Source of frame rotations — the seam between transform call sites and a
/// reduction theory.
///
/// Implementations return the rotation (and its rate) taking coordinates
/// **from** `from` **to** `to` at `epoch`. [`Iau76Fk5Provider`] is the
/// default; a custom implementation (test double, CIO-based reduction,
/// ANISE-backed provider, …) substitutes at any [`transform_state`] /
/// [`transform_state_covariance`] call site without other code changes.
pub trait FrameProvider {
    /// Produces the rotation mapping `from`-frame coordinates into
    /// `to`-frame coordinates at `epoch`.
    ///
    /// # Errors
    ///
    /// [`FrameError::UnsupportedPair`] when the provider has no rotation for
    /// this pair.
    fn rotation(&self, from: Frame, to: Frame, epoch: &Epoch) -> Result<FrameRotation, FrameError>;
}

/// The default [`FrameProvider`]: the IAU-76/FK5 reduction of the module
/// docs, composed through the chain for any frame pair.
///
/// # Earth-orientation parameters and accuracy floors
///
/// Both parameters require external Earth-orientation data that thresh does
/// not ingest yet, so they default to **zero** with these documented floors:
///
/// * `delta_ut1 = 0.0` assumes UT1 = UTC. IERS leap-second scheduling keeps
///   |ΔUT1| ≤ 0.9 s, which mis-rotates PEF/ITRF longitude by up to
///   0.9 s × ω⊕ ≈ 6.6e-5 rad — ≤ ~430 m at the equator (proportionally
///   less at altitude-normalized error angles, ~460 m at LEO radius).
/// * `polar_motion = None` omits **W**, making ITRF ≡ PEF. Pole offsets are
///   typically |xₚ|, |yₚ| ≲ 0.5″ ≈ 2.4e-6 rad — ≤ ~15 m of surface
///   displacement.
///
/// Inertial-only transforms (GCRF/MOD/TOD/TEME) do not depend on either
/// parameter.
///
/// # Example
///
/// ```
/// use nalgebra::{Matrix6, Vector3};
/// use thresh_core::frames::{Frame, FrameProvider, Iau76Fk5Provider};
/// use thresh_core::time::Epoch;
///
/// // EOP values are published in seconds (ΔUT1) and arcseconds (xp, yp).
/// let arcsec = std::f64::consts::PI / (180.0 * 3600.0);
/// let provider = Iau76Fk5Provider {
///     delta_ut1: -0.2,
///     polar_motion: Some((0.15 * arcsec, 0.30 * arcsec)),
/// };
/// let epoch = Epoch::from_gregorian_utc(2024, 1, 1, 0, 0, 0, 0);
/// let rot = provider.rotation(Frame::Gcrf, Frame::Itrf, &epoch).unwrap();
/// let (r_itrf, v_itrf) = rot.rotate_state(
///     &Vector3::new(7.0e6, 1.0e6, 2.0e6),
///     &Vector3::new(0.0, 7.5e3, 0.0),
/// );
/// let cov_itrf = rot.rotate_covariance(&Matrix6::identity());
/// assert!(cov_itrf[(0, 0)] > 0.0 && r_itrf.norm() > 0.0 && v_itrf.norm() > 0.0);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Iau76Fk5Provider {
    /// ΔUT1 = UT1 − UTC in **seconds** (IERS Bulletin A quantity; zero
    /// default ⇒ UT1 ≈ UTC, see the accuracy floors above).
    pub delta_ut1: f64,
    /// Polar-motion pole coordinates `(xp, yp)` in **radians**
    /// (bulletins publish arcseconds — multiply by π/(180·3600)); `None`
    /// omits the polar-motion leg entirely.
    pub polar_motion: Option<(f64, f64)>,
}

/// The per-epoch legs shared by every rotation the provider builds.
struct ChainLegs {
    /// GCRF → MOD (IAU-76 precession).
    precession: Matrix3<f64>,
    /// MOD → TOD (IAU-1980 nutation).
    nutation: Matrix3<f64>,
    /// Equation of the equinoxes (GAST − GMST), radians.
    eqe: f64,
    /// Greenwich Apparent Sidereal Time, radians.
    gast: f64,
    /// PEF → ITRF (polar motion; identity when EOP is absent).
    polar: Matrix3<f64>,
}

impl Iau76Fk5Provider {
    /// Phase 1: evaluate every leg of the chain once at `epoch`.
    fn chain_legs(&self, epoch: &Epoch) -> ChainLegs {
        let t_tt = legs::julian_centuries_tt(epoch);
        let nut = legs::nutation_iau1980(t_tt);
        let mean_obliquity = legs::mean_obliquity_iau1980(t_tt);
        let eqe = legs::equation_of_equinoxes_iau1994(t_tt, nut.dpsi, mean_obliquity);
        let gast = (legs::gmst_iau1982(epoch, self.delta_ut1) + eqe).rem_euclid(TAU);
        ChainLegs {
            precession: legs::precession_matrix_iau76(t_tt),
            nutation: legs::nutation_matrix(mean_obliquity, nut.dpsi, nut.deps),
            eqe,
            gast,
            polar: match self.polar_motion {
                Some((xp, yp)) => legs::polar_motion_matrix(xp, yp),
                None => Matrix3::identity(),
            },
        }
    }

    /// Phase 2: the rotation from GCRF into `frame` (with its rate).
    fn rotation_from_gcrf(chain: &ChainLegs, frame: Frame) -> FrameRotation {
        let nutation_precession = chain.nutation * chain.precession;
        match frame {
            Frame::Gcrf => inertial(Matrix3::identity()),
            Frame::Mod => inertial(chain.precession),
            Frame::Tod => inertial(nutation_precession),
            Frame::Teme => inertial(rotation_z(chain.eqe) * nutation_precession),
            Frame::Pef => spinning(&Matrix3::identity(), chain.gast, &nutation_precession),
            Frame::Itrf => spinning(&chain.polar, chain.gast, &nutation_precession),
        }
    }
}

impl FrameProvider for Iau76Fk5Provider {
    /// Composes any `from` → `to` pair through the chain (never errors:
    /// every [`Frame`] pair is supported).
    fn rotation(&self, from: Frame, to: Frame, epoch: &Epoch) -> Result<FrameRotation, FrameError> {
        let chain = self.chain_legs(epoch);
        let a = Self::rotation_from_gcrf(&chain, from);
        let b = Self::rotation_from_gcrf(&chain, to);
        Ok(compose_via_gcrf(&a, &b))
    }
}

/// An inertial leg: rotation with zero rate (see the module docs for why
/// precession/nutation rates are neglected).
fn inertial(r: Matrix3<f64>) -> FrameRotation {
    FrameRotation {
        r,
        r_dot: Matrix3::zeros(),
    }
}

/// An Earth-fixed leg: `W · R3(GAST) · N · P` with the ω⊕ rate on the GAST
/// spin (the only leg whose time dependence matters at tracking accuracy).
fn spinning(polar: &Matrix3<f64>, gast: f64, nutation_precession: &Matrix3<f64>) -> FrameRotation {
    FrameRotation {
        r: polar * rotation_z(gast) * nutation_precession,
        r_dot: polar * rotation_z_rate(gast) * nutation_precession * EARTH_ROTATION_RATE,
    }
}

/// Composes `from` → `to` from two GCRF-rooted rotations:
/// `R = R_to · R_fromᵀ`, `Ṙ = Ṙ_to · R_fromᵀ + R_to · Ṙ_fromᵀ`.
fn compose_via_gcrf(from: &FrameRotation, to: &FrameRotation) -> FrameRotation {
    let from_r_t = from.r.transpose();
    FrameRotation {
        r: to.r * from_r_t,
        r_dot: to.r_dot * from_r_t + to.r * from.r_dot.transpose(),
    }
}

/// d/dα of [`rotation_z`]`(α)`: multiplied by dα/dt = ω⊕ it forms the rate
/// block of the sidereal spin.
fn rotation_z_rate(angle: f64) -> Matrix3<f64> {
    let (s, c) = angle.sin_cos();
    Matrix3::new(-s, c, 0.0, -c, -s, 0.0, 0.0, 0.0, 0.0)
}

/// Transforms a position/velocity state (metres, m/s) between frames using
/// any [`FrameProvider`] — the provider-seam entry point.
///
/// # Errors
///
/// Propagates [`FrameError`] from the provider (the default
/// [`Iau76Fk5Provider`] never errors).
///
/// # Example
///
/// ```
/// use nalgebra::Vector3;
/// use thresh_core::frames::{transform_state, Frame, Iau76Fk5Provider};
/// use thresh_core::time::Epoch;
///
/// let epoch = Epoch::from_gregorian_utc(2024, 1, 1, 0, 0, 0, 0);
/// let (r_itrf, v_itrf) = transform_state(
///     &Iau76Fk5Provider::default(), // any FrameProvider substitutes here
///     Frame::Gcrf,
///     Frame::Itrf,
///     &epoch,
///     &Vector3::new(7.0e6, 1.0e6, 2.0e6),
///     &Vector3::new(0.0, 7.5e3, 0.0),
/// )
/// .unwrap();
/// // Going into the co-rotating frame removes the ~465 m/s equatorial spin.
/// assert!(r_itrf.norm() > 0.0 && v_itrf.norm() > 0.0);
/// ```
pub fn transform_state<P: FrameProvider + ?Sized>(
    provider: &P,
    from: Frame,
    to: Frame,
    epoch: &Epoch,
    position: &Vector3<f64>,
    velocity: &Vector3<f64>,
) -> Result<(Vector3<f64>, Vector3<f64>), FrameError> {
    let rot = provider.rotation(from, to, epoch)?;
    Ok(rot.rotate_state(position, velocity))
}

/// Position (metres), velocity (m/s), and 6×6 covariance returned by
/// [`transform_state_covariance`], all expressed in the target frame.
pub type TransformedStateCovariance = (Vector3<f64>, Vector3<f64>, Matrix6<f64>);

/// Transforms a state **and** its 6×6 covariance (position block first)
/// between frames using any [`FrameProvider`]; the covariance goes through
/// the `[[R, 0], [Ṙ, R]]` Jacobian (see
/// [`FrameRotation::rotate_covariance`]).
///
/// # Errors
///
/// Propagates [`FrameError`] from the provider (the default
/// [`Iau76Fk5Provider`] never errors).
pub fn transform_state_covariance<P: FrameProvider + ?Sized>(
    provider: &P,
    from: Frame,
    to: Frame,
    epoch: &Epoch,
    position: &Vector3<f64>,
    velocity: &Vector3<f64>,
    covariance: &Matrix6<f64>,
) -> Result<TransformedStateCovariance, FrameError> {
    let rot = provider.rotation(from, to, epoch)?;
    let (r, v) = rot.rotate_state(position, velocity);
    Ok((r, v, rot.rotate_covariance(covariance)))
}

/// The rotation between two frames from the **default** (zero-EOP)
/// [`Iau76Fk5Provider`] — use this to rotate covariances or many states at
/// one epoch without re-deriving the chain.
pub fn rotation_between(from: Frame, to: Frame, epoch: &Epoch) -> FrameRotation {
    Iau76Fk5Provider::default()
        .rotation(from, to, epoch)
        .expect("IAU-76/FK5 provider supports every Frame pair")
}

macro_rules! free_transform {
    ($(#[$doc:meta])* $name:ident, $from:expr, $to:expr) => {
        $(#[$doc])*
        ///
        /// Positions are metres, velocities m/s; the epoch's TT drives
        /// precession/nutation and UTC (≈ UT1, zero-EOP default) drives the
        /// sidereal legs. Uses the default [`Iau76Fk5Provider`]; construct a
        /// provider explicitly to supply EOP values or substitute a custom
        /// [`FrameProvider`].
        pub fn $name(
            position: &Vector3<f64>,
            velocity: &Vector3<f64>,
            epoch: &Epoch,
        ) -> (Vector3<f64>, Vector3<f64>) {
            rotation_between($from, $to, epoch).rotate_state(position, velocity)
        }
    };
}

free_transform!(
    /// TEME (SGP4 output frame) → GCRF.
    teme_to_gcrf,
    Frame::Teme,
    Frame::Gcrf
);
free_transform!(
    /// GCRF → TEME (SGP4 output frame).
    gcrf_to_teme,
    Frame::Gcrf,
    Frame::Teme
);
free_transform!(
    /// GCRF → ITRF (Earth-fixed; velocity loses the ω⊕ co-rotation term).
    gcrf_to_itrf,
    Frame::Gcrf,
    Frame::Itrf
);
free_transform!(
    /// ITRF (Earth-fixed) → GCRF (velocity regains the ω⊕ co-rotation
    /// term).
    itrf_to_gcrf,
    Frame::Itrf,
    Frame::Gcrf
);
free_transform!(
    /// TEME (SGP4 output frame) → ITRF (Earth-fixed).
    teme_to_itrf,
    Frame::Teme,
    Frame::Itrf
);
free_transform!(
    /// ITRF (Earth-fixed) → TEME (SGP4 output frame).
    itrf_to_teme,
    Frame::Itrf,
    Frame::Teme
);

#[cfg(test)]
mod tests {
    use super::*;

    const ARCSEC: f64 = std::f64::consts::PI / (180.0 * 3600.0);

    fn epoch_2024() -> Epoch {
        Epoch::from_gregorian_utc(2024, 1, 1, 0, 0, 0, 0)
    }

    /// The EOP set of the pyerfa reference run (session): ΔUT1 = −0.2 s,
    /// xp = 0.15″, yp = 0.30″.
    fn eop_provider() -> Iau76Fk5Provider {
        Iau76Fk5Provider {
            delta_ut1: -0.2,
            polar_motion: Some((0.15 * ARCSEC, 0.30 * ARCSEC)),
        }
    }

    fn r_gcrf_ref() -> Vector3<f64> {
        Vector3::new(7.0e6, 1.0e6, 2.0e6)
    }

    // -----------------------------------------------------------------
    // Reference values: generated this session with pyerfa 2.0.1.5 (ERFA,
    // SOFA-derived) via the scratchpad `gen_refs.py` end-to-end section —
    // pmat76/nutm80/gmst82/eqeq94/pom00 composed exactly as this module
    // composes them, at 2024-01-01T00:00:00 UTC with the EOP set above.
    // Tolerances (0.2 m) cover the ~2 cm GMST error from f64 JD
    // quantization of the Epoch accessors, nothing else.
    // -----------------------------------------------------------------

    /// Spec "IAU-76/FK5 transform chain": every leg of the GCRF→ITRF
    /// reduction matches the pyerfa composition, EOP included.
    #[test]
    fn provider_matches_pyerfa_reduction_with_explicit_eop() {
        let provider = eop_provider();
        let epoch = epoch_2024();
        let cases = [
            (
                Frame::Mod,
                Vector3::new(
                    6_989_849.576_460_495,
                    1_037_541.151_190_594_9,
                    2_016_311.299_884_305_3,
                ),
            ),
            (
                Frame::Tod,
                Vector3::new(
                    6_989_895.164_410_931,
                    1_037_295.657_931_955_9,
                    2_016_279.571_036_761_4,
                ),
            ),
            (
                Frame::Pef,
                Vector3::new(
                    -210_790.901_869_788_77,
                    -7_063_298.371_660_986,
                    2_016_279.571_036_761_4,
                ),
            ),
            (
                Frame::Itrf,
                Vector3::new(
                    -210_789.435_589_851_55,
                    -7_063_301.304_213_501,
                    2_016_269.451_174_512,
                ),
            ),
        ];
        for (frame, expected) in cases {
            let rot = provider.rotation(Frame::Gcrf, frame, &epoch).unwrap();
            let got = rot.rotate_position(&r_gcrf_ref());
            assert!(
                (got - expected).norm() < 0.2,
                "GCRF->{frame}: off by {} m",
                (got - expected).norm()
            );
        }
    }

    /// Spec "Explicit EOP parameters are honored": non-zero ΔUT1 and polar
    /// motion shift the ITRF result measurably; both the zero-EOP position
    /// and the shift magnitude match the pyerfa reference run.
    #[test]
    fn explicit_eop_parameters_are_honored() {
        let epoch = epoch_2024();
        let with_eop = eop_provider()
            .rotation(Frame::Gcrf, Frame::Itrf, &epoch)
            .unwrap()
            .rotate_position(&r_gcrf_ref());
        let zero_eop = Iau76Fk5Provider::default()
            .rotation(Frame::Gcrf, Frame::Itrf, &epoch)
            .unwrap()
            .rotate_position(&r_gcrf_ref());

        let expected_zero = Vector3::new(
            -210_893.914_627_434_07,
            -7_063_295.296_686_449,
            2_016_279.571_036_761_4,
        );
        assert!((zero_eop - expected_zero).norm() < 0.2);

        let shift = (with_eop - zero_eop).norm();
        assert!(shift > 100.0, "EOP must shift ITRF measurably: {shift} m");
        // pyerfa reference: 105.13977024604442 m.
        assert!((shift - 105.139_770_246_044_42).abs() < 0.05);
    }

    /// Spec "TEME and GCRF differ by precession and nutation": km-scale
    /// displacement at LEO magnitude and a modern epoch (pyerfa reference:
    /// 40932.31280163325 m), and the round trip recovers the input.
    #[test]
    fn teme_gcrf_displacement_is_kilometre_scale_and_round_trips() {
        let epoch = epoch_2024();
        let r_teme = Vector3::new(7.0e6, 0.0, 0.0);
        let v_teme = Vector3::new(0.0, 7_500.0, 0.0);

        let (r_gcrf, v_gcrf) = teme_to_gcrf(&r_teme, &v_teme, &epoch);
        let expected = Vector3::new(
            6_999_880.324_697_765_5,
            -37_568.035_431_982_35,
            -16_250.619_154_176_94,
        );
        assert!((r_gcrf - expected).norm() < 0.2);

        let displacement = (r_gcrf - r_teme).norm();
        assert!(
            displacement > 1_000.0 && displacement < 100_000.0,
            "expected km-scale TEME/GCRF displacement, got {displacement} m"
        );
        assert!((displacement - 40_932.312_801_633_25).abs() < 0.2);

        let (r_back, v_back) = gcrf_to_teme(&r_gcrf, &v_gcrf, &epoch);
        assert!((r_back - r_teme).norm() < 1e-6);
        assert!((v_back - v_teme).norm() < 1e-9);
    }

    /// The TEME junction is consistent by construction: TEME → PEF is the
    /// pure R3(GMST) spin (design: "TEME —R₃(GMST)→ PEF").
    #[test]
    fn teme_to_pef_is_pure_gmst_spin() {
        let epoch = epoch_2024();
        let rot = Iau76Fk5Provider::default()
            .rotation(Frame::Teme, Frame::Pef, &epoch)
            .unwrap();
        let gmst = legs::gmst_iau1982(&epoch, 0.0);
        assert!((rot.r - rotation_z(gmst)).norm() < 1e-12);
    }

    /// Spec "Round trip through the full chain": GCRF→ITRF→GCRF recovers
    /// position to millimetres and velocity to micrometres per second
    /// (exercised with full EOP so every leg participates).
    #[test]
    fn gcrf_itrf_round_trip_within_mm_and_um_per_s() {
        let provider = eop_provider();
        let epoch = epoch_2024();
        let r = Vector3::new(6_778_137.0, -1_200_000.0, 3_400_000.0);
        let v = Vector3::new(1_500.0, 7_100.0, -2_300.0);

        let (r_itrf, v_itrf) =
            transform_state(&provider, Frame::Gcrf, Frame::Itrf, &epoch, &r, &v).unwrap();
        let (r_back, v_back) = transform_state(
            &provider,
            Frame::Itrf,
            Frame::Gcrf,
            &epoch,
            &r_itrf,
            &v_itrf,
        )
        .unwrap();

        assert!((r_back - r).norm() < 1e-3, "{} m", (r_back - r).norm());
        assert!((v_back - v).norm() < 1e-6, "{} m/s", (v_back - v).norm());
    }

    /// Structural: every from→to pair at a spread of epochs is a proper
    /// rotation (R·Rᵀ = I, det = 1) — no external truth needed.
    #[test]
    fn all_pairs_are_proper_rotations() {
        let provider = eop_provider();
        let epochs = [
            Epoch::from_gregorian_utc(2010, 6, 15, 12, 0, 0, 0),
            epoch_2024(),
            Epoch::from_gregorian_utc(2030, 3, 20, 6, 30, 0, 0),
        ];
        for epoch in &epochs {
            for from in Frame::ALL {
                for to in Frame::ALL {
                    let rot = provider.rotation(from, to, epoch).unwrap();
                    assert!(
                        (rot.r * rot.r.transpose() - Matrix3::identity()).norm() < 1e-12,
                        "{from}->{to} not orthonormal"
                    );
                    assert!(
                        (rot.r.determinant() - 1.0).abs() < 1e-12,
                        "{from}->{to} det"
                    );
                }
            }
        }
    }

    /// Inertial ↔ inertial rates are exactly zero (the documented
    /// approximation), and same-frame rotations are the identity with zero
    /// rate even for Earth-fixed frames (Ṙ·Rᵀ + R·Ṙᵀ = d/dt(I) = 0).
    #[test]
    fn inertial_pairs_carry_zero_rate() {
        let provider = eop_provider();
        let epoch = epoch_2024();
        let inertial_frames = [Frame::Gcrf, Frame::Mod, Frame::Tod, Frame::Teme];
        for from in inertial_frames {
            for to in inertial_frames {
                let rot = provider.rotation(from, to, &epoch).unwrap();
                assert_eq!(rot.r_dot.norm(), 0.0, "{from}->{to} rate must be zero");
            }
        }
        let same = provider.rotation(Frame::Itrf, Frame::Itrf, &epoch).unwrap();
        assert!((same.r - Matrix3::identity()).norm() < 1e-14);
        assert!(same.r_dot.norm() < 1e-18);
    }

    /// The Earth-fixed rate block matches a central finite difference of
    /// the rotation itself: r_dot really is dR/dt. Tolerance covers the
    /// ~2 cm/s of finite-difference noise from f64 JD quantization
    /// (~465 m/s signal).
    #[test]
    fn earth_fixed_rate_matches_finite_difference() {
        let provider = eop_provider();
        let epoch = epoch_2024();
        let r = r_gcrf_ref();
        let dt = 0.5;
        for to in [Frame::Pef, Frame::Itrf] {
            let rot = provider.rotation(Frame::Gcrf, to, &epoch).unwrap();
            let plus = provider.rotation(Frame::Gcrf, to, &(epoch + dt)).unwrap();
            let minus = provider
                .rotation(Frame::Gcrf, to, &(epoch + (-dt)))
                .unwrap();
            let finite_difference = (plus.r * r - minus.r * r) / (2.0 * dt);
            let analytic = rot.r_dot * r;
            assert!(analytic.norm() > 400.0, "rate term missing for {to}");
            assert!(
                (finite_difference - analytic).norm() < 0.05,
                "GCRF->{to}: dR/dt mismatch {} m/s",
                (finite_difference - analytic).norm()
            );
        }
    }

    /// Spec "Rotating a covariance into ITRF": position block is R·P_rr·Rᵀ,
    /// the velocity block carries the ω⊕ coupling (Ṙ·P_rr·Ṙᵀ, since
    /// P_rv = 0 here), and the result stays symmetric PSD.
    #[test]
    fn covariance_into_itrf_reflects_omega_coupling_and_stays_psd() {
        let epoch = epoch_2024();
        let rot = Iau76Fk5Provider::default()
            .rotation(Frame::Gcrf, Frame::Itrf, &epoch)
            .unwrap();

        // Anisotropic diagonal: position variances (m²), velocity (m²/s²).
        let mut p = Matrix6::zeros();
        p[(0, 0)] = 1.0e6;
        p[(1, 1)] = 4.0e6;
        p[(2, 2)] = 9.0e6;
        p[(3, 3)] = 1.0;
        p[(4, 4)] = 4.0;
        p[(5, 5)] = 9.0;

        let q = rot.rotate_covariance(&p);
        let p_rr = p.fixed_view::<3, 3>(0, 0).into_owned();
        let p_vv = p.fixed_view::<3, 3>(3, 3).into_owned();
        let q_rr = q.fixed_view::<3, 3>(0, 0).into_owned();
        let q_vv = q.fixed_view::<3, 3>(3, 3).into_owned();

        let expected_rr = rot.r * p_rr * rot.r.transpose();
        assert!((q_rr - expected_rr).norm() < 1e-6 * expected_rr.norm());

        let rotated_vv = rot.r * p_vv * rot.r.transpose();
        let coupling = rot.r_dot * p_rr * rot.r_dot.transpose();
        assert!((q_vv - (rotated_vv + coupling)).norm() < 1e-9 * rotated_vv.norm());
        assert!(
            coupling.norm() > 1e-3,
            "omega-earth coupling must be visible: {:e}",
            coupling.norm()
        );

        assert!((q - q.transpose()).norm() < 1e-9);
        let eigenvalues = q.symmetric_eigenvalues();
        let max_eig = eigenvalues.max();
        assert!(
            eigenvalues.iter().all(|&l| l > -1e-9 * max_eig),
            "covariance must stay PSD: {eigenvalues:?}"
        );

        // Position-only 3×3 variant agrees with the position block.
        let q3 = rot.rotate_position_covariance(&p_rr);
        assert!((q3 - expected_rr).norm() < 1e-9 * expected_rr.norm());
    }

    /// Spec "Covariance round trip preserves the matrix": GCRF→TEME→GCRF on
    /// a dense symmetric PSD matrix (inertial pair ⇒ pure block rotation).
    #[test]
    fn covariance_round_trip_gcrf_teme_gcrf() {
        let epoch = epoch_2024();
        let provider = Iau76Fk5Provider::default();
        let a = Matrix6::from_fn(|i, j| {
            ((i * 6 + j) as f64 * 0.37).sin() + if i == j { 2.0 } else { 0.0 }
        });
        let p = a * a.transpose();

        let forward = provider.rotation(Frame::Gcrf, Frame::Teme, &epoch).unwrap();
        let back = provider.rotation(Frame::Teme, Frame::Gcrf, &epoch).unwrap();
        let p_round = back.rotate_covariance(&forward.rotate_covariance(&p));
        assert!(
            (p_round - p).norm() < 1e-12 * p.norm(),
            "relative error {:e}",
            (p_round - p).norm() / p.norm()
        );
    }

    /// A fixed quarter-turn about +z with no rate: a stand-in for any
    /// externally sourced rotation.
    struct QuarterTurnProvider;

    impl FrameProvider for QuarterTurnProvider {
        fn rotation(
            &self,
            _from: Frame,
            _to: Frame,
            _epoch: &Epoch,
        ) -> Result<FrameRotation, FrameError> {
            Ok(FrameRotation {
                r: rotation_z(std::f64::consts::FRAC_PI_2),
                r_dot: Matrix3::zeros(),
            })
        }
    }

    /// Spec "Custom provider substitutes without call-site changes": the
    /// same `transform_state` call site takes the test provider and the
    /// default provider interchangeably, and the substituted rotation is
    /// the one actually used.
    #[test]
    fn custom_provider_substitutes_without_call_site_changes() {
        let epoch = epoch_2024();
        let pos = Vector3::new(1.0, 0.0, 0.0);
        let vel = Vector3::zeros();

        let (custom, _) = transform_state(
            &QuarterTurnProvider,
            Frame::Teme,
            Frame::Gcrf,
            &epoch,
            &pos,
            &vel,
        )
        .unwrap();
        let (default, _) = transform_state(
            &Iau76Fk5Provider::default(),
            Frame::Teme,
            Frame::Gcrf,
            &epoch,
            &pos,
            &vel,
        )
        .unwrap();

        assert!((custom - Vector3::new(0.0, -1.0, 0.0)).norm() < 1e-15);
        assert!(
            (custom - default).norm() > 0.1,
            "substituted provider must be used"
        );

        // And through a &dyn seam, unchanged call shape.
        let dyn_provider: &dyn FrameProvider = &QuarterTurnProvider;
        let (via_dyn, _) =
            transform_state(dyn_provider, Frame::Teme, Frame::Gcrf, &epoch, &pos, &vel).unwrap();
        assert_eq!(via_dyn, custom);
    }

    /// A provider that refuses every pair, to exercise the error path.
    struct RefusingProvider;

    impl FrameProvider for RefusingProvider {
        fn rotation(
            &self,
            from: Frame,
            to: Frame,
            _epoch: &Epoch,
        ) -> Result<FrameRotation, FrameError> {
            Err(FrameError::UnsupportedPair { from, to })
        }
    }

    /// Provider errors surface through the transform entry points, and the
    /// error names both frames.
    #[test]
    fn provider_errors_surface_through_transforms() {
        let epoch = epoch_2024();
        let err = transform_state(
            &RefusingProvider,
            Frame::Gcrf,
            Frame::Itrf,
            &epoch,
            &Vector3::zeros(),
            &Vector3::zeros(),
        )
        .unwrap_err();
        assert_eq!(
            err,
            FrameError::UnsupportedPair {
                from: Frame::Gcrf,
                to: Frame::Itrf
            }
        );
        let message = err.to_string();
        assert!(
            message.contains("GCRF") && message.contains("ITRF"),
            "{message}"
        );
    }

    /// Free functions agree with the explicit provider path bit-for-bit and
    /// compose consistently (TEME→ITRF direct vs via GCRF).
    #[test]
    fn free_functions_match_provider_path() {
        let epoch = epoch_2024();
        let r = Vector3::new(6_778_137.0, -1_200_000.0, 3_400_000.0);
        let v = Vector3::new(1_500.0, 7_100.0, -2_300.0);

        let via_free = gcrf_to_itrf(&r, &v, &epoch);
        let via_provider = transform_state(
            &Iau76Fk5Provider::default(),
            Frame::Gcrf,
            Frame::Itrf,
            &epoch,
            &r,
            &v,
        )
        .unwrap();
        assert_eq!(via_free, via_provider);

        let (r_direct, v_direct) = teme_to_itrf(&r, &v, &epoch);
        let (r_mid, v_mid) = teme_to_gcrf(&r, &v, &epoch);
        let (r_via, v_via) = gcrf_to_itrf(&r_mid, &v_mid, &epoch);
        assert!((r_direct - r_via).norm() < 1e-6);
        assert!((v_direct - v_via).norm() < 1e-9);

        let (r_back, v_back) = itrf_to_teme(&r_direct, &v_direct, &epoch);
        assert!((r_back - r).norm() < 1e-3);
        assert!((v_back - v).norm() < 1e-6);

        let (r_g, v_g) = itrf_to_gcrf(&r_direct, &v_direct, &epoch);
        assert!((r_g - r_mid).norm() < 1e-6);
        assert!((v_g - v_mid).norm() < 1e-9);
    }

    /// `FrameRotation::inverse` and `jacobian` layout.
    #[test]
    fn frame_rotation_inverse_and_jacobian_layout() {
        let epoch = epoch_2024();
        let rot = eop_provider()
            .rotation(Frame::Gcrf, Frame::Itrf, &epoch)
            .unwrap();

        let inv = rot.inverse();
        assert!((inv.r * rot.r - Matrix3::identity()).norm() < 1e-14);
        assert_eq!(inv.r_dot, rot.r_dot.transpose());

        let j = rot.jacobian();
        assert_eq!(j.fixed_view::<3, 3>(0, 0).into_owned(), rot.r);
        assert_eq!(j.fixed_view::<3, 3>(3, 3).into_owned(), rot.r);
        assert_eq!(j.fixed_view::<3, 3>(3, 0).into_owned(), rot.r_dot);
        assert_eq!(j.fixed_view::<3, 3>(0, 3).into_owned(), Matrix3::zeros());
    }

    /// `Frame` display names, Earth-fixed classification, and serde round
    /// trip.
    #[test]
    fn frame_display_classification_and_serde() {
        let names = ["GCRF", "MOD", "TOD", "TEME", "PEF", "ITRF"];
        for (frame, name) in Frame::ALL.into_iter().zip(names) {
            assert_eq!(frame.to_string(), name);
        }
        assert!(Frame::Pef.is_earth_fixed());
        assert!(Frame::Itrf.is_earth_fixed());
        assert!(!Frame::Gcrf.is_earth_fixed());
        assert!(!Frame::Teme.is_earth_fixed());

        let json = serde_json::to_string(&Frame::Teme).unwrap();
        assert_eq!(json, "\"TEME\"", "serde form matches Display and fixtures");
        assert_eq!(serde_json::from_str::<Frame>(&json).unwrap(), Frame::Teme);
    }

    /// `transform_state_covariance` bundles state and covariance through
    /// one provider call, agreeing with the piecewise path.
    #[test]
    fn transform_state_covariance_matches_piecewise() {
        let epoch = epoch_2024();
        let provider = eop_provider();
        let r = r_gcrf_ref();
        let v = Vector3::new(0.0, 7.5e3, 0.0);
        let p = Matrix6::identity() * 100.0;

        let (r1, v1, p1) =
            transform_state_covariance(&provider, Frame::Gcrf, Frame::Itrf, &epoch, &r, &v, &p)
                .unwrap();
        let rot = provider.rotation(Frame::Gcrf, Frame::Itrf, &epoch).unwrap();
        let (r2, v2) = rot.rotate_state(&r, &v);
        assert_eq!((r1, v1), (r2, v2));
        assert_eq!(p1, rot.rotate_covariance(&p));
    }
}
