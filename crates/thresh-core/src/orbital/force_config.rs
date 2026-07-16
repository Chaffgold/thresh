//! Configurable force-model stack composing into one time-aware
//! acceleration closure (design Decision 2 of `orbit-propagation-fidelity`).
//!
//! [`ForceModelConfig`] selects a gravity fidelity tier plus optional drag,
//! solar-radiation pressure, and lunisolar third-body forces;
//! [`ForceModelConfig::build`] sums the enabled terms into a single closure
//! on the **time-aware seam**
//! `Fn(f64, &r, &v) -> a` (design Decision 1 — see
//! [`super::dormand_prince::TimeAwareAccel`]), with the arc's [`Epoch`]
//! bound at construction. The existing time-free seam and all its
//! consumers (filter models, `rk4_step`, calibrated benchmarks) are
//! untouched.
//!
//! # Earth-fixed legs go through the `FrameProvider`
//!
//! Forces that are naturally Earth-fixed — spherical-harmonic gravity and
//! the co-rotating Harris-Priester atmosphere — rotate GCRF → ITRF through
//! the supplied [`FrameProvider`] at `epoch + t` and rotate the resulting
//! acceleration back, never through an ad-hoc rotation (spec: "Earth-fixed
//! force legs use the frame provider"). Substituting a provider with
//! explicit EOP therefore propagates into force evaluation with no code
//! changes.
//!
//! # Constant-set discipline
//!
//! Every gravity tier evaluates with the **EGM96 constant set**
//! ([`egm96_gravity_model`]: μ = 3.986004415e14 m³/s², a = 6378136.3 m,
//! J2 from C̄₂₀) so that switching fidelity never mixes constants — the
//! same set the golden fixtures record for all tiers including the J2
//! baseline (`test-data/golden/propagation/`).

use nalgebra::Vector3;
use serde::{Deserialize, Serialize};

use super::atmosphere::{drag_acceleration, harris_priester_density_itrf};
use super::egm96::{egm96_acceleration_itrf, egm96_gravity_model};
use super::ephemeris::{moon_position_gcrf, sun_position_gcrf};
use super::gravity::{
    GravityModel, j2_acceleration, j3_acceleration, j4_acceleration, two_body_acceleration,
};
use super::srp::srp_acceleration;
use super::third_body::{GM_MOON, GM_SUN, third_body_acceleration};
use crate::frames::{Frame, FrameProvider, FrameRotation};
use crate::time::Epoch;

/// Gravity fidelity tier (design Decision 2). All tiers share the EGM96
/// constant set — see the module docs.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum GravityFidelity {
    /// Point-mass central body only.
    TwoBody,
    /// Two-body + J2, the existing [`j2_acceleration`] math.
    J2,
    /// Two-body + J2 plus the closed-form J3/J4 zonal perturbations
    /// (typically [`super::gravity::EARTH_J3`] / [`super::gravity::EARTH_J4`],
    /// or the table-derived `egm96_jn(n)`).
    Zonal {
        /// J₃ zonal coefficient (dimensionless).
        j3: f64,
        /// J₄ zonal coefficient (dimensionless).
        j4: f64,
    },
    /// Truncated EGM96 spherical harmonics
    /// ([`egm96_acceleration_itrf`], clamped to its embedded 12×12 table),
    /// evaluated in ITRF through the frame provider.
    Harmonics {
        /// Maximum harmonic degree n.
        degree: u32,
        /// Maximum harmonic order m (clamped to `degree`).
        order: u32,
    },
}

/// Atmospheric-drag configuration: which density model feeds the
/// ballistic-coefficient drag term.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum DragConfig {
    /// Harris-Priester density ([`harris_priester_density_itrf`]),
    /// evaluated in ITRF through the frame provider with the true
    /// co-rotation relative velocity and the diurnal bulge toward the
    /// epoch Sun.
    HarrisPriester {
        /// Inverse ballistic coefficient `Cd·A/m` (m²/kg).
        inv_beta: f64,
    },
    /// The existing piecewise-exponential model via [`drag_acceleration`]
    /// (time-free, inertial-frame co-rotation approximation — unchanged).
    Exponential {
        /// Inverse ballistic coefficient `Cd·A/m` (m²/kg).
        inv_beta: f64,
    },
}

/// Cannonball SRP configuration (cylindrical Earth shadow — see
/// [`super::srp`] for the documented approximation and the solar-pressure
/// convention).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct SrpConfig {
    /// Reflectivity coefficient times area over mass, `Cr·A/m` (m²/kg).
    pub cr_area_over_mass: f64,
}

/// Lunisolar third-body toggles (positions from [`super::ephemeris`],
/// GM values from [`super::third_body`]).
#[derive(Debug, Clone, Copy, PartialEq, Default, Serialize, Deserialize)]
pub struct ThirdBodyConfig {
    /// Enable the solar third-body term.
    pub sun: bool,
    /// Enable the lunar third-body term.
    pub moon: bool,
}

/// Selectable perturbation-force stack (design Decision 2's shape). Each
/// force is individually toggleable; the default — J2 gravity with every
/// optional force disabled — reproduces the existing two-body/J2 math
/// exactly.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ForceModelConfig {
    /// Gravity fidelity tier.
    pub gravity: GravityFidelity,
    /// Optional atmospheric drag.
    pub drag: Option<DragConfig>,
    /// Optional cannonball SRP with cylindrical eclipse.
    pub srp: Option<SrpConfig>,
    /// Lunisolar third-body toggles.
    pub third_body: ThirdBodyConfig,
}

impl Default for ForceModelConfig {
    /// J2 gravity, no drag, no SRP, no third bodies — the existing
    /// baseline behavior.
    fn default() -> Self {
        Self {
            gravity: GravityFidelity::J2,
            drag: None,
            srp: None,
            third_body: ThirdBodyConfig::default(),
        }
    }
}

impl ForceModelConfig {
    /// Composes the enabled forces into one time-aware acceleration
    /// closure `(t_s, &position_m, &velocity_m_s) -> acceleration_m_s2`,
    /// where `t_s` is seconds past `epoch` — directly consumable by
    /// [`super::dormand_prince::propagate_on_grid`] /
    /// [`super::dormand_prince::propagate_orbital_state`].
    ///
    /// Position and velocity are **GCRF**; Earth-fixed legs rotate through
    /// `provider` at `epoch + t_s` (module docs). The returned closure
    /// borrows `provider`.
    ///
    /// # Panics
    ///
    /// The closure panics if `provider` cannot produce the GCRF → ITRF
    /// rotation needed by an Earth-fixed leg (the default
    /// [`crate::frames::Iau76Fk5Provider`] supports every pair and never
    /// errors).
    ///
    /// # Example
    ///
    /// ```
    /// use nalgebra::Vector3;
    /// use thresh_core::frames::Iau76Fk5Provider;
    /// use thresh_core::orbital::egm96::egm96_gravity_model;
    /// use thresh_core::orbital::force_config::ForceModelConfig;
    /// use thresh_core::orbital::gravity::j2_acceleration;
    /// use thresh_core::time::Epoch;
    ///
    /// let provider = Iau76Fk5Provider::default();
    /// let epoch = Epoch::from_gregorian_utc(2026, 2, 1, 6, 30, 0, 0);
    /// let accel = ForceModelConfig::default().build(epoch, &provider);
    ///
    /// let r = Vector3::new(6.878e6, 0.0, 0.0);
    /// let v = Vector3::new(0.0, 7.61e3, 0.0);
    /// // The default configuration is exactly the existing J2 math.
    /// assert_eq!(accel(0.0, &r, &v), j2_acceleration(&r, &egm96_gravity_model()));
    /// ```
    pub fn build<P: FrameProvider>(
        self,
        epoch: Epoch,
        provider: &P,
    ) -> impl Fn(f64, &Vector3<f64>, &Vector3<f64>) -> Vector3<f64> {
        self.validate();
        let model = egm96_gravity_model();
        move |t_s: f64, position: &Vector3<f64>, velocity: &Vector3<f64>| {
            let at = epoch + t_s;
            let earth_fixed = self
                .needs_earth_fixed()
                .then(|| gcrf_to_itrf(provider, &at));
            let sun = self.needs_sun().then(|| sun_position_gcrf(&at));
            gravity_leg(&self.gravity, &model, position, earth_fixed.as_ref())
                + drag_leg(
                    self.drag,
                    position,
                    velocity,
                    earth_fixed.as_ref(),
                    sun.as_ref(),
                )
                + srp_leg(self.srp, position, sun.as_ref())
                + third_body_leg(self.third_body, position, sun.as_ref(), &at)
        }
    }

    /// Panic with a clear message on physically invalid coefficients —
    /// a negative or non-finite `Cd·A/m` / `Cr·A/m` silently reverses or
    /// poisons the acceleration, and non-finite zonals poison gravity.
    /// Checked once at [`ForceModelConfig::build`], never per evaluation
    /// (same pattern as the integrator's `validate_config`).
    fn validate(&self) {
        if let GravityFidelity::Zonal { j3, j4 } = self.gravity {
            assert!(
                j3.is_finite() && j4.is_finite(),
                "zonal coefficients must be finite, got j3 = {j3}, j4 = {j4}"
            );
        }
        if let Some(
            DragConfig::HarrisPriester { inv_beta } | DragConfig::Exponential { inv_beta },
        ) = self.drag
        {
            assert!(
                inv_beta.is_finite() && inv_beta >= 0.0,
                "drag inv_beta (Cd*A/m) must be finite and >= 0, got {inv_beta}"
            );
        }
        if let Some(srp) = self.srp {
            assert!(
                srp.cr_area_over_mass.is_finite() && srp.cr_area_over_mass >= 0.0,
                "SRP Cr*A/m must be finite and >= 0, got {}",
                srp.cr_area_over_mass
            );
        }
    }

    /// True when an enabled force evaluates in the Earth-fixed frame
    /// (harmonic gravity or Harris-Priester drag) — the closure then
    /// computes the provider's GCRF → ITRF rotation once per evaluation.
    fn needs_earth_fixed(&self) -> bool {
        matches!(self.gravity, GravityFidelity::Harmonics { .. })
            || matches!(self.drag, Some(DragConfig::HarrisPriester { .. }))
    }

    /// True when an enabled force needs the epoch Sun position (SRP,
    /// the Harris-Priester diurnal bulge, or the solar third body).
    fn needs_sun(&self) -> bool {
        matches!(self.drag, Some(DragConfig::HarrisPriester { .. }))
            || self.srp.is_some()
            || self.third_body.sun
    }
}

/// The provider's GCRF → ITRF rotation at `at` (see
/// [`ForceModelConfig::build`]'s panic contract).
fn gcrf_to_itrf<P: FrameProvider>(provider: &P, at: &Epoch) -> FrameRotation {
    provider
        .rotation(Frame::Gcrf, Frame::Itrf, at)
        .expect("force stack requires a provider supporting GCRF -> ITRF")
}

/// Gravity leg. Precondition: `earth_fixed` is `Some` for the
/// `Harmonics` tier (established by `needs_earth_fixed` in the builder).
fn gravity_leg(
    gravity: &GravityFidelity,
    model: &GravityModel,
    position: &Vector3<f64>,
    earth_fixed: Option<&FrameRotation>,
) -> Vector3<f64> {
    match *gravity {
        GravityFidelity::TwoBody => two_body_acceleration(position, model),
        GravityFidelity::J2 => j2_acceleration(position, model),
        GravityFidelity::Zonal { j3, j4 } => {
            j2_acceleration(position, model)
                + j3_acceleration(position, model, j3)
                + j4_acceleration(position, model, j4)
        }
        GravityFidelity::Harmonics { degree, order } => {
            let rot = earth_fixed.expect("ITRF rotation precomputed for harmonics");
            let r_itrf = rot.rotate_position(position);
            rot.inverse()
                .rotate_position(&egm96_acceleration_itrf(&r_itrf, degree, order))
        }
    }
}

/// Drag leg (zero when disabled). Precondition: `earth_fixed` and `sun`
/// are `Some` for the Harris-Priester variant (established by the builder).
fn drag_leg(
    drag: Option<DragConfig>,
    position: &Vector3<f64>,
    velocity: &Vector3<f64>,
    earth_fixed: Option<&FrameRotation>,
    sun: Option<&Vector3<f64>>,
) -> Vector3<f64> {
    match drag {
        None => Vector3::zeros(),
        Some(DragConfig::Exponential { inv_beta }) => {
            drag_acceleration(position, velocity, inv_beta)
        }
        Some(DragConfig::HarrisPriester { inv_beta }) => harris_priester_drag(
            position,
            velocity,
            inv_beta,
            earth_fixed.expect("ITRF rotation precomputed for Harris-Priester drag"),
            sun.expect("Sun position precomputed for the Harris-Priester bulge"),
        ),
    }
}

/// Harris-Priester drag in the Earth-fixed frame: rotate the state into
/// ITRF (the provider's `Ṙ·r` term subtracts the ω⊕ co-rotation velocity
/// exactly — the fixture's `v_itrf = R·v − ω × (R·r)` convention), apply
/// `a = −½·ρ·‖v‖·v·(Cd·A/m)` with the bulge density toward the ITRF Sun,
/// and rotate the acceleration back.
fn harris_priester_drag(
    position: &Vector3<f64>,
    velocity: &Vector3<f64>,
    inv_beta: f64,
    rot: &FrameRotation,
    sun_gcrf: &Vector3<f64>,
) -> Vector3<f64> {
    let (r_itrf, v_itrf) = rot.rotate_state(position, velocity);
    let sun_itrf = rot.rotate_position(sun_gcrf);
    let rho = harris_priester_density_itrf(&r_itrf, &sun_itrf);
    let a_itrf = -0.5 * rho * v_itrf.norm() * inv_beta * v_itrf;
    rot.inverse().rotate_position(&a_itrf)
}

/// SRP leg (zero when disabled). Precondition: `sun` is `Some` when
/// enabled (established by the builder).
fn srp_leg(
    srp: Option<SrpConfig>,
    position: &Vector3<f64>,
    sun: Option<&Vector3<f64>>,
) -> Vector3<f64> {
    match srp {
        None => Vector3::zeros(),
        Some(config) => srp_acceleration(
            position,
            sun.expect("Sun position precomputed for SRP"),
            config.cr_area_over_mass,
        ),
    }
}

/// Third-body leg (zero when both bodies are disabled). Precondition:
/// `sun` is `Some` when the solar term is enabled (established by the
/// builder); the Moon position is evaluated here since nothing else needs
/// it.
fn third_body_leg(
    config: ThirdBodyConfig,
    position: &Vector3<f64>,
    sun: Option<&Vector3<f64>>,
    at: &Epoch,
) -> Vector3<f64> {
    let mut accel = Vector3::zeros();
    if config.sun {
        let sun = sun.expect("Sun position precomputed for the solar third body");
        accel += third_body_acceleration(position, sun, GM_SUN);
    }
    if config.moon {
        accel += third_body_acceleration(position, &moon_position_gcrf(at), GM_MOON);
    }
    accel
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::frames::Iau76Fk5Provider;
    use crate::orbital::dormand_prince::{DpConfig, propagate_on_grid};
    use crate::orbital::gravity::{EARTH_J3, EARTH_J4};

    /// The LEO golden arc's epoch and initial state
    /// (`test-data/golden/propagation/leo-full-force.json`).
    fn leo_epoch() -> Epoch {
        Epoch::from_iso8601("2026-02-01T06:30:00.000Z").unwrap()
    }

    fn leo_state() -> (Vector3<f64>, Vector3<f64>) {
        (
            Vector3::new(3_949_660.918608629, 5_192_041.83008058, 1_814_987.910218083),
            Vector3::new(-4_891.21762680897, 1_744.6892050458443, 5_653.01383781457),
        )
    }

    /// The LEO golden arc's full stack: EGM96 12×12, Harris-Priester drag,
    /// SRP, Sun+Moon third body (parameters from the fixture).
    fn full_stack() -> ForceModelConfig {
        ForceModelConfig {
            gravity: GravityFidelity::Harmonics {
                degree: 12,
                order: 12,
            },
            drag: Some(DragConfig::HarrisPriester { inv_beta: 0.011 }),
            srp: Some(SrpConfig {
                cr_area_over_mass: 0.0065,
            }),
            third_body: ThirdBodyConfig {
                sun: true,
                moon: true,
            },
        }
    }

    fn gravity_only(gravity: GravityFidelity) -> ForceModelConfig {
        ForceModelConfig {
            gravity,
            drag: None,
            srp: None,
            third_body: ThirdBodyConfig::default(),
        }
    }

    // ── Spec: "Baseline configuration matches the existing math" ─────────

    /// Two-body and J2 tiers reproduce the existing `two_body_acceleration`
    /// / `j2_acceleration` paths **bitwise** (same functions, same EGM96
    /// constant set), at t = 0 and mid-arc.
    #[test]
    fn baseline_tiers_match_existing_math_bitwise() {
        let provider = Iau76Fk5Provider::default();
        let (r, v) = leo_state();
        let model = egm96_gravity_model();

        let two_body = gravity_only(GravityFidelity::TwoBody).build(leo_epoch(), &provider);
        let j2 = gravity_only(GravityFidelity::J2).build(leo_epoch(), &provider);
        for t in [0.0, 500.0] {
            assert_eq!(two_body(t, &r, &v), two_body_acceleration(&r, &model));
            assert_eq!(j2(t, &r, &v), j2_acceleration(&r, &model));
        }
    }

    /// The `Default` configuration (every optional force disabled) is the
    /// J2 baseline — the requirement's "disabling every optional force
    /// SHALL reproduce the existing two-body/J2 behavior".
    #[test]
    fn default_config_is_the_j2_baseline() {
        let provider = Iau76Fk5Provider::default();
        let (r, v) = leo_state();
        let accel = ForceModelConfig::default().build(leo_epoch(), &provider);
        assert_eq!(
            accel(0.0, &r, &v),
            j2_acceleration(&r, &egm96_gravity_model())
        );
    }

    /// The zonal tier is exactly the sum of the three closed-form pieces.
    #[test]
    fn zonal_tier_composes_the_closed_forms() {
        let provider = Iau76Fk5Provider::default();
        let (r, v) = leo_state();
        let model = egm96_gravity_model();
        let accel = gravity_only(GravityFidelity::Zonal {
            j3: EARTH_J3,
            j4: EARTH_J4,
        })
        .build(leo_epoch(), &provider);
        let expected = j2_acceleration(&r, &model)
            + j3_acceleration(&r, &model, EARTH_J3)
            + j4_acceleration(&r, &model, EARTH_J4);
        assert_eq!(accel(0.0, &r, &v), expected);
    }

    // ── Spec: "Forces are individually toggleable" ───────────────────────

    /// Toggling one force off changes the composed acceleration by exactly
    /// that force's contribution (to summation round-off, bounded by ulps
    /// of the dominant central term).
    #[test]
    fn each_force_toggles_in_isolation() {
        let provider = Iau76Fk5Provider::default();
        let epoch = leo_epoch();
        let (r, v) = leo_state();
        let t = 1_234.0;
        let at = epoch + t;
        let full = full_stack();
        let full_accel = full.build(epoch, &provider)(t, &r, &v);
        let scale = full_accel.norm();

        let rot = gcrf_to_itrf(&provider, &at);
        let sun = sun_position_gcrf(&at);
        let cases: [(ForceModelConfig, Vector3<f64>, &str); 4] = [
            (
                ForceModelConfig { srp: None, ..full },
                srp_acceleration(&r, &sun, 0.0065),
                "srp",
            ),
            (
                ForceModelConfig { drag: None, ..full },
                harris_priester_drag(&r, &v, 0.011, &rot, &sun),
                "drag",
            ),
            (
                ForceModelConfig {
                    third_body: ThirdBodyConfig {
                        sun: false,
                        moon: true,
                    },
                    ..full
                },
                third_body_acceleration(&r, &sun, GM_SUN),
                "third-body sun",
            ),
            (
                ForceModelConfig {
                    third_body: ThirdBodyConfig {
                        sun: true,
                        moon: false,
                    },
                    ..full
                },
                third_body_acceleration(&r, &moon_position_gcrf(&at), GM_MOON),
                "third-body moon",
            ),
        ];
        for (without, expected, name) in cases {
            let difference = full_accel - without.build(epoch, &provider)(t, &r, &v);
            assert!(
                (difference - expected).norm() <= 1e-12 * scale,
                "{name}: toggle difference {difference:?} != contribution {expected:?}"
            );
            assert!(expected.norm() > 0.0, "{name}: contribution must be live");
        }
    }

    // ── Spec: "Earth-fixed force legs use the frame provider" ────────────

    /// The harmonic leg is exactly provider-rotation → ITRF evaluation →
    /// inverse rotation at `epoch + t`, and the Earth's rotation between
    /// two `t`s visibly moves the acceleration.
    #[test]
    fn harmonics_leg_mirrors_the_provider_rotation_at_epoch_plus_t() {
        let provider = Iau76Fk5Provider::default();
        let epoch = leo_epoch();
        let (r, v) = leo_state();
        let accel = gravity_only(GravityFidelity::Harmonics {
            degree: 12,
            order: 12,
        })
        .build(epoch, &provider);

        let t = 600.0;
        let rot = provider
            .rotation(Frame::Gcrf, Frame::Itrf, &(epoch + t))
            .unwrap();
        let expected = rot.inverse().rotate_position(&egm96_acceleration_itrf(
            &rot.rotate_position(&r),
            12,
            12,
        ));
        assert_eq!(accel(t, &r, &v), expected);

        // 600 s of Earth rotation changes the tesseral field at fixed GCRF r.
        assert!((accel(0.0, &r, &v) - accel(t, &r, &v)).norm() > 0.0);
    }

    // ── Spec: "Provider parameters shift Earth-fixed forces" ─────────────

    /// A non-zero ΔUT1 provider shifts the harmonic acceleration (rotated
    /// Earth-fixed frame) but leaves the purely inertial two-body tier
    /// untouched.
    #[test]
    fn delta_ut1_shifts_harmonic_but_not_inertial_forces() {
        let default_provider = Iau76Fk5Provider::default();
        let shifted_provider = Iau76Fk5Provider {
            delta_ut1: 0.4,
            polar_motion: None,
        };
        let epoch = leo_epoch();
        let (r, v) = leo_state();

        let harmonics = gravity_only(GravityFidelity::Harmonics {
            degree: 12,
            order: 12,
        });
        let a_default = harmonics.build(epoch, &default_provider)(0.0, &r, &v);
        let a_shifted = harmonics.build(epoch, &shifted_provider)(0.0, &r, &v);
        assert!(
            (a_default - a_shifted).norm() > 1e-12,
            "ΔUT1 = 0.4 s must rotate the Earth-fixed field: delta {}",
            (a_default - a_shifted).norm()
        );

        let two_body = gravity_only(GravityFidelity::TwoBody);
        assert_eq!(
            two_body.build(epoch, &default_provider)(0.0, &r, &v),
            two_body.build(epoch, &shifted_provider)(0.0, &r, &v),
        );
    }

    // ── The composed closure rides the time-aware integrator seam ────────

    /// `build`'s closure feeds `propagate_on_grid` directly (the
    /// `TimeAwareAccel` blanket impl) and produces a sane LEO step.
    #[test]
    fn composed_closure_drives_the_adaptive_integrator() {
        let provider = Iau76Fk5Provider::default();
        let (r, v) = leo_state();
        let accel = ForceModelConfig::default().build(leo_epoch(), &provider);
        let solution = propagate_on_grid(&r, &v, &[60.0], &DpConfig::default(), accel);
        let sample = &solution.samples[0];
        assert_eq!(sample.offset_s, 60.0);
        // One minute of LEO flight moves the spacecraft a few hundred km
        // but keeps the radius near-orbital.
        assert!((sample.position - r).norm() > 1e5);
        assert!((sample.position.norm() - r.norm()).abs() < 5e4);
    }
}
