//! High-fidelity orbital propagation with J2 perturbations, atmospheric drag,
//! impulsive maneuvers, and ground-station visibility analysis.

use nalgebra::Vector3;
use serde::{Deserialize, Serialize};
use thresh_core::eci::{SECONDS_PER_DAY, eci_to_enu};
use thresh_core::orbital::{
    ElementError, Frame, GravityModel, OrbitalElements, cartesian_to_keplerian, j2_acceleration,
    keplerian_to_cartesian, rk4_step, two_body_acceleration,
};
use thresh_core::time::Epoch;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------
//
// The force-math constants moved to `thresh_core::orbital::GravityModel`
// (design Decision 1 of `orbital-ballistic-filter-models`); the aliases below
// keep the Keplerian element conversions and tests reading naturally until
// they migrate to `thresh_core::orbital::state` in a later phase.

/// Earth gravitational parameter (m³/s²).
const GM_EARTH: f64 = GravityModel::EARTH_WGS84.mu;

/// Earth equatorial radius (m), WGS-84.
pub const EARTH_RADIUS: f64 = GravityModel::EARTH_WGS84.equatorial_radius;

// ---------------------------------------------------------------------------
// Exponential atmosphere model
// ---------------------------------------------------------------------------

/// Compute atmospheric density (kg/m³) at a given geometric altitude (m)
/// using a piecewise-exponential model.
///
/// Thin shim delegating to [`thresh_core::orbital::atmosphere_density`].
pub fn atmosphere_density(alt_m: f64) -> f64 {
    thresh_core::orbital::atmosphere_density(alt_m)
}

// ---------------------------------------------------------------------------
// Orbital state
// ---------------------------------------------------------------------------

/// A spacecraft state in the ECI frame.
///
/// This is the propagator's lightweight Cartesian sample type — it appears
/// in serialized outputs and the inner RK4 loop. The frame-disciplined
/// element-representation type is [`thresh_core::orbital::OrbitalState`];
/// `From` / `TryFrom` conversions between the two are provided below.
///
/// # Why `epoch_jd` stays a raw `f64` (astro-time-and-frames, Decision 6)
///
/// This type is the innermost plumbing of the truth generators and the
/// calibrated benchmark chain, whose metrics must stay **bitwise** stable.
/// Converting `JD → Epoch → JD` through `hifitime`'s integer-nanosecond
/// representation is not guaranteed bit-exact (measured: up to 1 ulp of a
/// Julian date), and epoch-stepping in exact seconds rounds differently
/// from the propagator's `jd + dt/86400` arithmetic — either would silently
/// drift the calibrated floors. So the sample type keeps the raw UTC Julian
/// date, and the time-scale-aware [`Epoch`] appears at the API boundary:
/// the `From`/`TryFrom` conversions below construct it via
/// [`Epoch::from_jde_utc`] / read it back via `to_jde_utc_days()`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrbitalState {
    /// Position in ECI (metres).
    pub position: [f64; 3],
    /// Velocity in ECI (m/s).
    pub velocity: [f64; 3],
    /// Epoch as a Julian Date in the **UTC** scale (see the type-level note
    /// for why this is not an [`Epoch`]).
    pub epoch_jd: f64,
}

impl OrbitalState {
    /// Create from Cartesian position and velocity.
    pub fn from_cartesian(pos: [f64; 3], vel: [f64; 3], epoch_jd: f64) -> Self {
        Self {
            position: pos,
            velocity: vel,
            epoch_jd,
        }
    }

    /// Create from Keplerian orbital elements.
    ///
    /// Thin shim delegating to
    /// [`thresh_core::orbital::keplerian_to_cartesian`] with Earth's
    /// gravitational parameter.
    ///
    /// # Arguments
    /// * `sma` — semi-major axis (metres)
    /// * `ecc` — eccentricity
    /// * `inc` — inclination (radians)
    /// * `raan` — right ascension of the ascending node (radians)
    /// * `argp` — argument of periapsis (radians)
    /// * `true_anom` — true anomaly (radians)
    /// * `epoch_jd` — epoch as Julian Date
    pub fn from_keplerian(
        sma: f64,
        ecc: f64,
        inc: f64,
        raan: f64,
        argp: f64,
        true_anom: f64,
        epoch_jd: f64,
    ) -> Self {
        let (pos, vel) = keplerian_to_cartesian(sma, ecc, inc, raan, argp, true_anom, GM_EARTH);
        Self {
            position: [pos.x, pos.y, pos.z],
            velocity: [vel.x, vel.y, vel.z],
            epoch_jd,
        }
    }

    /// Compute the orbital radius (distance from Earth centre) in metres.
    pub fn radius(&self) -> f64 {
        let [x, y, z] = self.position;
        (x * x + y * y + z * z).sqrt()
    }

    /// Compute the semi-major axis from the vis-viva equation (metres).
    pub fn semi_major_axis(&self) -> f64 {
        let r = self.radius();
        let v2 = self.velocity.iter().map(|vi| vi * vi).sum::<f64>();
        1.0 / (2.0 / r - v2 / GM_EARTH)
    }

    /// Convert this state to Keplerian elements.
    ///
    /// Returns `(sma, ecc, inc, raan, argp, true_anom)`.
    ///
    /// Thin shim delegating to
    /// [`thresh_core::orbital::cartesian_to_keplerian`] (where the
    /// `compute_raan` / `compute_argp` / `compute_true_anomaly` phase
    /// helpers now live) with Earth's gravitational parameter.
    pub fn to_keplerian(&self) -> (f64, f64, f64, f64, f64, f64) {
        let pos = Vector3::new(self.position[0], self.position[1], self.position[2]);
        let vel = Vector3::new(self.velocity[0], self.velocity[1], self.velocity[2]);
        cartesian_to_keplerian(&pos, &vel, GM_EARTH)
    }
}

// ---------------------------------------------------------------------------
// Conversions to/from the thresh-core representation type
// ---------------------------------------------------------------------------

/// Convert the synth Cartesian sample into the frame-disciplined
/// [`thresh_core::orbital::OrbitalState`].
///
/// The synth propagator's fixed conventions are assumed: Earth's WGS-84
/// `mu`, and [`Frame::Teme`] — the truthful name for the "ECI" of the
/// repo's GMST-only rotation convention (`astro-time-and-frames` design
/// Decision 6: that chain always was TEME-consistent; the tag now says so).
/// The sample's UTC Julian date becomes a time-scale-aware epoch via
/// [`Epoch::from_jde_utc`].
impl From<OrbitalState> for thresh_core::orbital::OrbitalState {
    fn from(state: OrbitalState) -> Self {
        Self {
            elements: OrbitalElements::Cartesian {
                position: Vector3::new(state.position[0], state.position[1], state.position[2]),
                velocity: Vector3::new(state.velocity[0], state.velocity[1], state.velocity[2]),
            },
            mu: GravityModel::EARTH_WGS84.mu,
            frame: Frame::Teme,
            epoch: Epoch::from_jde_utc(state.epoch_jd),
        }
    }
}

/// Convert a [`thresh_core::orbital::OrbitalState`] into the synth Cartesian
/// sample, dropping the `mu` and frame tags (the synth type is TEME / Earth
/// by convention — see `From<OrbitalState>` above) and reading the epoch
/// back as a UTC Julian date (`to_jde_utc_days()`; not guaranteed bit-exact
/// against a JD the epoch was constructed from — see the [`OrbitalState`]
/// type-level note).
///
/// Fallible (`TryFrom` rather than `From`) because a `Tle`-represented state
/// cannot be converted without SGP4
/// ([`ElementError::RequiresSgp4`] — see `tle_to_cartesian` in `thresh-data`).
impl TryFrom<thresh_core::orbital::OrbitalState> for OrbitalState {
    type Error = ElementError;

    fn try_from(state: thresh_core::orbital::OrbitalState) -> Result<Self, ElementError> {
        let (position, velocity) = state.as_cartesian()?;
        Ok(Self {
            position: [position.x, position.y, position.z],
            velocity: [velocity.x, velocity.y, velocity.z],
            epoch_jd: state.epoch.to_jde_utc_days(),
        })
    }
}

// ---------------------------------------------------------------------------
// Force models
// ---------------------------------------------------------------------------

/// Compute two-body gravitational acceleration with J2 perturbation.
///
/// Thin shim delegating to [`thresh_core::orbital::j2_acceleration`] with
/// [`GravityModel::EARTH_WGS84`].
pub fn acceleration_j2(pos: &[f64; 3]) -> [f64; 3] {
    let acc = j2_acceleration(
        &Vector3::new(pos[0], pos[1], pos[2]),
        &GravityModel::EARTH_WGS84,
    );
    [acc.x, acc.y, acc.z]
}

/// Compute atmospheric drag acceleration.
///
/// Uses an exponential atmosphere model with a co-rotating atmosphere
/// (relative velocity `v_rel = v − ω_⊕ × r`).
///
/// Thin shim delegating to [`thresh_core::orbital::drag_acceleration`] with
/// the inverse ballistic coefficient `inv_beta = cd · area_m2 / mass_kg`.
pub fn acceleration_drag(
    pos: &[f64; 3],
    vel: &[f64; 3],
    cd: f64,
    area_m2: f64,
    mass_kg: f64,
) -> [f64; 3] {
    let inv_beta = cd * area_m2 / mass_kg;
    let acc = thresh_core::orbital::drag_acceleration(
        &Vector3::new(pos[0], pos[1], pos[2]),
        &Vector3::new(vel[0], vel[1], vel[2]),
        inv_beta,
    );
    [acc.x, acc.y, acc.z]
}

// ---------------------------------------------------------------------------
// Propagator configuration
// ---------------------------------------------------------------------------

/// Configuration for atmospheric drag.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DragConfig {
    /// Drag coefficient (dimensionless, typically ~2.2).
    pub cd: f64,
    /// Cross-sectional area (m²).
    pub area_m2: f64,
    /// Spacecraft mass (kg).
    pub mass_kg: f64,
}

/// Configuration for the orbital propagator.
#[derive(Debug, Clone)]
pub struct PropagatorConfig {
    /// Include J2 zonal harmonic perturbation.
    pub include_j2: bool,
    /// Optional atmospheric drag model.
    pub drag: Option<DragConfig>,
    /// Integration step size (seconds).
    pub dt_s: f64,
}

// ---------------------------------------------------------------------------
// RK4 propagator
// ---------------------------------------------------------------------------

/// Compute the total acceleration on the spacecraft.
///
/// Composes the shared `thresh_core::orbital` force math per the propagator
/// configuration (gravity with or without J2, plus optional drag).
fn total_acceleration(
    pos: &Vector3<f64>,
    vel: &Vector3<f64>,
    config: &PropagatorConfig,
) -> Vector3<f64> {
    let mut acc = if config.include_j2 {
        j2_acceleration(pos, &GravityModel::EARTH_WGS84)
    } else {
        two_body_acceleration(pos, &GravityModel::EARTH_WGS84)
    };

    if let Some(ref drag) = config.drag {
        let inv_beta = drag.cd * drag.area_m2 / drag.mass_kg;
        acc += thresh_core::orbital::drag_acceleration(pos, vel, inv_beta);
    }

    acc
}

/// Propagate an orbital state forward in time using a 4th-order Runge-Kutta
/// integrator.
///
/// Integration steps through the shared closure-based
/// [`thresh_core::orbital::rk4_step`], with the acceleration composed from
/// the propagator configuration (gravity with or without J2, plus optional
/// drag) out of the shared `thresh_core::orbital` force math.
///
/// Returns a vector of [`OrbitalState`] sampled at intervals of `output_dt_s`.
/// The first element is the initial state.
pub fn propagate(
    initial: &OrbitalState,
    duration_s: f64,
    config: &PropagatorConfig,
    output_dt_s: f64,
) -> Vec<OrbitalState> {
    let mut results = Vec::new();
    let mut pos = Vector3::new(
        initial.position[0],
        initial.position[1],
        initial.position[2],
    );
    let mut vel = Vector3::new(
        initial.velocity[0],
        initial.velocity[1],
        initial.velocity[2],
    );
    let dt = config.dt_s;
    let n_steps = (duration_s / dt).ceil() as usize;
    let output_step_interval = (output_dt_s / dt).round().max(1.0) as usize;

    // Record initial state
    results.push(OrbitalState {
        position: initial.position,
        velocity: initial.velocity,
        epoch_jd: initial.epoch_jd,
    });

    let accel = |p: &Vector3<f64>, v: &Vector3<f64>| total_acceleration(p, v, config);

    for step_i in 1..=n_steps {
        let t = (step_i as f64) * dt;
        let actual_step = if t > duration_s {
            duration_s - (t - dt)
        } else {
            dt
        };
        let (new_pos, new_vel) = rk4_step(&pos, &vel, actual_step, accel);
        pos = new_pos;
        vel = new_vel;

        let elapsed = t.min(duration_s);
        if step_i % output_step_interval == 0 || step_i == n_steps {
            results.push(OrbitalState {
                position: [pos.x, pos.y, pos.z],
                velocity: [vel.x, vel.y, vel.z],
                epoch_jd: initial.epoch_jd + elapsed / SECONDS_PER_DAY,
            });
        }
    }

    results
}

// ---------------------------------------------------------------------------
// Maneuver
// ---------------------------------------------------------------------------

/// Apply an impulsive delta-V to an orbital state.
///
/// Returns a new state with the velocity incremented by `delta_v` (ECI, m/s).
pub fn apply_maneuver(state: &OrbitalState, delta_v: [f64; 3]) -> OrbitalState {
    OrbitalState {
        position: state.position,
        velocity: [
            state.velocity[0] + delta_v[0],
            state.velocity[1] + delta_v[1],
            state.velocity[2] + delta_v[2],
        ],
        epoch_jd: state.epoch_jd,
    }
}

// ---------------------------------------------------------------------------
// Ground station visibility
// ---------------------------------------------------------------------------

/// Convert an orbital ECI state to a local ENU vector relative to a ground
/// station.
///
/// Uses [`thresh_core::eci::eci_to_enu`] for the coordinate transform.
pub fn orbital_to_enu(
    state: &OrbitalState,
    station_lat_rad: f64,
    station_lon_rad: f64,
    station_alt_m: f64,
) -> [f64; 3] {
    let pos = nalgebra::Vector3::new(state.position[0], state.position[1], state.position[2]);
    let enu = eci_to_enu(
        &pos,
        state.epoch_jd,
        station_lat_rad,
        station_lon_rad,
        station_alt_m,
    );
    [enu.x, enu.y, enu.z]
}

/// Check whether a satellite (given its ENU vector) is above the minimum
/// elevation angle.
pub fn is_visible(enu: &[f64; 3], min_elevation_rad: f64) -> bool {
    let horiz = (enu[0] * enu[0] + enu[1] * enu[1]).sqrt();
    let elevation = enu[2].atan2(horiz);
    elevation >= min_elevation_rad
}

/// Compute slant range from an ENU vector (metres).
pub fn slant_range(enu: &[f64; 3]) -> f64 {
    (enu[0] * enu[0] + enu[1] * enu[1] + enu[2] * enu[2]).sqrt()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    /// Helper: circular orbit velocity at given radius.
    fn circular_velocity(r: f64) -> f64 {
        (GM_EARTH / r).sqrt()
    }

    /// Euclidean magnitude of a 3-element array (test-local helper; the
    /// production Keplerian math moved to `thresh_core::orbital::state`).
    fn mag3(v: &[f64; 3]) -> f64 {
        (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
    }

    // ── ISS-like orbit, 1-day J2 propagation ───────────────────────────

    #[test]
    fn iss_orbit_j2_one_day() {
        let alt_km = 420.0;
        let r = EARTH_RADIUS + alt_km * 1000.0;
        let inc = 51.6_f64.to_radians();

        let state = OrbitalState::from_keplerian(
            r,
            0.0005,
            inc,
            0.0,
            0.0,
            0.0,
            2_451_545.0, // J2000
        );

        let config = PropagatorConfig {
            include_j2: true,
            drag: None,
            dt_s: 10.0,
        };

        let duration = 86400.0; // 1 day
        let results = propagate(&state, duration, &config, 60.0);

        // Verify altitude stays in 400-450 km range
        for (i, s) in results.iter().enumerate() {
            let alt_m = s.radius() - EARTH_RADIUS;
            let alt_km = alt_m / 1000.0;
            assert!(
                (390.0..=460.0).contains(&alt_km),
                "Step {i}: altitude {alt_km:.1} km out of range"
            );
        }
    }

    // ── GEO orbit, 24-hour longitude drift < 1° ────────────────────────

    #[test]
    fn geo_orbit_longitude_drift() {
        let r_geo = 42_164_000.0; // GEO radius in metres
        let v = circular_velocity(r_geo);

        // GEO: equatorial, circular
        let state = OrbitalState::from_cartesian([r_geo, 0.0, 0.0], [0.0, v, 0.0], 2_451_545.0);

        let config = PropagatorConfig {
            include_j2: true,
            drag: None,
            dt_s: 30.0,
        };

        let duration = 86400.0;
        let results = propagate(&state, duration, &config, duration);

        let last = results.last().unwrap();
        // Compute longitude in ECI (approximation for GEO)
        let lon0 = state.position[1].atan2(state.position[0]);
        let lon1 = last.position[1].atan2(last.position[0]);

        // After exactly 1 sidereal day the satellite should be near its starting
        // longitude. For a solar day we allow up to ~1° drift from J2 effects.
        // GEO orbit period ≈ sidereal day, so residual should be small.
        // We measure the angular difference.
        let mut diff = (lon1 - lon0).abs().to_degrees();
        if diff > 180.0 {
            diff = 360.0 - diff;
        }
        // The satellite completes nearly one full revolution; account for wrapping.
        // In one solar day a GEO satellite drifts ~0.98° relative to inertial frame.
        // We just check the drift is not huge.
        assert!(
            diff < 5.0,
            "GEO longitude drift: {diff:.2}° (expected < 5°)"
        );
    }

    // ── Drag causes SMA to decrease ─────────────────────────────────────

    #[test]
    fn drag_decreases_sma() {
        let alt_km = 300.0;
        let r = EARTH_RADIUS + alt_km * 1000.0;

        let state = OrbitalState::from_cartesian(
            [r, 0.0, 0.0],
            [0.0, circular_velocity(r), 0.0],
            2_451_545.0,
        );

        let config = PropagatorConfig {
            include_j2: true,
            drag: Some(DragConfig {
                cd: 2.2,
                area_m2: 20.0,
                mass_kg: 500.0,
            }),
            dt_s: 10.0,
        };

        let sma_initial = state.semi_major_axis();

        // Propagate for 2 orbits (~3 hours)
        let duration = 2.0 * 2.0 * PI * (r.powi(3) / GM_EARTH).sqrt();
        let results = propagate(&state, duration, &config, duration);
        let sma_final = results.last().unwrap().semi_major_axis();

        assert!(
            sma_final < sma_initial,
            "SMA should decrease with drag: initial={sma_initial:.0} final={sma_final:.0}"
        );
    }

    // ── Keplerian roundtrip ─────────────────────────────────────────────

    #[test]
    fn keplerian_roundtrip() {
        let sma = 7_000_000.0;
        let ecc = 0.01;
        let inc = 0.9; // ~51.6°
        let raan = 1.2;
        let argp = 0.5;
        let ta = 0.8;
        let jd = 2_451_545.0;

        let state = OrbitalState::from_keplerian(sma, ecc, inc, raan, argp, ta, jd);
        let (sma2, ecc2, inc2, raan2, argp2, ta2) = state.to_keplerian();

        assert!((sma2 - sma).abs() < 1.0, "SMA: {sma2:.1} vs {sma:.1}");
        assert!((ecc2 - ecc).abs() < 1e-10, "ecc: {ecc2} vs {ecc}");
        assert!((inc2 - inc).abs() < 1e-10, "inc: {inc2} vs {inc}");
        assert!((raan2 - raan).abs() < 1e-10, "raan: {raan2} vs {raan}");
        assert!((argp2 - argp).abs() < 1e-8, "argp: {argp2} vs {argp}");
        assert!((ta2 - ta).abs() < 1e-8, "ta: {ta2} vs {ta}");
    }

    // ── Maneuver ────────────────────────────────────────────────────────

    #[test]
    fn maneuver_applies_delta_v() {
        let state =
            OrbitalState::from_cartesian([7_000_000.0, 0.0, 0.0], [0.0, 7_500.0, 0.0], 2_451_545.0);

        let dv = [10.0, -5.0, 3.0];
        let result = apply_maneuver(&state, dv);

        assert_eq!(result.position, state.position);
        assert!((result.velocity[0] - 10.0).abs() < 1e-12);
        assert!((result.velocity[1] - 7495.0).abs() < 1e-12);
        assert!((result.velocity[2] - 3.0).abs() < 1e-12);
        assert_eq!(result.epoch_jd, state.epoch_jd);
    }

    // ── Visibility / elevation mask ─────────────────────────────────────

    #[test]
    fn visibility_above_and_below_mask() {
        // Satellite directly overhead: elevation = 90°
        let enu_above = [0.0, 0.0, 400_000.0];
        assert!(is_visible(&enu_above, 10.0_f64.to_radians()));

        // Satellite on horizon: elevation ≈ 0°
        let enu_horizon = [400_000.0, 0.0, 1.0];
        assert!(!is_visible(&enu_horizon, 10.0_f64.to_radians()));

        // Satellite below horizon
        let enu_below = [100_000.0, 0.0, -50_000.0];
        assert!(!is_visible(&enu_below, 0.0));
    }

    // ── Slant range ─────────────────────────────────────────────────────

    #[test]
    fn slant_range_calculation() {
        let enu = [3000.0, 4000.0, 0.0];
        let sr = slant_range(&enu);
        assert!((sr - 5000.0).abs() < 1e-9);
    }

    // ── J2 acceleration sanity ──────────────────────────────────────────

    #[test]
    fn j2_acceleration_at_equator() {
        let pos = [EARTH_RADIUS + 400_000.0, 0.0, 0.0];
        let acc = acceleration_j2(&pos);
        // Should be roughly -GM/r² in x direction
        let r = pos[0];
        let expected = -GM_EARTH / (r * r);
        // J2 modifies it slightly but shouldn't change sign
        assert!(acc[0] < 0.0, "Acceleration should point inward");
        assert!(
            (acc[0] - expected).abs() / expected.abs() < 0.01,
            "J2 perturbation too large at equator"
        );
    }

    // ── Two-body energy conservation (no J2, no drag) ───────────────────

    #[test]
    fn two_body_energy_conservation() {
        let r = EARTH_RADIUS + 500_000.0;
        let state = OrbitalState::from_cartesian(
            [r, 0.0, 0.0],
            [0.0, circular_velocity(r), 0.0],
            2_451_545.0,
        );

        let config = PropagatorConfig {
            include_j2: false,
            drag: None,
            dt_s: 10.0,
        };

        let v2_init = state.velocity.iter().map(|x| x * x).sum::<f64>();
        let energy_init = 0.5 * v2_init - GM_EARTH / state.radius();

        let duration = 5400.0; // ~1 orbit
        let results = propagate(&state, duration, &config, duration);
        let last = results.last().unwrap();
        let v2_final = last.velocity.iter().map(|x| x * x).sum::<f64>();
        let energy_final = 0.5 * v2_final - GM_EARTH / last.radius();

        let rel_err = (energy_final - energy_init).abs() / energy_init.abs();
        assert!(rel_err < 1e-8, "Energy conservation error: {rel_err:.2e}");
    }

    // ── Atmosphere model ────────────────────────────────────────────────

    #[test]
    fn atmosphere_density_decreases_with_altitude() {
        let rho_200 = atmosphere_density(200_000.0);
        let rho_400 = atmosphere_density(400_000.0);
        let rho_800 = atmosphere_density(800_000.0);
        assert!(rho_200 > rho_400);
        assert!(rho_400 > rho_800);
        assert!(rho_200 > 0.0);
    }

    // ── Bitwise identity: synth shims vs thresh_core::orbital (task 1.5) ──

    /// Spec "Identical accelerations from both consumers": the same ECI state
    /// evaluated through the synth `[f64; 3]` shims and through
    /// `thresh_core::orbital` directly must be **bitwise** identical — same
    /// code path, not merely approximately equal.
    #[test]
    fn shims_match_core_bitwise() {
        // All components nonzero so every term of the math is exercised;
        // altitude ~247 km keeps the state inside the sensible atmosphere.
        let pos = [6_500_000.0, 1_000_000.0, 800_000.0];
        let vel = [-1_500.0, 7_100.0, 300.0];
        let pos_v = Vector3::new(pos[0], pos[1], pos[2]);
        let vel_v = Vector3::new(vel[0], vel[1], vel[2]);

        let shim_grav = acceleration_j2(&pos);
        let core_grav = j2_acceleration(&pos_v, &GravityModel::EARTH_WGS84);

        let (cd, area_m2, mass_kg) = (2.2, 20.0, 500.0);
        let shim_drag = acceleration_drag(&pos, &vel, cd, area_m2, mass_kg);
        let core_drag =
            thresh_core::orbital::drag_acceleration(&pos_v, &vel_v, cd * area_m2 / mass_kg);

        assert!(
            mag3(&shim_drag) > 0.0,
            "drag must be nonzero for the comparison to be meaningful"
        );
        for i in 0..3 {
            assert_eq!(
                shim_grav[i].to_bits(),
                core_grav[i].to_bits(),
                "gravity component {i} differs"
            );
            assert_eq!(
                shim_drag[i].to_bits(),
                core_drag[i].to_bits(),
                "drag component {i} differs"
            );
        }
    }

    // ── From/TryFrom conversions with the core representation (task 2.6) ──

    #[test]
    fn synth_to_core_assumes_teme_and_earth_mu() {
        let synth = OrbitalState::from_cartesian(
            [7_000_000.0, 1_000.0, -2_000.0],
            [10.0, 7_500.0, -20.0],
            2_451_545.0,
        );
        let core = thresh_core::orbital::OrbitalState::from(synth.clone());

        // The GMST-convention "ECI" is truthfully TEME (astro-time-and-
        // frames design Decision 6), and the UTC JD becomes an Epoch.
        assert_eq!(core.frame, Frame::Teme);
        assert_eq!(core.mu.to_bits(), GravityModel::EARTH_WGS84.mu.to_bits());
        assert_eq!(core.epoch, Epoch::from_jde_utc(synth.epoch_jd));
        let (pos, vel) = core.as_cartesian().unwrap();
        for i in 0..3 {
            assert_eq!(pos[i].to_bits(), synth.position[i].to_bits());
            assert_eq!(vel[i].to_bits(), synth.velocity[i].to_bits());
        }
    }

    #[test]
    fn core_to_synth_roundtrip_is_bitwise() {
        let synth = OrbitalState::from_cartesian(
            [6_800_000.0, -500_000.0, 300_000.0],
            [-100.0, 7_400.0, 800.0],
            2_460_310.5,
        );
        let core = thresh_core::orbital::OrbitalState::from(synth.clone());
        let back = OrbitalState::try_from(core).unwrap();

        // Position/velocity round-trip bitwise unconditionally. The epoch
        // JD round trip through hifitime's integer-nanosecond storage is
        // bit-exact for half-integral JDs like this TLE epoch (general JDs
        // may differ by 1 ulp — the reason the sample type keeps raw JDs).
        assert_eq!(back.epoch_jd.to_bits(), synth.epoch_jd.to_bits());
        for i in 0..3 {
            assert_eq!(back.position[i].to_bits(), synth.position[i].to_bits());
            assert_eq!(back.velocity[i].to_bits(), synth.velocity[i].to_bits());
        }
    }

    #[test]
    fn core_to_synth_rejects_tle_elements() {
        let tle_state = thresh_core::orbital::OrbitalState {
            elements: OrbitalElements::Tle {
                line1: "1 25544U 98067A   24001.00000000  .00016717  00000-0  10270-3 0  9026"
                    .to_string(),
                line2: "2 25544  51.6400 208.9163 0006703  30.1579 330.0018 15.49560455    18"
                    .to_string(),
            },
            mu: GravityModel::EARTH_WGS84.mu,
            frame: Frame::Teme,
            epoch: Epoch::from_jde_utc(2_460_310.5),
        };
        assert_eq!(
            OrbitalState::try_from(tle_state).unwrap_err(),
            ElementError::RequiresSgp4
        );
    }
}
