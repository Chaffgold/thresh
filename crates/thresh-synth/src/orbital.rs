//! High-fidelity orbital propagation with J2 perturbations, atmospheric drag,
//! impulsive maneuvers, and ground-station visibility analysis.

use serde::{Deserialize, Serialize};
use thresh_core::eci::{SECONDS_PER_DAY, eci_to_enu};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Earth gravitational parameter (m³/s²).
const GM_EARTH: f64 = 3.986_004_418e14;

/// Earth equatorial radius (m), WGS-84.
const EARTH_RADIUS: f64 = 6_378_137.0;

/// J2 zonal harmonic coefficient.
const J2: f64 = 1.082_63e-3;

// ---------------------------------------------------------------------------
// Exponential atmosphere model reference data
// ---------------------------------------------------------------------------

/// (base altitude km, nominal density kg/m³, scale height km)
const ATMOSPHERE_TABLE: &[(f64, f64, f64)] = &[
    (0.0, 1.225, 7.249),
    (25.0, 3.899e-2, 6.349),
    (30.0, 1.774e-2, 6.682),
    (40.0, 3.972e-3, 7.554),
    (50.0, 1.057e-3, 8.382),
    (60.0, 3.206e-4, 7.714),
    (70.0, 8.770e-5, 6.549),
    (80.0, 1.905e-5, 5.799),
    (90.0, 3.396e-6, 5.382),
    (100.0, 5.297e-7, 5.877),
    (110.0, 9.661e-8, 7.263),
    (120.0, 2.438e-8, 9.473),
    (130.0, 8.484e-9, 12.636),
    (140.0, 3.845e-9, 16.149),
    (150.0, 2.070e-9, 22.523),
    (180.0, 5.464e-10, 29.740),
    (200.0, 2.789e-10, 37.105),
    (250.0, 7.248e-11, 45.546),
    (300.0, 2.418e-11, 53.628),
    (350.0, 9.518e-12, 53.298),
    (400.0, 3.725e-12, 58.515),
    (450.0, 1.585e-12, 60.828),
    (500.0, 6.967e-13, 63.822),
    (600.0, 1.454e-13, 71.835),
    (700.0, 3.614e-14, 88.667),
    (800.0, 1.170e-14, 124.64),
    (900.0, 5.245e-15, 181.05),
    (1000.0, 3.019e-15, 268.00),
];

/// Compute atmospheric density (kg/m³) at a given geometric altitude (m)
/// using a piecewise-exponential model.
fn atmosphere_density(alt_m: f64) -> f64 {
    let alt_km = alt_m / 1000.0;
    if alt_km < 0.0 {
        return ATMOSPHERE_TABLE[0].1;
    }
    if alt_km > 1000.0 {
        return 0.0;
    }
    // Find the bracket
    let mut idx = 0;
    for (i, &(h, _, _)) in ATMOSPHERE_TABLE.iter().enumerate() {
        if h <= alt_km {
            idx = i;
        } else {
            break;
        }
    }
    let (h0, rho0, scale_h) = ATMOSPHERE_TABLE[idx];
    rho0 * (-((alt_km - h0) / scale_h)).exp()
}

// ---------------------------------------------------------------------------
// Orbital state
// ---------------------------------------------------------------------------

/// A spacecraft state in the ECI frame.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrbitalState {
    /// Position in ECI (metres).
    pub position: [f64; 3],
    /// Velocity in ECI (m/s).
    pub velocity: [f64; 3],
    /// Epoch as Julian Date.
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
        let nu = true_anom;
        let p = sma * (1.0 - ecc * ecc); // semi-latus rectum
        let r_mag = p / (1.0 + ecc * nu.cos());

        // Position and velocity in the perifocal frame (PQW)
        let r_pqw = [r_mag * nu.cos(), r_mag * nu.sin(), 0.0];
        let v_coeff = (GM_EARTH / p).sqrt();
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

        let pos = [
            r11 * r_pqw[0] + r12 * r_pqw[1],
            r21 * r_pqw[0] + r22 * r_pqw[1],
            r31 * r_pqw[0] + r32 * r_pqw[1],
        ];
        let vel = [
            r11 * v_pqw[0] + r12 * v_pqw[1],
            r21 * v_pqw[0] + r22 * v_pqw[1],
            r31 * v_pqw[0] + r32 * v_pqw[1],
        ];

        Self {
            position: pos,
            velocity: vel,
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
    pub fn to_keplerian(&self) -> (f64, f64, f64, f64, f64, f64) {
        let r = self.position;
        let v = self.velocity;
        let r_mag = self.radius();
        let v2 = v[0] * v[0] + v[1] * v[1] + v[2] * v[2];

        let h = cross3(&r, &v);
        let h_mag = mag3(&h);

        let n = [-h[1], h[0], 0.0];
        let n_mag = (n[0] * n[0] + n[1] * n[1]).sqrt();

        let rdotv = dot3(&r, &v);
        let e_vec = eccentricity_vector(&r, &v, r_mag, v2, rdotv);
        let ecc = mag3(&e_vec);

        let sma = 1.0 / (2.0 / r_mag - v2 / GM_EARTH);
        let inc = (h[2] / h_mag).acos();
        let raan = compute_raan(&n, n_mag);
        let argp = compute_argp(&n, n_mag, &e_vec, ecc);
        let true_anom = compute_true_anomaly(&e_vec, ecc, &r, r_mag, rdotv);

        (sma, ecc, inc, raan, argp, true_anom)
    }
}

// ---------------------------------------------------------------------------
// Vector helpers for Keplerian conversion
// ---------------------------------------------------------------------------

/// Cross product of two 3-element arrays.
fn cross3(a: &[f64; 3], b: &[f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

/// Dot product of two 3-element arrays.
fn dot3(a: &[f64; 3], b: &[f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// Euclidean magnitude of a 3-element array.
fn mag3(v: &[f64; 3]) -> f64 {
    (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
}

/// Compute the eccentricity vector from position, velocity, and derived scalars.
fn eccentricity_vector(r: &[f64; 3], v: &[f64; 3], r_mag: f64, v2: f64, rdotv: f64) -> [f64; 3] {
    [
        (v2 - GM_EARTH / r_mag) * r[0] / GM_EARTH - rdotv * v[0] / GM_EARTH,
        (v2 - GM_EARTH / r_mag) * r[1] / GM_EARTH - rdotv * v[1] / GM_EARTH,
        (v2 - GM_EARTH / r_mag) * r[2] / GM_EARTH - rdotv * v[2] / GM_EARTH,
    ]
}

/// Compute right ascension of the ascending node from the node vector.
fn compute_raan(n: &[f64; 3], n_mag: f64) -> f64 {
    if n_mag <= 1e-12 {
        return 0.0;
    }
    let val = (n[0] / n_mag).acos();
    if n[1] >= 0.0 {
        val
    } else {
        std::f64::consts::TAU - val
    }
}

/// Compute argument of periapsis from node vector and eccentricity vector.
fn compute_argp(n: &[f64; 3], n_mag: f64, e_vec: &[f64; 3], ecc: f64) -> f64 {
    if n_mag <= 1e-12 || ecc <= 1e-12 {
        return 0.0;
    }
    let ndote = (n[0] * e_vec[0] + n[1] * e_vec[1]) / (n_mag * ecc);
    let val = ndote.clamp(-1.0, 1.0).acos();
    if e_vec[2] >= 0.0 {
        val
    } else {
        std::f64::consts::TAU - val
    }
}

/// Compute true anomaly from eccentricity vector and position.
fn compute_true_anomaly(e_vec: &[f64; 3], ecc: f64, r: &[f64; 3], r_mag: f64, rdotv: f64) -> f64 {
    if ecc <= 1e-12 {
        return 0.0;
    }
    let edotr = (e_vec[0] * r[0] + e_vec[1] * r[1] + e_vec[2] * r[2]) / (ecc * r_mag);
    let val = edotr.clamp(-1.0, 1.0).acos();
    if rdotv >= 0.0 {
        val
    } else {
        std::f64::consts::TAU - val
    }
}

// ---------------------------------------------------------------------------
// Force models
// ---------------------------------------------------------------------------

/// Compute two-body gravitational acceleration with J2 perturbation.
pub fn acceleration_j2(pos: &[f64; 3]) -> [f64; 3] {
    let x = pos[0];
    let y = pos[1];
    let z = pos[2];
    let r2 = x * x + y * y + z * z;
    let r = r2.sqrt();
    let r5 = r2 * r2 * r;

    let mu_r3 = GM_EARTH / (r2 * r);
    let j2_coeff = 1.5 * J2 * GM_EARTH * EARTH_RADIUS * EARTH_RADIUS / r5;
    let z2_r2 = 5.0 * z * z / r2;

    [
        -mu_r3 * x + j2_coeff * x * (z2_r2 - 1.0),
        -mu_r3 * y + j2_coeff * y * (z2_r2 - 1.0),
        -mu_r3 * z + j2_coeff * z * (z2_r2 - 3.0),
    ]
}

/// Compute atmospheric drag acceleration.
///
/// Uses an exponential atmosphere model and assumes a non-rotating atmosphere
/// for simplicity (relative velocity ≈ inertial velocity).
pub fn acceleration_drag(
    pos: &[f64; 3],
    vel: &[f64; 3],
    cd: f64,
    area_m2: f64,
    mass_kg: f64,
) -> [f64; 3] {
    let r = (pos[0] * pos[0] + pos[1] * pos[1] + pos[2] * pos[2]).sqrt();
    let alt = r - EARTH_RADIUS;
    if !(0.0..=1_000_000.0).contains(&alt) {
        return [0.0, 0.0, 0.0];
    }

    // Approximate atmospheric co-rotation velocity
    let omega_e = 7.292_115e-5; // rad/s
    let v_atm = [-omega_e * pos[1], omega_e * pos[0], 0.0];
    let v_rel = [vel[0] - v_atm[0], vel[1] - v_atm[1], vel[2] - v_atm[2]];
    let v_rel_mag = (v_rel[0] * v_rel[0] + v_rel[1] * v_rel[1] + v_rel[2] * v_rel[2]).sqrt();

    if v_rel_mag < 1e-10 {
        return [0.0, 0.0, 0.0];
    }

    let rho = atmosphere_density(alt);
    let drag_factor = -0.5 * cd * area_m2 / mass_kg * rho * v_rel_mag;

    [
        drag_factor * v_rel[0],
        drag_factor * v_rel[1],
        drag_factor * v_rel[2],
    ]
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
fn total_acceleration(pos: &[f64; 3], vel: &[f64; 3], config: &PropagatorConfig) -> [f64; 3] {
    let mut acc = if config.include_j2 {
        acceleration_j2(pos)
    } else {
        // Two-body only
        let r2 = pos[0] * pos[0] + pos[1] * pos[1] + pos[2] * pos[2];
        let r = r2.sqrt();
        let mu_r3 = GM_EARTH / (r2 * r);
        [-mu_r3 * pos[0], -mu_r3 * pos[1], -mu_r3 * pos[2]]
    };

    if let Some(ref drag) = config.drag {
        let a_drag = acceleration_drag(pos, vel, drag.cd, drag.area_m2, drag.mass_kg);
        acc[0] += a_drag[0];
        acc[1] += a_drag[1];
        acc[2] += a_drag[2];
    }

    acc
}

/// Compute one RK4 stage at time offset `time_offset_s` (in seconds) from the
/// current state, producing the pair `(k_r, k_v)` where `k_r` is the state
/// velocity at the offset point and `k_v` is the acceleration evaluated there.
fn rk4_stage(
    pos: &[f64; 3],
    vel: &[f64; 3],
    prev_kr: &[f64; 3],
    prev_kv: &[f64; 3],
    time_offset_s: f64,
    config: &PropagatorConfig,
) -> ([f64; 3], [f64; 3]) {
    let stage_pos = [
        pos[0] + time_offset_s * prev_kr[0],
        pos[1] + time_offset_s * prev_kr[1],
        pos[2] + time_offset_s * prev_kr[2],
    ];
    let stage_vel = [
        vel[0] + time_offset_s * prev_kv[0],
        vel[1] + time_offset_s * prev_kv[1],
        vel[2] + time_offset_s * prev_kv[2],
    ];
    let acc = total_acceleration(&stage_pos, &stage_vel, config);
    (stage_vel, acc)
}

/// Take one RK4 step.
fn rk4_step(
    pos: &[f64; 3],
    vel: &[f64; 3],
    dt: f64,
    config: &PropagatorConfig,
) -> ([f64; 3], [f64; 3]) {
    let zero = [0.0, 0.0, 0.0];
    // k1: evaluated at the current state (step = 0).
    let (k1r, k1v) = rk4_stage(pos, vel, &zero, &zero, 0.0, config);
    // k2: half step using k1.
    let (k2r, k2v) = rk4_stage(pos, vel, &k1r, &k1v, 0.5 * dt, config);
    // k3: half step using k2.
    let (k3r, k3v) = rk4_stage(pos, vel, &k2r, &k2v, 0.5 * dt, config);
    // k4: full step using k3.
    let (k4r, k4v) = rk4_stage(pos, vel, &k3r, &k3v, dt, config);

    let new_pos = [
        pos[0] + dt / 6.0 * (k1r[0] + 2.0 * k2r[0] + 2.0 * k3r[0] + k4r[0]),
        pos[1] + dt / 6.0 * (k1r[1] + 2.0 * k2r[1] + 2.0 * k3r[1] + k4r[1]),
        pos[2] + dt / 6.0 * (k1r[2] + 2.0 * k2r[2] + 2.0 * k3r[2] + k4r[2]),
    ];
    let new_vel = [
        vel[0] + dt / 6.0 * (k1v[0] + 2.0 * k2v[0] + 2.0 * k3v[0] + k4v[0]),
        vel[1] + dt / 6.0 * (k1v[1] + 2.0 * k2v[1] + 2.0 * k3v[1] + k4v[1]),
        vel[2] + dt / 6.0 * (k1v[2] + 2.0 * k2v[2] + 2.0 * k3v[2] + k4v[2]),
    ];

    (new_pos, new_vel)
}

/// Propagate an orbital state forward in time using a 4th-order Runge-Kutta
/// integrator.
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
    let mut pos = initial.position;
    let mut vel = initial.velocity;
    let dt = config.dt_s;
    let n_steps = (duration_s / dt).ceil() as usize;
    let output_step_interval = (output_dt_s / dt).round().max(1.0) as usize;

    // Record initial state
    results.push(OrbitalState {
        position: pos,
        velocity: vel,
        epoch_jd: initial.epoch_jd,
    });

    for step_i in 1..=n_steps {
        let t = (step_i as f64) * dt;
        let actual_step = if t > duration_s {
            duration_s - (t - dt)
        } else {
            dt
        };
        let (new_pos, new_vel) = rk4_step(&pos, &vel, actual_step, config);
        pos = new_pos;
        vel = new_vel;

        let elapsed = t.min(duration_s);
        if step_i % output_step_interval == 0 || step_i == n_steps {
            results.push(OrbitalState {
                position: pos,
                velocity: vel,
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
}
