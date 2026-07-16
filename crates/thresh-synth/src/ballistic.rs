//! Phased ballistic truth generation over a round rotating Earth.
//!
//! Design Decision 5 of the `orbital-ballistic-filter-models` change: the
//! flat-Earth `Ballistic` segment toy was removed and replaced by this
//! generator, which integrates **in ECI** with the shared closure-based RK4
//! ([`thresh_core::orbital::rk4_step`]) and switches the acceleration
//! closure by flight phase:
//!
//! - **Boost** (`t < burn_time_s`): gravity + constant-magnitude thrust.
//!   Vertical rise until `pitch_over_s`, an instantaneous downrange pitch
//!   kick of `pitch_kick_rad`, then a zero-lift gravity turn with thrust
//!   aligned to the atmosphere-relative velocity `v_rel = v − ω_⊕ × r`.
//!   No mass depletion, no staging (out of scope).
//! - **Midcourse** (burnout → sensible atmosphere): exactly the existing J2
//!   RK4 path — the acceleration closure is
//!   [`thresh_core::orbital::j2_acceleration`], identical to
//!   [`crate::orbital::propagate`] with `include_j2: true, drag: None`.
//! - **Reentry** (altitude < 100 km, descending): gravity + J2 plus
//!   [`thresh_core::orbital::drag_acceleration`]`(r, v, 1/β)` — the same
//!   closure `thresh_filter`'s `BallisticReentry` model integrates, so
//!   truth and filter physics are identical by construction.
//!
//! The rotating round Earth enters through the physics, not the frame: the
//! launch pad inherits the Earth-rotation velocity `ω_⊕ × r`, and drag uses
//! the co-rotating atmosphere. Altitude and the terminal surface are
//! spherical (`‖r‖ − equatorial_radius`), consistent with the shared drag
//! convention; launch "vertical" is the geocentric radial direction under
//! the same approximation.

use nalgebra::Vector3;
use serde::{Deserialize, Serialize};
use thresh_core::eci::{SECONDS_PER_DAY, ecef_to_eci, eci_to_ecef};
use thresh_core::geodetic::{ecef_to_enu, wgs84_to_ecef};
use thresh_core::orbital::{GravityModel, drag_acceleration, j2_acceleration, rk4_step};

use crate::orbital::{OrbitalState, orbital_to_enu};
use crate::trajectory::Waypoint;

/// Earth gravity model shared with the orbital propagator and filter models.
const EARTH: GravityModel = GravityModel::EARTH_WGS84;

/// Reentry-phase altitude threshold (m): below this, a descending
/// post-burnout vehicle is in the reentry phase and feels drag
/// (100 km Kármán-line convention, design Decision 5).
const REENTRY_ALTITUDE_M: f64 = 100_000.0;

/// Safety cap on integrated flight time (s) so a misconfigured profile
/// (e.g. thrust high enough to reach orbit) cannot loop forever. Any
/// suborbital ballistic flight is far shorter than two hours.
const MAX_FLIGHT_S: f64 = 7_200.0;

/// Configuration for a phased boost / midcourse / reentry ballistic
/// trajectory (design Decision 5 of `orbital-ballistic-filter-models`).
///
/// Launch geometry is geodetic (WGS-84); the generated states are ECI
/// samples ([`OrbitalState`]) like the orbital propagator's output, so the
/// downstream ENU / radar pipeline consumes ballistic truth exactly like
/// the SGP4 orbital source (see [`ballistic_to_enu`] /
/// [`ballistic_to_waypoints`]).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BallisticProfile {
    /// Launch geodetic latitude (rad). Launches from the exact poles are
    /// unsupported (the launch azimuth is undefined there).
    pub launch_lat_rad: f64,
    /// Launch geodetic longitude (rad).
    pub launch_lon_rad: f64,
    /// Launch altitude above the WGS-84 ellipsoid (m).
    pub launch_alt_m: f64,
    /// Launch azimuth (rad, clockwise from north): the downrange direction
    /// of the pitch kick and subsequent gravity turn.
    pub launch_azimuth_rad: f64,
    /// Constant thrust acceleration magnitude during boost (m/s²).
    /// No mass depletion or staging is modeled.
    pub thrust_accel: f64,
    /// Boost duration (s); thrust cuts off instantaneously at burnout.
    pub burn_time_s: f64,
    /// Duration of the initial vertical rise (s); the pitch kick is applied
    /// at this time.
    pub pitch_over_s: f64,
    /// Instantaneous downrange pitch kick angle (rad) applied to the
    /// atmosphere-relative velocity at `pitch_over_s`, after which the
    /// zero-lift gravity turn takes over.
    pub pitch_kick_rad: f64,
    /// Ballistic coefficient β = m/(C_d·A) (kg/m²) for the reentry drag.
    pub beta: f64,
    /// Launch epoch as a Julian Date in the **UTC** scale.
    ///
    /// Kept as a raw `f64` (not `thresh_core::time::Epoch`) for the same
    /// reason as [`OrbitalState::epoch_jd`](crate::orbital::OrbitalState):
    /// this profile seeds the calibrated ballistic benchmark chain, whose
    /// metrics must stay bitwise stable across the `astro-time-and-frames`
    /// migration (design Decision 6), and JD↔`Epoch` round trips are not
    /// guaranteed bit-exact.
    pub epoch_jd: f64,
}

/// Flight phase of the ballistic trajectory, evaluated per RK4 step.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Phase {
    /// Vertical rise: thrust along the geocentric up direction.
    BoostVertical,
    /// Gravity turn: thrust along the atmosphere-relative velocity.
    BoostTurn,
    /// Exoatmospheric coast: two-body + J2 gravity only.
    Midcourse,
    /// Descending below [`REENTRY_ALTITUDE_M`]: gravity + β-drag.
    Reentry,
}

/// Generate the phased ballistic trajectory as ECI samples on a `dt` grid.
///
/// The first sample is the launch pad at `profile.epoch_jd` (co-rotating
/// with the Earth, so its ECI velocity is `ω_⊕ × r`); subsequent samples
/// advance by `dt` seconds each. Integration terminates once the vehicle
/// descends through the spherically-approximated surface
/// (`‖r‖ ≤ equatorial_radius`), so the final sample is at or just below
/// the surface.
///
/// # Panics
/// Panics if `dt` is not positive and finite.
pub fn generate(profile: &BallisticProfile, dt: f64) -> Vec<OrbitalState> {
    assert!(dt > 0.0 && dt.is_finite(), "dt must be positive and finite");
    integrate_profile(profile, dt)
}

// ---------------------------------------------------------------------------
// Phase helpers (design Decision 5: each ≤ 15 cognitive complexity)
// ---------------------------------------------------------------------------

/// ECI velocity of the co-rotating Earth (atmosphere / launch pad) at `pos`:
/// `ω_⊕ × r`. Matches the co-rotation term inside
/// [`thresh_core::orbital::drag_acceleration`].
fn earth_rotation_velocity(pos: &Vector3<f64>) -> Vector3<f64> {
    use thresh_core::eci::EARTH_ROTATION_RATE;
    Vector3::new(
        -EARTH_ROTATION_RATE * pos.y,
        EARTH_ROTATION_RATE * pos.x,
        0.0,
    )
}

/// Launch pad state in ECI at the profile epoch: geodetic → ECEF → ECI.
/// The pad is Earth-fixed, so its ECI velocity is exactly `ω_⊕ × r`
/// (supplied by [`ecef_to_eci`] from a zero ECEF velocity).
fn launch_state_eci(profile: &BallisticProfile) -> (Vector3<f64>, Vector3<f64>) {
    let pad_ecef = wgs84_to_ecef(
        profile.launch_lat_rad,
        profile.launch_lon_rad,
        profile.launch_alt_m,
    );
    ecef_to_eci(&pad_ecef, &Vector3::zeros(), profile.epoch_jd)
}

/// Downrange unit vector at `pos` for the given launch azimuth: the
/// horizontal direction `sin(az)·east + cos(az)·north` in the geocentric
/// local frame (up = `r̂`, consistent with the module's spherical
/// approximation). Falls back to the ECI x axis for the degenerate polar
/// case where east is undefined.
fn downrange_direction(pos: &Vector3<f64>, azimuth_rad: f64) -> Vector3<f64> {
    let up = pos.normalize();
    let horizontal = Vector3::new(-up.y, up.x, 0.0); // ẑ × up
    let east = if horizontal.norm() > 1e-12 {
        horizontal.normalize()
    } else {
        Vector3::x()
    };
    let north = up.cross(&east);
    east * azimuth_rad.sin() + north * azimuth_rad.cos()
}

/// Rodrigues rotation of `v` about the unit `axis` by `angle` radians.
fn rotate_about_axis(v: &Vector3<f64>, axis: &Vector3<f64>, angle: f64) -> Vector3<f64> {
    let (s, c) = angle.sin_cos();
    v * c + axis.cross(v) * s + axis * (axis.dot(v)) * (1.0 - c)
}

/// Apply the instantaneous downrange pitch kick at `pitch_over_s`: rotate
/// the atmosphere-relative velocity by `pitch_kick_rad` from up toward the
/// launch-azimuth downrange direction, preserving its magnitude and the
/// co-rotation component. Returns the new ECI velocity.
fn apply_pitch_kick(
    profile: &BallisticProfile,
    pos: &Vector3<f64>,
    vel: &Vector3<f64>,
) -> Vector3<f64> {
    let v_atm = earth_rotation_velocity(pos);
    let v_rel = vel - v_atm;
    let up = pos.normalize();
    let downrange = downrange_direction(pos, profile.launch_azimuth_rad);
    // up ⊥ downrange and both are unit, so the axis is unit: rotating up
    // about it by +angle tilts it toward downrange.
    let axis = up.cross(&downrange);
    v_atm + rotate_about_axis(&v_rel, &axis, profile.pitch_kick_rad)
}

/// Classify the flight phase at time `t` and state `(pos, vel)`, evaluated
/// once per RK4 step (design Decision 5: phase boundaries are time/altitude
/// events). Reentry requires *descending* below [`REENTRY_ALTITUDE_M`]; a
/// post-burnout vehicle still ascending through the atmosphere coasts
/// drag-free in midcourse (documented fidelity floor).
fn phase_of(profile: &BallisticProfile, t: f64, pos: &Vector3<f64>, vel: &Vector3<f64>) -> Phase {
    if t < profile.burn_time_s {
        return if t < profile.pitch_over_s {
            Phase::BoostVertical
        } else {
            Phase::BoostTurn
        };
    }
    let altitude = pos.norm() - EARTH.equatorial_radius;
    let descending = pos.dot(vel) < 0.0;
    if altitude < REENTRY_ALTITUDE_M && descending {
        Phase::Reentry
    } else {
        Phase::Midcourse
    }
}

/// Boost-phase acceleration: shared two-body + J2 gravity plus constant
/// thrust acceleration. Thrust points along geocentric up during the
/// vertical rise, and along the atmosphere-relative velocity during the
/// zero-lift gravity turn (`gravity_turn = true`), falling back to up while
/// `v_rel` is negligible (e.g. the first instants on the pad).
fn boost_accel(
    profile: &BallisticProfile,
    gravity_turn: bool,
    pos: &Vector3<f64>,
    vel: &Vector3<f64>,
) -> Vector3<f64> {
    let up = pos.normalize();
    let v_rel = vel - earth_rotation_velocity(pos);
    let thrust_dir = if gravity_turn && v_rel.norm() > 1e-6 {
        v_rel.normalize()
    } else {
        up
    };
    j2_acceleration(pos, &EARTH) + thrust_dir * profile.thrust_accel
}

/// Total acceleration for one RK4 step of the given phase.
///
/// Midcourse is **exactly** the shared [`j2_acceleration`] — identical to
/// [`crate::orbital::propagate`] with `include_j2: true, drag: None` — and
/// reentry adds [`drag_acceleration`]`(r, v, 1/β)`, the same closure the
/// `BallisticReentry` filter model integrates (truth/filter identity).
fn phase_acceleration(
    profile: &BallisticProfile,
    phase: Phase,
    pos: &Vector3<f64>,
    vel: &Vector3<f64>,
) -> Vector3<f64> {
    match phase {
        Phase::BoostVertical => boost_accel(profile, false, pos, vel),
        Phase::BoostTurn => boost_accel(profile, true, pos, vel),
        Phase::Midcourse => j2_acceleration(pos, &EARTH),
        Phase::Reentry => {
            j2_acceleration(pos, &EARTH) + drag_acceleration(pos, vel, 1.0 / profile.beta)
        }
    }
}

/// Build the ECI sample for elapsed time `t` (s) past the profile epoch.
fn sample_state(
    profile: &BallisticProfile,
    t: f64,
    pos: &Vector3<f64>,
    vel: &Vector3<f64>,
) -> OrbitalState {
    OrbitalState {
        position: [pos.x, pos.y, pos.z],
        velocity: [vel.x, vel.y, vel.z],
        epoch_jd: profile.epoch_jd + t / SECONDS_PER_DAY,
    }
}

/// Integration loop: per step, apply the (one-shot) pitch kick when its
/// time arrives, classify the phase, take one shared RK4 step under that
/// phase's acceleration closure, and record the sample. Terminates at the
/// spherically-approximated surface on descent, or at [`MAX_FLIGHT_S`] as
/// a safety cap.
fn integrate_profile(profile: &BallisticProfile, dt: f64) -> Vec<OrbitalState> {
    let (mut pos, mut vel) = launch_state_eci(profile);
    let mut samples = vec![sample_state(profile, 0.0, &pos, &vel)];
    let mut kicked = false;
    let max_steps = (MAX_FLIGHT_S / dt).ceil() as usize;
    for step in 0..max_steps {
        let t = step as f64 * dt;
        if !kicked && t >= profile.pitch_over_s {
            vel = apply_pitch_kick(profile, &pos, &vel);
            kicked = true;
        }
        let phase = phase_of(profile, t, &pos, &vel);
        let (p, v) = rk4_step(&pos, &vel, dt, |p, v| {
            phase_acceleration(profile, phase, p, v)
        });
        pos = p;
        vel = v;
        samples.push(sample_state(profile, (step + 1) as f64 * dt, &pos, &vel));
        if pos.dot(&vel) < 0.0 && pos.norm() <= EARTH.equatorial_radius {
            break;
        }
    }
    samples
}

// ---------------------------------------------------------------------------
// Output adapters (task 4.4): ENU station projection + waypoint grid
// ---------------------------------------------------------------------------

/// Project ballistic ECI truth samples into a ground station's local ENU
/// frame, reusing [`orbital_to_enu`] (→ [`thresh_core::eci::eci_to_enu`])
/// per sample — the same station projection the SGP4 orbital source uses.
///
/// No visibility filtering is applied; below-horizon samples come out with
/// negative Up and are the scenario layer's concern (see
/// [`crate::orbital::is_visible`]).
pub fn ballistic_to_enu(
    states: &[OrbitalState],
    station_lat_rad: f64,
    station_lon_rad: f64,
    station_alt_m: f64,
) -> Vec<[f64; 3]> {
    states
        .iter()
        .map(|s| orbital_to_enu(s, station_lat_rad, station_lon_rad, station_alt_m))
        .collect()
}

/// Adapt ballistic ECI truth samples to the [`Waypoint`] grid the radar
/// generator consumes ([`crate::radar_trajectory::TargetTrack`] /
/// [`crate::measurement_gen`]), so ballistic truth feeds the radar pipeline
/// exactly like the SGP4 orbital source: ECI → station ENU → radar RAE.
///
/// Waypoint times are seconds since the first sample's epoch; positions and
/// velocities are in the station's ENU frame (the velocity is the
/// Earth-fixed observer's, i.e. it includes the frame-rotation term — a
/// co-rotating launch pad has near-zero ENU velocity).
pub fn ballistic_to_waypoints(
    states: &[OrbitalState],
    station_lat_rad: f64,
    station_lon_rad: f64,
    station_alt_m: f64,
) -> Vec<Waypoint> {
    let Some(first) = states.first() else {
        return Vec::new();
    };
    let station_ecef = wgs84_to_ecef(station_lat_rad, station_lon_rad, station_alt_m);
    states
        .iter()
        .map(|s| {
            let pos_eci = Vector3::from(s.position);
            let vel_eci = Vector3::from(s.velocity);
            let (pos_ecef, vel_ecef) = eci_to_ecef(&pos_eci, &vel_eci, s.epoch_jd);
            let enu_pos = ecef_to_enu(&pos_ecef, station_lat_rad, station_lon_rad, station_alt_m);
            // ENU velocity is rotation-only: feed the rotate-and-translate
            // helper an offset from the station reference so the
            // translation cancels (R·((ref + v) − ref) = R·v).
            let enu_vel = ecef_to_enu(
                &(station_ecef + vel_ecef),
                station_lat_rad,
                station_lon_rad,
                station_alt_m,
            );
            Waypoint {
                time: (s.epoch_jd - first.epoch_jd) * SECONDS_PER_DAY,
                position: [enu_pos.x, enu_pos.y, enu_pos.z],
                velocity: [enu_vel.x, enu_vel.y, enu_vel.z],
            }
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::measurement_gen::RadarConfig;
    use crate::orbital::PropagatorConfig;
    use crate::radar_trajectory::{
        TargetTrack, TrajectoryRadarConfig, measurements_from_trajectory,
    };
    use rand::SeedableRng;
    use rand::rngs::StdRng;
    use thresh_core::geodetic::ecef_to_wgs84;

    /// MRBM-class test profile: equatorial due-east launch, ~5 g constant
    /// thrust for 65 s, β = 2000 kg/m². Observed shape at dt = 1 s: burnout
    /// 79 km / 2.9 km/s, apogee 408 km, ground range 647 km, flight 667 s,
    /// reentry (descending below 100 km) from t ≈ 615 s.
    fn mrbm_profile() -> BallisticProfile {
        BallisticProfile {
            launch_lat_rad: 0.0,
            launch_lon_rad: 0.0,
            launch_alt_m: 0.0,
            launch_azimuth_rad: 90.0_f64.to_radians(),
            thrust_accel: 50.0,
            burn_time_s: 65.0,
            pitch_over_s: 10.0,
            pitch_kick_rad: 0.30,
            beta: 2_000.0,
            epoch_jd: 2_451_545.0,
        }
    }

    fn altitude(s: &OrbitalState) -> f64 {
        s.radius() - EARTH.equatorial_radius
    }

    fn is_descending(s: &OrbitalState) -> bool {
        Vector3::from(s.position).dot(&Vector3::from(s.velocity)) < 0.0
    }

    /// Atmosphere-relative velocity of a sample: `v − ω_⊕ × r`.
    fn v_rel_of(s: &OrbitalState) -> Vector3<f64> {
        Vector3::from(s.velocity) - earth_rotation_velocity(&Vector3::from(s.position))
    }

    /// Flight-path angle of the atmosphere-relative velocity above the local
    /// horizon (rad).
    fn flight_path_angle(s: &OrbitalState) -> f64 {
        let up = Vector3::from(s.position).normalize();
        let v_rel = v_rel_of(s);
        (v_rel.dot(&up) / v_rel.norm()).asin()
    }

    /// Equatorial pad position at 1 km altitude with its co-rotating ECI
    /// velocity — a v_rel ≈ 0 state for the boost helpers.
    fn corotating_pad_state() -> (Vector3<f64>, Vector3<f64>) {
        let pos = Vector3::new(EARTH.equatorial_radius + 1_000.0, 0.0, 0.0);
        let vel = earth_rotation_velocity(&pos);
        (pos, vel)
    }

    // ── Task 4.1: per-helper unit tests ─────────────────────────────────

    #[test]
    fn downrange_direction_is_horizontal_unit_vector() {
        // Equatorial point on the x axis: up = x̂, east = ŷ, north = ẑ.
        let pos = Vector3::new(EARTH.equatorial_radius, 0.0, 0.0);
        let north = downrange_direction(&pos, 0.0);
        let east = downrange_direction(&pos, 90.0_f64.to_radians());
        assert!((north - Vector3::z()).norm() < 1e-12, "azimuth 0 → north");
        assert!((east - Vector3::y()).norm() < 1e-12, "azimuth 90° → east");
        // Generic azimuth: unit length, perpendicular to up.
        let d = downrange_direction(&pos, 0.7);
        assert!((d.norm() - 1.0).abs() < 1e-12);
        assert!(d.dot(&pos.normalize()).abs() < 1e-12);
    }

    #[test]
    fn rotate_about_axis_matches_quarter_turn() {
        let rotated = rotate_about_axis(&Vector3::x(), &Vector3::z(), std::f64::consts::FRAC_PI_2);
        assert!((rotated - Vector3::y()).norm() < 1e-12);
        // Component along the axis is preserved.
        let v = Vector3::new(1.0, 2.0, 3.0);
        let r = rotate_about_axis(&v, &Vector3::z(), 1.1);
        assert!((r.z - v.z).abs() < 1e-12);
        assert!((r.norm() - v.norm()).abs() < 1e-12);
    }

    #[test]
    fn pitch_kick_rotates_v_rel_by_kick_angle() {
        let profile = mrbm_profile();
        let (pos, pad_vel) = corotating_pad_state();
        // Vertical-rise state: v_rel straight up at 400 m/s.
        let vel = pad_vel + pos.normalize() * 400.0;
        let kicked = apply_pitch_kick(&profile, &pos, &vel);

        let v_rel_before = vel - earth_rotation_velocity(&pos);
        let v_rel_after = kicked - earth_rotation_velocity(&pos);
        // Magnitude preserved, rotated by exactly pitch_kick_rad.
        assert!((v_rel_after.norm() - v_rel_before.norm()).abs() < 1e-9);
        let cos_angle = v_rel_before.dot(&v_rel_after) / (v_rel_before.norm() * v_rel_after.norm());
        assert!((cos_angle.acos() - profile.pitch_kick_rad).abs() < 1e-9);
        // Tilted toward the launch-azimuth downrange direction (east here).
        let downrange = downrange_direction(&pos, profile.launch_azimuth_rad);
        assert!(v_rel_after.dot(&downrange) > 0.0);
    }

    #[test]
    fn phase_of_classifies_time_and_altitude_events() {
        let profile = mrbm_profile();
        let up = Vector3::new(1.0, 0.0, 0.0);
        let at_alt = |alt: f64| up * (EARTH.equatorial_radius + alt);
        let ascending = up * 3_000.0;
        let descending = up * -3_000.0;

        // Time events during boost.
        let p = at_alt(1_000.0);
        assert_eq!(
            phase_of(&profile, 5.0, &p, &ascending),
            Phase::BoostVertical
        );
        assert_eq!(phase_of(&profile, 30.0, &p, &ascending), Phase::BoostTurn);
        // Post-burnout altitude/direction events: ascending through the
        // atmosphere coasts drag-free (documented fidelity floor)…
        let p80 = at_alt(80_000.0);
        assert_eq!(phase_of(&profile, 70.0, &p80, &ascending), Phase::Midcourse);
        // …descending below 100 km is reentry; above it, still midcourse.
        assert_eq!(phase_of(&profile, 600.0, &p80, &descending), Phase::Reentry);
        let p150 = at_alt(150_000.0);
        assert_eq!(
            phase_of(&profile, 600.0, &p150, &descending),
            Phase::Midcourse
        );
    }

    #[test]
    fn boost_accel_thrust_directions() {
        let profile = mrbm_profile();
        let (pos, pad_vel) = corotating_pad_state();
        let up = pos.normalize();
        let gravity = j2_acceleration(&pos, &EARTH);

        // Vertical rise: thrust exactly along geocentric up.
        let vertical = boost_accel(&profile, false, &pos, &pad_vel) - gravity;
        assert!((vertical - up * profile.thrust_accel).norm() < 1e-9);

        // Gravity turn with negligible v_rel falls back to up.
        let fallback = boost_accel(&profile, true, &pos, &pad_vel) - gravity;
        assert!((fallback - up * profile.thrust_accel).norm() < 1e-9);

        // Gravity turn with a real v_rel: thrust along v̂_rel.
        let v_rel = Vector3::new(300.0, 500.0, 100.0);
        let turn = boost_accel(&profile, true, &pos, &(pad_vel + v_rel)) - gravity;
        assert!((turn - v_rel.normalize() * profile.thrust_accel).norm() < 1e-9);
    }

    #[test]
    fn integrate_profile_grid_starts_at_corotating_pad_and_terminates_at_surface() {
        let profile = mrbm_profile();
        let dt = 1.0;
        let states = generate(&profile, dt);
        assert!(states.len() > 600, "MRBM flight should exceed 600 s");

        // First sample is the pad state: matches the geodetic→ECI launch
        // helper bitwise and co-rotates with the Earth (~465 m/s eastward
        // at the equator).
        let (pad_pos, pad_vel) = launch_state_eci(&profile);
        let first = &states[0];
        assert_eq!(first.position, [pad_pos.x, pad_pos.y, pad_pos.z]);
        assert_eq!(first.velocity, [pad_vel.x, pad_vel.y, pad_vel.z]);
        let pad_speed = Vector3::from(first.velocity).norm();
        assert!(
            (460.0..470.0).contains(&pad_speed),
            "equatorial pad ECI speed {pad_speed} m/s"
        );

        // Uniform dt sample grid (epochs advance by dt/86400; the tolerance
        // reflects the ~5e-5 s quantization of an f64 Julian date).
        for pair in states.windows(2) {
            let step_s = (pair[1].epoch_jd - pair[0].epoch_jd) * SECONDS_PER_DAY;
            assert!((step_s - dt).abs() < 1e-3);
        }

        // Terminates at the spherically-approximated surface on descent:
        // last sample at/below the surface, previous still above.
        let last = states.last().unwrap();
        assert!(is_descending(last));
        assert!(altitude(last) <= 0.0);
        assert!(altitude(&states[states.len() - 2]) > 0.0);
        for s in &states {
            assert!(s.position.iter().chain(&s.velocity).all(|v| v.is_finite()));
        }
    }

    #[test]
    fn integrate_profile_caps_runaway_flight() {
        // Thrust/burn high enough to reach orbit: no surface impact, so the
        // MAX_FLIGHT_S safety cap must terminate the loop.
        let mut profile = mrbm_profile();
        profile.thrust_accel = 60.0;
        profile.burn_time_s = 400.0;
        let dt = 10.0;
        let states = generate(&profile, dt);
        let expected = (MAX_FLIGHT_S / dt) as usize + 1;
        assert_eq!(states.len(), expected, "capped at MAX_FLIGHT_S");
        assert!(altitude(states.last().unwrap()) > 0.0, "never impacted");
    }

    #[test]
    #[should_panic(expected = "dt must be positive")]
    fn generate_rejects_non_positive_dt() {
        generate(&mrbm_profile(), 0.0);
    }

    // ── Task 4.2: boost phase behavior ──────────────────────────────────

    #[test]
    fn boost_rises_vertically_until_pitch_over() {
        let profile = mrbm_profile();
        let states = generate(&profile, 1.0);
        let enu = ballistic_to_enu(&states, 0.0, 0.0, 0.0);

        // Through the vertical rise the pad-frame horizontal offset stays
        // tiny (only Coriolis-scale drift) while Up climbs monotonically.
        for k in 1..=10 {
            let [e, n, u] = enu[k];
            assert!(
                e.hypot(n) < 20.0,
                "t={k}: horizontal offset {} m during vertical rise",
                e.hypot(n)
            );
            assert!(u > enu[k - 1][2], "t={k}: Up must increase");
        }
        // Kinematics ≈ ½(a_thrust − g)t² at pitch-over.
        let expected = 0.5 * (profile.thrust_accel - 9.81) * 100.0;
        let up_10 = enu[10][2];
        assert!(
            (up_10 - expected).abs() < 0.15 * expected,
            "vertical-rise altitude {up_10} m vs ~{expected} m"
        );
    }

    #[test]
    fn gravity_turn_bends_v_rel_downrange_monotonically() {
        let profile = mrbm_profile();
        let states = generate(&profile, 1.0);

        // Post-kick (kick applies at t = 10, first turned sample t = 11):
        // zero-lift gravity turn — thrust along v_rel adds no torque, so
        // gravity alone rotates v_rel downward: the flight-path angle
        // decreases monotonically through boost.
        let mut gamma_prev = flight_path_angle(&states[11]);
        for (k, s) in states.iter().enumerate().take(66).skip(12) {
            let gamma = flight_path_angle(s);
            assert!(gamma < gamma_prev, "t={k}: γ must decrease ({gamma})");
            gamma_prev = gamma;
        }

        // At burnout the turn has made real progress: well below vertical
        // and moving downrange (east for the 90° azimuth) at speed.
        let burnout = &states[65];
        let gamma_bo = flight_path_angle(burnout);
        assert!(gamma_bo < flight_path_angle(&states[11]) - 0.1);
        let east =
            downrange_direction(&Vector3::from(burnout.position), profile.launch_azimuth_rad);
        assert!(
            v_rel_of(burnout).dot(&east) > 500.0,
            "burnout downrange v_rel component"
        );
    }

    // ── Task 4.6: truth-generation scenarios ────────────────────────────

    // Spec "Phased ballistic trajectory generation": position and velocity
    // are continuous at both phase boundaries — the acceleration closure
    // switches, the state never jumps. (The pitch kick at t = 10 s is a
    // designed velocity discontinuity *inside* boost, not a phase boundary.)
    #[test]
    fn position_and_velocity_continuous_at_phase_boundaries() {
        let profile = mrbm_profile();
        let dt = 1.0;
        let states = generate(&profile, dt);

        // Boundary 1: burnout (boost → midcourse) at t = 65.
        let burnout = profile.burn_time_s as usize;
        // Boundary 2: first descending sample below 100 km (→ reentry).
        let reentry = states
            .iter()
            .position(|s| is_descending(s) && altitude(s) < REENTRY_ALTITUDE_M)
            .expect("trajectory must reenter");
        assert!(reentry > burnout);

        for &k in &[burnout, reentry] {
            let (p0, v0) = (
                Vector3::from(states[k].position),
                Vector3::from(states[k].velocity),
            );
            let (p1, v1) = (
                Vector3::from(states[k + 1].position),
                Vector3::from(states[k + 1].velocity),
            );
            // Across the boundary step the acceleration is gravity plus (at
            // 100 km) negligible drag ≈ 10 m/s²: a velocity or position jump
            // would blow far past these one-step kinematic bounds.
            assert!(
                (v1 - v0).norm() < 15.0 * dt,
                "velocity jump {} m/s at boundary sample {k}",
                (v1 - v0).norm()
            );
            assert!(
                (p1 - p0 - v0 * dt).norm() < 8.0 * dt * dt,
                "position jump at boundary sample {k}"
            );
        }
    }

    // Spec "Round rotating Earth" (curvature): the impact point sits on the
    // spherically-approximated round Earth, so seen from the launch station
    // it is depressed by the curvature drop R·(1 − cos θ) — a flat-Earth
    // tangent-plane trajectory would impact near Up ≈ 0.
    #[test]
    fn ground_track_follows_round_earth() {
        let profile = mrbm_profile();
        let states = generate(&profile, 1.0);
        let last = states.last().unwrap();

        // Impact is at the surface (within one 1 s descent step).
        assert!(altitude(last) <= 0.0 && altitude(last) > -5_000.0);

        // Central angle launch → impact in the Earth-fixed frame.
        let (impact_ecef, _) = eci_to_ecef(
            &Vector3::from(last.position),
            &Vector3::from(last.velocity),
            last.epoch_jd,
        );
        let pad_ecef = wgs84_to_ecef(0.0, 0.0, 0.0);
        let theta = (impact_ecef.normalize().dot(&pad_ecef.normalize())).acos();
        let ground_range = theta * EARTH.equatorial_radius;
        assert!(
            (500_000.0..800_000.0).contains(&ground_range),
            "MRBM ground range {ground_range} m"
        );

        // The launch-station ENU Up of the impact matches the round-Earth
        // depression ‖r_impact‖·cos θ − ‖r_pad‖ (≈ −33 km here), not the
        // flat-Earth ≈ 0.
        let enu = ballistic_to_enu(&states, 0.0, 0.0, 0.0);
        let up_impact = enu.last().unwrap()[2];
        let expected = impact_ecef.norm() * theta.cos() - pad_ecef.norm();
        assert!(
            up_impact < -20_000.0,
            "flat-Earth-like impact Up {up_impact}"
        );
        assert!(
            (up_impact - expected).abs() < 0.05 * expected.abs(),
            "impact Up {up_impact} vs round-Earth {expected}"
        );
    }

    // Spec "Round rotating Earth" (rotation): a due-north equatorial launch
    // would stay on the launch meridian for all time on a non-rotating Earth
    // (the trajectory plane contains the meridian by symmetry). With rotation
    // modeled, the vehicle keeps the pad's eastward velocity while its
    // angular rate at altitude lags ω_⊕, so the Earth-fixed impact point
    // shifts measurably off the meridian during the flight time.
    #[test]
    fn impact_point_shifts_with_earth_rotation() {
        let mut profile = mrbm_profile();
        profile.launch_azimuth_rad = 0.0; // due north
        let states = generate(&profile, 1.0);
        let last = states.last().unwrap();
        let (impact_ecef, _) = eci_to_ecef(
            &Vector3::from(last.position),
            &Vector3::from(last.velocity),
            last.epoch_jd,
        );
        let (lat, lon, _) = ecef_to_wgs84(&impact_ecef);

        assert!(lat > 4.0_f64.to_radians(), "northward ground track");
        // Westward longitude shift (observed ≈ 29 km for the ~650 s flight;
        // ω_⊕·T·R ≈ 34 km sets the scale).
        let shift_m = lon * EARTH.equatorial_radius;
        assert!(
            shift_m < -5_000.0,
            "impact must shift westward off the launch meridian, got {shift_m} m"
        );
        assert!(shift_m.abs() < 100_000.0, "shift beyond rotation scale");
    }

    // Spec "Midcourse consistency with the orbital propagator": a midcourse
    // state propagated independently through `propagate` with
    // `include_j2: true, drag: None` matches the generator's samples — both
    // integrate the identical shared closure (`j2_acceleration`) with the
    // identical `rk4_step`, so the tolerance is float-noise 1e-6 m.
    #[test]
    fn midcourse_matches_orbital_propagator() {
        let profile = mrbm_profile();
        let dt = 1.0;
        let states = generate(&profile, dt);

        let (start, duration) = (100usize, 400.0);
        let config = PropagatorConfig {
            include_j2: true,
            drag: None,
            dt_s: dt,
        };
        let reference = crate::orbital::propagate(&states[start], duration, &config, dt);
        assert_eq!(reference.len(), duration as usize + 1);

        for (k, r) in reference.iter().enumerate() {
            let s = &states[start + k];
            // Guard the phase assumption: the whole window is exoatmospheric
            // midcourse (above 100 km), so the generator used the J2-only
            // closure throughout.
            assert!(
                altitude(s) > REENTRY_ALTITUDE_M,
                "sample {k} left midcourse"
            );
            for i in 0..3 {
                assert!(
                    (s.position[i] - r.position[i]).abs() < 1e-6,
                    "position[{i}] at +{k} s differs by {} m",
                    (s.position[i] - r.position[i]).abs()
                );
                assert!((s.velocity[i] - r.velocity[i]).abs() < 1e-9);
            }
        }
    }

    // ── Task 4.4: output adapters ───────────────────────────────────────

    #[test]
    fn enu_projection_starts_at_station_and_matches_orbital_helper() {
        let profile = mrbm_profile();
        let states = generate(&profile, 1.0);
        let enu = ballistic_to_enu(&states, 0.0, 0.0, 0.0);
        assert_eq!(enu.len(), states.len());

        // The pad is the station: the first sample projects to the origin.
        assert!(enu[0].iter().all(|c| c.abs() < 1.0), "pad ENU {:?}", enu[0]);

        // Same station projection as the SGP4 orbital source (bitwise —
        // it is the same `orbital_to_enu` per sample).
        for (k, s) in states.iter().enumerate().step_by(100) {
            let direct = orbital_to_enu(s, 0.0, 0.0, 0.0);
            assert_eq!(enu[k], direct, "sample {k}");
        }
    }

    #[test]
    fn waypoint_adapter_feeds_radar_generator_like_orbital_source() {
        let profile = mrbm_profile();
        let dt = 1.0;
        let states = generate(&profile, dt);
        // Downrange station ~333 km east of the pad on the equator.
        let (st_lat, st_lon, st_alt) = (0.0, 3.0_f64.to_radians(), 0.0);
        let waypoints = ballistic_to_waypoints(&states, st_lat, st_lon, st_alt);
        assert_eq!(waypoints.len(), states.len());

        // Times are seconds since the first sample on the dt grid (the
        // tolerance reflects f64 Julian-date quantization, ~5e-5 s).
        for (k, wp) in waypoints.iter().enumerate() {
            assert!((wp.time - k as f64 * dt).abs() < 1e-3);
        }
        // The co-rotating pad has near-zero Earth-fixed (ENU) velocity.
        let v0 = Vector3::from(waypoints[0].velocity).norm();
        assert!(v0 < 1.0, "pad ENU speed {v0} m/s");
        // ENU velocity is consistent with the ENU position differences
        // (validates the rotation-only velocity transform mid-flight).
        let (a, b) = (&waypoints[300], &waypoints[301]);
        let dp = (Vector3::from(b.position) - Vector3::from(a.position)) / dt;
        let v_mid = (Vector3::from(a.velocity) + Vector3::from(b.velocity)) / 2.0;
        assert!(
            (dp - v_mid).norm() < 2.0,
            "ENU velocity inconsistent with positions: {} m/s",
            (dp - v_mid).norm()
        );

        // The radar generator consumes the ballistic waypoint grid exactly
        // like the SGP4 orbital source: every tick of the flight yields a
        // radar measurement with a deterministic-detection config.
        let target = TargetTrack {
            waypoints,
            class_id: 1,
            size_override: None,
        };
        let config = TrajectoryRadarConfig {
            sample_rate_hz: 1.0 / dt,
            max_range_m: 1_500_000.0,
            radar: RadarConfig {
                p_detection: 1.0,
                max_range: 1_500_000.0,
                ..RadarConfig::default()
            },
            ..TrajectoryRadarConfig::default()
        };
        let mut rng = StdRng::seed_from_u64(42);
        let per_tick = measurements_from_trajectory(&[target], &config, &mut rng).unwrap();
        assert_eq!(per_tick.len(), states.len());
        assert!(
            per_tick.iter().all(|tick| tick.len() == 1),
            "every tick must yield exactly one radar measurement"
        );
    }
}
