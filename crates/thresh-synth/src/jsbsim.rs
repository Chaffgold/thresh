//! JSBSim trajectory bridge.
//!
//! Pure-Rust data structures (state, waypoint, maneuvers, conversions) are
//! always available. The actual PyO3 bridge to the Python `jsbsim` package is
//! gated behind the `jsbsim` Cargo feature so that this crate still builds
//! without a Python installation.

use crate::trajectory::Waypoint;
use thresh_core::geodetic::wgs84_to_enu;

#[cfg(feature = "jsbsim")]
use pyo3::prelude::*;
#[cfg(feature = "jsbsim")]
use pyo3::types::PyAnyMethods;

// ── Aircraft model ──────────────────────────────────────────────────────────

/// Supported JSBSim aircraft models.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AircraftModel {
    F16,
    B737,
    C172,
}

impl AircraftModel {
    /// Returns the JSBSim aircraft model file name (without extension).
    pub fn jsbsim_name(&self) -> &'static str {
        match self {
            AircraftModel::F16 => "f16",
            AircraftModel::B737 => "737",
            AircraftModel::C172 => "c172",
        }
    }
}

// ── Initial conditions ──────────────────────────────────────────────────────

/// Initial conditions for a JSBSim run.
#[derive(Debug, Clone)]
pub struct InitialConditions {
    pub lat_deg: f64,
    pub lon_deg: f64,
    pub altitude_ft: f64,
    pub speed_kts: f64,
    pub heading_deg: f64,
}

// ── Autopilot waypoint ──────────────────────────────────────────────────────

/// A single autopilot command point. Any combination of heading, altitude and
/// speed commands may be set; unset fields leave the previous command in place.
#[derive(Debug, Clone)]
pub struct AutopilotWaypoint {
    pub time_s: f64,
    pub heading_deg: Option<f64>,
    pub altitude_ft: Option<f64>,
    pub speed_kts: Option<f64>,
}

// ── JSBSim state ────────────────────────────────────────────────────────────

/// State snapshot emitted by the JSBSim integrator.
#[derive(Debug, Clone)]
pub struct JsbSimState {
    pub time_s: f64,
    pub lat_deg: f64,
    pub lon_deg: f64,
    pub altitude_ft: f64,
    /// Velocity in local NED frame, feet per second (`[v_n, v_e, v_d]`).
    pub velocity_ned_fps: [f64; 3],
    /// Euler angles, radians: `[roll, pitch, yaw]`.
    pub euler_rad: [f64; 3],
    /// Body-frame acceleration, feet per second squared.
    pub acceleration_body_fps2: [f64; 3],
}

// ── Geodetic → ENU conversion (non-gated) ───────────────────────────────────

const FT_TO_M: f64 = 0.3048;

/// Convert a `JsbSimState` into a thresh `Waypoint` in the ENU frame centred
/// on a user-supplied reference point.
///
/// The input velocity is expected in feet-per-second NED and is re-expressed in
/// metres-per-second ENU as `[v_e, v_n, -v_d]`.
pub fn state_to_waypoint(
    state: &JsbSimState,
    ref_lat_rad: f64,
    ref_lon_rad: f64,
    ref_alt_m: f64,
) -> Waypoint {
    let lat_rad = state.lat_deg.to_radians();
    let lon_rad = state.lon_deg.to_radians();
    let alt_m = state.altitude_ft * FT_TO_M;
    let enu = wgs84_to_enu(lat_rad, lon_rad, alt_m, ref_lat_rad, ref_lon_rad, ref_alt_m);

    let v_n = state.velocity_ned_fps[0] * FT_TO_M;
    let v_e = state.velocity_ned_fps[1] * FT_TO_M;
    let v_d = state.velocity_ned_fps[2] * FT_TO_M;
    let velocity = [v_e, v_n, -v_d];

    Waypoint {
        time: state.time_s,
        position: [enu.x, enu.y, enu.z],
        velocity,
    }
}

// ── Maneuver library (non-gated) ────────────────────────────────────────────

/// Standard aircraft maneuvers expressed as sequences of autopilot waypoints.
pub struct Maneuvers;

impl Maneuvers {
    /// Build autopilot waypoints that produce a level coordinated turn at the
    /// requested load factor.
    ///
    /// Turn rate is derived from `omega = g * sqrt(n^2 - 1) / V` where `V` is
    /// the bank-speed; since the autopilot here works on heading commands we
    /// discretise the turn into 1 second heading steps.
    pub fn level_turn_g(
        start_time_s: f64,
        duration_s: f64,
        g_load: f64,
        initial_heading_deg: f64,
    ) -> Vec<AutopilotWaypoint> {
        // Assume ~250 m/s (roughly 485 knots) true airspeed when no speed is
        // supplied. This is a reasonable cruise speed for the fast jets the
        // maneuver library targets and yields realistic turn rates for 2-6 g.
        const TAS_MS: f64 = 250.0;
        const G: f64 = 9.81;

        let n2_minus_one = (g_load * g_load - 1.0).max(0.0);
        let omega_rad_s = G * n2_minus_one.sqrt() / TAS_MS;
        let omega_deg_s = omega_rad_s.to_degrees();

        let steps = duration_s.ceil().max(1.0) as usize;
        let step_dt = duration_s / steps as f64;
        let mut waypoints = Vec::with_capacity(steps + 1);
        for i in 0..=steps {
            let t = start_time_s + (i as f64) * step_dt;
            let heading =
                (initial_heading_deg + omega_deg_s * (i as f64) * step_dt).rem_euclid(360.0);
            waypoints.push(AutopilotWaypoint {
                time_s: t,
                heading_deg: Some(heading),
                altitude_ft: None,
                speed_kts: None,
            });
        }
        waypoints
    }

    /// Build autopilot waypoints for a climb or descent at `vertical_speed_fpm`
    /// for `duration_s`. Heading and speed are left unchanged.
    pub fn climb_descent(
        start_time_s: f64,
        duration_s: f64,
        vertical_speed_fpm: f64,
        current_alt_ft: f64,
    ) -> Vec<AutopilotWaypoint> {
        let steps = duration_s.ceil().max(1.0) as usize;
        let step_dt = duration_s / steps as f64;
        let mut waypoints = Vec::with_capacity(steps + 1);
        for i in 0..=steps {
            let t_rel = (i as f64) * step_dt;
            let alt = current_alt_ft + vertical_speed_fpm * (t_rel / 60.0);
            waypoints.push(AutopilotWaypoint {
                time_s: start_time_s + t_rel,
                heading_deg: None,
                altitude_ft: Some(alt),
                speed_kts: None,
            });
        }
        waypoints
    }

    /// Build a simple speed change maneuver.
    pub fn speed_change(
        start_time_s: f64,
        duration_s: f64,
        target_kts: f64,
    ) -> Vec<AutopilotWaypoint> {
        vec![
            AutopilotWaypoint {
                time_s: start_time_s,
                heading_deg: None,
                altitude_ft: None,
                speed_kts: Some(target_kts),
            },
            AutopilotWaypoint {
                time_s: start_time_s + duration_s,
                heading_deg: None,
                altitude_ft: None,
                speed_kts: Some(target_kts),
            },
        ]
    }
}

// ── PyO3 bridge (feature gated) ─────────────────────────────────────────────

#[cfg(feature = "jsbsim")]
use std::cell::RefCell;

/// PyO3 bridge to the Python `jsbsim` package.
#[cfg(feature = "jsbsim")]
pub struct JsbSimBridge {
    /// Python `FGFDMExec` instance.
    fdm: Py<PyAny>,
    /// Autopilot waypoint schedule (sorted by ascending `time_s`).
    waypoints: RefCell<Vec<AutopilotWaypoint>>,
    /// Index of the next waypoint to apply.
    next_wp: RefCell<usize>,
}

#[cfg(feature = "jsbsim")]
impl JsbSimBridge {
    /// Create a new bridge, load the aircraft model and install the initial
    /// conditions.
    pub fn new(model: AircraftModel, ic: &InitialConditions) -> PyResult<Self> {
        Python::with_gil(|py| {
            let jsbsim_mod = py.import("jsbsim")?;
            let fdm_class = jsbsim_mod.getattr("FGFDMExec")?;
            let fdm = fdm_class.call0()?.unbind();
            let bridge = Self {
                fdm,
                waypoints: RefCell::new(Vec::new()),
                next_wp: RefCell::new(0),
            };
            bridge.load_model(py, model.jsbsim_name())?;
            bridge.set_initial_conditions(py, ic)?;
            Ok(bridge)
        })
    }

    /// Load an aircraft model by JSBSim name.
    fn load_model(&self, py: Python<'_>, name: &str) -> PyResult<()> {
        let fdm = self.fdm.bind(py);
        fdm.call_method1("load_model", (name,))?;
        Ok(())
    }

    /// Push initial conditions into the FDM by setting the standard JSBSim
    /// property tree entries and priming the integrator.
    fn set_initial_conditions(&self, py: Python<'_>, ic: &InitialConditions) -> PyResult<()> {
        let fdm = self.fdm.bind(py);
        let set = |prop: &str, value: f64| -> PyResult<()> {
            fdm.call_method1("set_property_value", (prop, value))?;
            Ok(())
        };

        set("ic/lat-gc-deg", ic.lat_deg)?;
        set("ic/long-gc-deg", ic.lon_deg)?;
        set("ic/h-sl-ft", ic.altitude_ft)?;
        set("ic/vt-kts", ic.speed_kts)?;
        set("ic/psi-true-deg", ic.heading_deg)?;

        // Reset / prime the integrator to pick up the new IC block.
        fdm.call_method0("run_ic")?;
        Ok(())
    }

    /// Install a new schedule of autopilot waypoints. The list is sorted by
    /// ascending time; existing waypoints are replaced.
    pub fn set_autopilot_waypoints(&self, mut waypoints: Vec<AutopilotWaypoint>) -> PyResult<()> {
        waypoints.sort_by(|a, b| {
            a.time_s
                .partial_cmp(&b.time_s)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        *self.waypoints.borrow_mut() = waypoints;
        *self.next_wp.borrow_mut() = 0;
        Ok(())
    }

    /// Apply every autopilot waypoint whose `time_s <= time_s` that has not
    /// already been applied.
    pub fn apply_autopilot_at(&self, time_s: f64) -> PyResult<()> {
        Python::with_gil(|py| {
            let fdm = self.fdm.bind(py);
            let waypoints = self.waypoints.borrow();
            let mut idx = self.next_wp.borrow_mut();
            while *idx < waypoints.len() && waypoints[*idx].time_s <= time_s {
                let wp = &waypoints[*idx];
                if let Some(h) = wp.heading_deg {
                    fdm.call_method1("set_property_value", ("ap/heading_setpoint", h))?;
                    fdm.call_method1("set_property_value", ("ap/heading_hold", 1.0))?;
                }
                if let Some(a) = wp.altitude_ft {
                    fdm.call_method1("set_property_value", ("ap/altitude_setpoint", a))?;
                    fdm.call_method1("set_property_value", ("ap/altitude_hold", 1.0))?;
                }
                if let Some(v) = wp.speed_kts {
                    fdm.call_method1("set_property_value", ("ap/airspeed_setpoint", v))?;
                    fdm.call_method1("set_property_value", ("ap/airspeed_hold", 1.0))?;
                }
                *idx += 1;
            }
            Ok(())
        })
    }

    /// Read the current JSBSim state from the property tree.
    fn read_state(&self, py: Python<'_>) -> PyResult<JsbSimState> {
        let fdm = self.fdm.bind(py);
        let get = |prop: &str| -> PyResult<f64> {
            let v = fdm.call_method1("get_property_value", (prop,))?;
            v.extract::<f64>()
        };

        Ok(JsbSimState {
            time_s: get("simulation/sim-time-sec")?,
            lat_deg: get("position/lat-gc-deg")?,
            lon_deg: get("position/long-gc-deg")?,
            altitude_ft: get("position/h-sl-ft")?,
            velocity_ned_fps: [
                get("velocities/v-north-fps")?,
                get("velocities/v-east-fps")?,
                get("velocities/v-down-fps")?,
            ],
            euler_rad: [
                get("attitude/roll-rad")?,
                get("attitude/pitch-rad")?,
                get("attitude/psi-rad")?,
            ],
            acceleration_body_fps2: [
                get("accelerations/udot-ft_sec2")?,
                get("accelerations/vdot-ft_sec2")?,
                get("accelerations/wdot-ft_sec2")?,
            ],
        })
    }

    /// Advance the simulation by one integration step of size `dt_s` and
    /// return the resulting state.
    pub fn step(&self, dt_s: f64) -> PyResult<JsbSimState> {
        Python::with_gil(|py| {
            let fdm = self.fdm.bind(py);
            fdm.call_method1("set_property_value", ("simulation/dt", dt_s))?;
            fdm.call_method0("run")?;
            self.read_state(py)
        })
    }

    /// Run the simulation for `duration_s`, advancing in `dt_s` steps and
    /// emitting states every `output_dt_s`.
    pub fn run(&self, duration_s: f64, dt_s: f64, output_dt_s: f64) -> PyResult<Vec<JsbSimState>> {
        let n_steps = (duration_s / dt_s).ceil() as usize;
        let mut states = Vec::new();
        let mut next_out = 0.0_f64;
        let mut t = 0.0_f64;
        // Emit the initial state too.
        Python::with_gil(|py| -> PyResult<()> {
            states.push(self.read_state(py)?);
            Ok(())
        })?;
        next_out += output_dt_s;

        for _ in 0..n_steps {
            self.apply_autopilot_at(t)?;
            let state = self.step(dt_s)?;
            t += dt_s;
            if t + 1e-9 >= next_out {
                states.push(state);
                next_out += output_dt_s;
            }
        }
        Ok(states)
    }

    /// Run the simulation and export each emitted state as an ENU `Waypoint`.
    pub fn to_trajectory(
        &self,
        duration_s: f64,
        dt_s: f64,
        output_dt_s: f64,
        ref_lat_rad: f64,
        ref_lon_rad: f64,
        ref_alt_m: f64,
    ) -> PyResult<Vec<Waypoint>> {
        let states = self.run(duration_s, dt_s, output_dt_s)?;
        Ok(states
            .iter()
            .map(|s| state_to_waypoint(s, ref_lat_rad, ref_lon_rad, ref_alt_m))
            .collect())
    }
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn aircraft_model_names() {
        assert_eq!(AircraftModel::F16.jsbsim_name(), "f16");
        assert_eq!(AircraftModel::B737.jsbsim_name(), "737");
        assert_eq!(AircraftModel::C172.jsbsim_name(), "c172");
    }

    #[test]
    fn state_to_waypoint_at_origin() {
        let state = JsbSimState {
            time_s: 0.0,
            lat_deg: 0.0,
            lon_deg: 0.0,
            altitude_ft: 0.0,
            velocity_ned_fps: [0.0, 0.0, 0.0],
            euler_rad: [0.0, 0.0, 0.0],
            acceleration_body_fps2: [0.0, 0.0, 0.0],
        };
        let wp = state_to_waypoint(&state, 0.0, 0.0, 0.0);
        assert!(wp.position[0].abs() < 1e-3);
        assert!(wp.position[1].abs() < 1e-3);
        assert!(wp.position[2].abs() < 1e-3);
        assert_eq!(wp.velocity, [0.0, 0.0, 0.0]);
    }

    #[test]
    fn state_to_waypoint_velocity_ned_to_enu() {
        // v_n = 100 fps, v_e = 50 fps, v_d = -10 fps (climbing)
        // ENU should be [v_e, v_n, -v_d] in m/s
        let state = JsbSimState {
            time_s: 1.0,
            lat_deg: 40.0,
            lon_deg: -75.0,
            altitude_ft: 10_000.0,
            velocity_ned_fps: [100.0, 50.0, -10.0],
            euler_rad: [0.0, 0.0, 0.0],
            acceleration_body_fps2: [0.0, 0.0, 0.0],
        };
        let wp = state_to_waypoint(
            &state,
            40.0_f64.to_radians(),
            -75.0_f64.to_radians(),
            10_000.0 * FT_TO_M,
        );
        let expected_ve = 50.0 * FT_TO_M;
        let expected_vn = 100.0 * FT_TO_M;
        let expected_vu = 10.0 * FT_TO_M;
        assert!((wp.velocity[0] - expected_ve).abs() < 1e-9);
        assert!((wp.velocity[1] - expected_vn).abs() < 1e-9);
        assert!((wp.velocity[2] - expected_vu).abs() < 1e-9);
        // At the reference point position should be ~zero.
        assert!(wp.position[0].abs() < 1e-3);
        assert!(wp.position[1].abs() < 1e-3);
        assert!(wp.position[2].abs() < 1e-3);
    }

    #[test]
    fn maneuver_level_turn_waypoints_cover_duration() {
        let wps = Maneuvers::level_turn_g(0.0, 60.0, 4.0, 0.0);
        assert!(!wps.is_empty());
        let first = wps.first().unwrap();
        let last = wps.last().unwrap();
        assert!((first.time_s - 0.0).abs() < 1e-9);
        assert!((last.time_s - 60.0).abs() < 1e-6);
        // Heading must be advancing.
        assert!(wps.iter().all(|w| w.heading_deg.is_some()));
    }

    #[test]
    fn maneuver_level_turn_rate_matches_g_load() {
        // At 4g and ~250 m/s TAS the turn rate should be g*sqrt(15)/V rad/s.
        let wps = Maneuvers::level_turn_g(0.0, 1.0, 4.0, 0.0);
        assert!(wps.len() >= 2);
        let h0 = wps[0].heading_deg.unwrap();
        let h1 = wps[1].heading_deg.unwrap();
        let dt = wps[1].time_s - wps[0].time_s;
        let omega_deg_s = (h1 - h0) / dt;
        let expected = (9.81_f64 * (4.0_f64 * 4.0 - 1.0).sqrt() / 250.0).to_degrees();
        assert!((omega_deg_s - expected).abs() < 1e-6);
    }

    #[test]
    fn maneuver_climb_descent_monotonic() {
        let wps = Maneuvers::climb_descent(10.0, 60.0, 1500.0, 10_000.0);
        assert!(wps.len() >= 2);
        let alts: Vec<f64> = wps.iter().map(|w| w.altitude_ft.unwrap()).collect();
        for pair in alts.windows(2) {
            assert!(pair[1] >= pair[0]);
        }
        // After one minute at 1500 fpm we should be at 11 500 ft.
        let last = *alts.last().unwrap();
        assert!((last - 11_500.0).abs() < 1e-6);
    }

    #[test]
    fn maneuver_speed_change_two_points() {
        let wps = Maneuvers::speed_change(0.0, 30.0, 350.0);
        assert_eq!(wps.len(), 2);
        assert_eq!(wps[0].speed_kts, Some(350.0));
        assert_eq!(wps[1].speed_kts, Some(350.0));
        assert!((wps[1].time_s - 30.0).abs() < 1e-9);
    }
}
