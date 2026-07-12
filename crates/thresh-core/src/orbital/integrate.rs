//! Fixed-step 4th-order Runge–Kutta integration for second-order dynamics.
//!
//! Generalized from the propagator in `thresh-synth/src/orbital.rs` (design
//! Decision 1 of the `orbital-ballistic-filter-models` change): the
//! acceleration is supplied as a closure over `(position, velocity)` so the
//! same integrator serves the synth truth generator and the filter motion
//! models with arbitrary force compositions.

use nalgebra::Vector3;

/// Compute one RK4 stage at time offset `time_offset_s` (in seconds) from the
/// current state, producing the pair `(k_r, k_v)` where `k_r` is the state
/// velocity at the offset point and `k_v` is the acceleration evaluated there.
pub fn rk4_stage(
    pos: &Vector3<f64>,
    vel: &Vector3<f64>,
    prev_kr: &Vector3<f64>,
    prev_kv: &Vector3<f64>,
    time_offset_s: f64,
    accel: impl Fn(&Vector3<f64>, &Vector3<f64>) -> Vector3<f64>,
) -> (Vector3<f64>, Vector3<f64>) {
    let stage_pos = Vector3::new(
        pos.x + time_offset_s * prev_kr.x,
        pos.y + time_offset_s * prev_kr.y,
        pos.z + time_offset_s * prev_kr.z,
    );
    let stage_vel = Vector3::new(
        vel.x + time_offset_s * prev_kv.x,
        vel.y + time_offset_s * prev_kv.y,
        vel.z + time_offset_s * prev_kv.z,
    );
    let acc = accel(&stage_pos, &stage_vel);
    (stage_vel, acc)
}

/// Take one RK4 step of size `dt` (seconds) under the acceleration closure,
/// returning the new `(position, velocity)`.
pub fn rk4_step(
    pos: &Vector3<f64>,
    vel: &Vector3<f64>,
    dt: f64,
    accel: impl Fn(&Vector3<f64>, &Vector3<f64>) -> Vector3<f64>,
) -> (Vector3<f64>, Vector3<f64>) {
    let zero = Vector3::zeros();
    // k1: evaluated at the current state (step = 0).
    let (k1r, k1v) = rk4_stage(pos, vel, &zero, &zero, 0.0, &accel);
    // k2: half step using k1.
    let (k2r, k2v) = rk4_stage(pos, vel, &k1r, &k1v, 0.5 * dt, &accel);
    // k3: half step using k2.
    let (k3r, k3v) = rk4_stage(pos, vel, &k2r, &k2v, 0.5 * dt, &accel);
    // k4: full step using k3.
    let (k4r, k4v) = rk4_stage(pos, vel, &k3r, &k3v, dt, &accel);

    let new_pos = Vector3::new(
        pos.x + dt / 6.0 * (k1r.x + 2.0 * k2r.x + 2.0 * k3r.x + k4r.x),
        pos.y + dt / 6.0 * (k1r.y + 2.0 * k2r.y + 2.0 * k3r.y + k4r.y),
        pos.z + dt / 6.0 * (k1r.z + 2.0 * k2r.z + 2.0 * k3r.z + k4r.z),
    );
    let new_vel = Vector3::new(
        vel.x + dt / 6.0 * (k1v.x + 2.0 * k2v.x + 2.0 * k3v.x + k4v.x),
        vel.y + dt / 6.0 * (k1v.y + 2.0 * k2v.y + 2.0 * k3v.y + k4v.y),
        vel.z + dt / 6.0 * (k1v.z + 2.0 * k2v.z + 2.0 * k3v.z + k4v.z),
    );

    (new_pos, new_vel)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::orbital::gravity::{GravityModel, two_body_acceleration};

    // ── rk4_stage ────────────────────────────────────────────────────────

    #[test]
    fn rk4_stage_at_zero_offset_evaluates_current_state() {
        let pos = Vector3::new(7_000_000.0, 0.0, 0.0);
        let vel = Vector3::new(0.0, 7_500.0, 0.0);
        let zero = Vector3::zeros();
        let accel = |p: &Vector3<f64>, _: &Vector3<f64>| {
            two_body_acceleration(p, &GravityModel::EARTH_WGS84)
        };

        let (kr, kv) = rk4_stage(&pos, &vel, &zero, &zero, 0.0, accel);

        assert_eq!(kr, vel);
        assert_eq!(kv, accel(&pos, &vel));
    }

    #[test]
    fn rk4_stage_advances_state_linearly_by_offset() {
        let pos = Vector3::new(1.0, 2.0, 3.0);
        let vel = Vector3::new(4.0, 5.0, 6.0);
        let prev_kr = Vector3::new(10.0, 20.0, 30.0);
        let prev_kv = Vector3::new(-1.0, -2.0, -3.0);
        let offset = 0.5;
        // Acceleration that just echoes the stage position, so the stage
        // point itself is observable through the returned k_v.
        let accel = |p: &Vector3<f64>, _: &Vector3<f64>| *p;

        let (kr, kv) = rk4_stage(&pos, &vel, &prev_kr, &prev_kv, offset, accel);

        assert_eq!(kr, vel + offset * prev_kv);
        assert_eq!(kv, pos + offset * prev_kr);
    }

    // ── rk4_step ─────────────────────────────────────────────────────────

    #[test]
    fn rk4_step_is_exact_for_constant_acceleration() {
        let pos = Vector3::new(1.0, -2.0, 3.0);
        let vel = Vector3::new(10.0, 0.0, -5.0);
        let a = Vector3::new(0.0, 0.0, -9.81);
        let dt = 2.0;

        let (new_pos, new_vel) = rk4_step(&pos, &vel, dt, |_, _| a);

        // RK4 integrates a quadratic exactly: p + v·dt + a·dt²/2, v + a·dt.
        let expected_pos = pos + dt * vel + 0.5 * dt * dt * a;
        let expected_vel = vel + dt * a;
        assert!((new_pos - expected_pos).norm() < 1e-12);
        assert!((new_vel - expected_vel).norm() < 1e-12);
    }

    #[test]
    fn rk4_step_conserves_two_body_energy_over_one_orbit() {
        let gravity = GravityModel::EARTH_WGS84;
        let r = gravity.equatorial_radius + 500_000.0;
        let v_circ = (gravity.mu / r).sqrt();
        let mut pos = Vector3::new(r, 0.0, 0.0);
        let mut vel = Vector3::new(0.0, v_circ, 0.0);
        let accel = |p: &Vector3<f64>, _: &Vector3<f64>| two_body_acceleration(p, &gravity);

        let energy =
            |p: &Vector3<f64>, v: &Vector3<f64>| 0.5 * v.norm_squared() - gravity.mu / p.norm();
        let energy_init = energy(&pos, &vel);

        // ~1 orbit at a 10 s step (matches the synth propagator's test regime).
        let dt = 10.0;
        for _ in 0..540 {
            let (p, v) = rk4_step(&pos, &vel, dt, accel);
            pos = p;
            vel = v;
        }

        let rel_err = (energy(&pos, &vel) - energy_init).abs() / energy_init.abs();
        assert!(rel_err < 1e-8, "Energy conservation error: {rel_err:.2e}");
    }
}
