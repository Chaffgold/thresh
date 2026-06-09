//! Trajectory generation: CV, CA, CTRV, ballistic, and kinematic-bicycle
//! segments with stitching.

use nalgebra::Vector3;
use serde::{Deserialize, Serialize};

/// A single trajectory waypoint (ground truth).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Waypoint {
    pub time: f64,
    pub position: [f64; 3],
    pub velocity: [f64; 3],
}

impl Waypoint {
    /// Linear interpolation of position and velocity at `t` between `prev` and `next`.
    ///
    /// Used by [`crate::radar_trajectory`] to upsample sparse (e.g. 1 Hz ADS-B)
    /// trajectories to the synth's sample rate. The interpolation is
    /// constant-velocity inside the segment: position is linearly interpolated
    /// in time, and velocity is linearly interpolated between the endpoint
    /// velocities.
    ///
    /// If `t` is outside `[prev.time, next.time]` the result is the nearer
    /// endpoint (clamped, not extrapolated). Both endpoints are returned
    /// unchanged when their times are equal (avoids divide-by-zero).
    pub fn interpolate(prev: &Waypoint, next: &Waypoint, t: f64) -> Waypoint {
        if t <= prev.time || (next.time - prev.time).abs() < 1e-9 {
            return prev.clone();
        }
        if t >= next.time {
            return next.clone();
        }
        let alpha = (t - prev.time) / (next.time - prev.time);
        let lerp = |a: f64, b: f64| a + (b - a) * alpha;
        Waypoint {
            time: t,
            position: [
                lerp(prev.position[0], next.position[0]),
                lerp(prev.position[1], next.position[1]),
                lerp(prev.position[2], next.position[2]),
            ],
            velocity: [
                lerp(prev.velocity[0], next.velocity[0]),
                lerp(prev.velocity[1], next.velocity[1]),
                lerp(prev.velocity[2], next.velocity[2]),
            ],
        }
    }
}

/// A trajectory segment type.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SegmentType {
    /// Constant velocity.
    Cv,
    /// Constant acceleration.
    Ca { acceleration: [f64; 3] },
    /// Constant turn rate and velocity (2D).
    Ctrv { turn_rate: f64 },
    /// Ballistic (gravity + optional drag).
    Ballistic { drag_coefficient: f64 },
    /// 2-DOF kinematic bicycle model (automotive-tracking-pipeline, task 3.1).
    ///
    /// Ground-vehicle steering kinematics: the heading turns at
    /// `ω = v · tan(steering_angle) / wheelbase` while the speed changes by
    /// `acceleration` (speed is clamped at zero — no reversing through a
    /// braking segment). Altitude follows the vertical velocity like `Ctrv`
    /// (set `vz = 0` for flat roads). With `steering_angle = 0` this reduces
    /// to in-lane constant acceleration; with constant speed it traces a
    /// circle of radius `wheelbase / tan(steering_angle)`.
    KinematicBicycle {
        /// Front-wheel steering angle in radians (positive = left).
        steering_angle: f64,
        /// Longitudinal acceleration in m/s² (negative = braking).
        acceleration: f64,
        /// Wheelbase in metres (typical car ≈ 2.7).
        wheelbase: f64,
    },
}

/// A trajectory segment with duration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Segment {
    pub segment_type: SegmentType,
    pub duration: f64,
}

/// A complete multi-segment trajectory.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Trajectory {
    pub target_id: u32,
    pub initial_position: [f64; 3],
    pub initial_velocity: [f64; 3],
    pub segments: Vec<Segment>,
    pub dt: f64,
}

const GRAVITY: f64 = 9.81;

/// Advance a single kinematic step of one segment type, updating `pos` and
/// `vel` in place.
fn advance_segment_step(
    segment_type: &SegmentType,
    pos: &mut Vector3<f64>,
    vel: &mut Vector3<f64>,
    dt: f64,
) {
    match segment_type {
        SegmentType::Cv => step_cv(pos, vel, dt),
        SegmentType::Ca { acceleration } => step_ca(pos, vel, acceleration, dt),
        SegmentType::Ctrv { turn_rate } => step_ctrv(pos, vel, *turn_rate, dt),
        SegmentType::Ballistic { drag_coefficient } => {
            step_ballistic(pos, vel, *drag_coefficient, dt)
        }
        SegmentType::KinematicBicycle {
            steering_angle,
            acceleration,
            wheelbase,
        } => step_bicycle(pos, vel, *steering_angle, *acceleration, *wheelbase, dt),
    }
}

fn step_cv(pos: &mut Vector3<f64>, vel: &Vector3<f64>, dt: f64) {
    *pos += *vel * dt;
}

fn step_ca(pos: &mut Vector3<f64>, vel: &mut Vector3<f64>, acceleration: &[f64; 3], dt: f64) {
    let acc = Vector3::from_row_slice(acceleration);
    *pos += *vel * dt + acc * 0.5 * dt * dt;
    *vel += acc * dt;
}

fn step_ctrv(pos: &mut Vector3<f64>, vel: &mut Vector3<f64>, omega: f64, dt: f64) {
    let speed = (vel.x * vel.x + vel.y * vel.y).sqrt();
    let heading = vel.y.atan2(vel.x);
    let new_heading = heading + omega * dt;

    if omega.abs() < 1e-8 {
        *pos += *vel * dt;
    } else {
        let r = speed / omega;
        pos.x += r * (new_heading.sin() - heading.sin());
        pos.y += r * (-new_heading.cos() + heading.cos());
        pos.z += vel.z * dt;
    }
    vel.x = speed * new_heading.cos();
    vel.y = speed * new_heading.sin();
}

/// One 2-DOF kinematic-bicycle step. Heading and speed are derived from the
/// planar velocity (like [`step_ctrv`]); the heading advances at the bicycle
/// turn rate `ω = v · tan(δ) / L` evaluated mid-step, and the position
/// integrates with the mid-step speed and heading (midpoint rule, accurate to
/// O(dt²) for the synth's small `dt`).
fn step_bicycle(
    pos: &mut Vector3<f64>,
    vel: &mut Vector3<f64>,
    steering_angle: f64,
    acceleration: f64,
    wheelbase: f64,
    dt: f64,
) {
    let speed = (vel.x * vel.x + vel.y * vel.y).sqrt();
    let heading = vel.y.atan2(vel.x);

    let new_speed = (speed + acceleration * dt).max(0.0);
    let mid_speed = 0.5 * (speed + new_speed);
    let omega = if wheelbase.abs() > 1e-9 {
        mid_speed * steering_angle.tan() / wheelbase
    } else {
        0.0
    };
    let new_heading = heading + omega * dt;
    let mid_heading = heading + 0.5 * omega * dt;

    pos.x += mid_speed * mid_heading.cos() * dt;
    pos.y += mid_speed * mid_heading.sin() * dt;
    pos.z += vel.z * dt;
    vel.x = new_speed * new_heading.cos();
    vel.y = new_speed * new_heading.sin();
}

fn step_ballistic(pos: &mut Vector3<f64>, vel: &mut Vector3<f64>, drag: f64, dt: f64) {
    let speed = vel.norm();
    let drag_force = if speed > 1e-10 {
        *vel * (-drag * speed)
    } else {
        Vector3::zeros()
    };
    let acc = drag_force + Vector3::new(0.0, 0.0, -GRAVITY);
    *pos += *vel * dt + acc * 0.5 * dt * dt;
    *vel += acc * dt;
}

impl Trajectory {
    /// Generate all waypoints for this trajectory.
    pub fn generate(&self) -> Vec<Waypoint> {
        let mut waypoints = Vec::new();
        let mut pos = Vector3::from_row_slice(&self.initial_position);
        let mut vel = Vector3::from_row_slice(&self.initial_velocity);
        let mut time = 0.0;

        waypoints.push(Waypoint {
            time,
            position: [pos.x, pos.y, pos.z],
            velocity: [vel.x, vel.y, vel.z],
        });

        for seg in &self.segments {
            self.generate_segment(seg, &mut pos, &mut vel, &mut time, &mut waypoints);
        }
        waypoints
    }

    /// Generate waypoints for a single segment, appending them to `waypoints`
    /// and advancing `pos`, `vel`, and `time` in place.
    fn generate_segment(
        &self,
        seg: &Segment,
        pos: &mut Vector3<f64>,
        vel: &mut Vector3<f64>,
        time: &mut f64,
        waypoints: &mut Vec<Waypoint>,
    ) {
        let n_steps = (seg.duration / self.dt).ceil() as usize;
        for _ in 0..n_steps {
            advance_segment_step(&seg.segment_type, pos, vel, self.dt);
            *time += self.dt;
            waypoints.push(Waypoint {
                time: *time,
                position: [pos.x, pos.y, pos.z],
                velocity: [vel.x, vel.y, vel.z],
            });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cv_trajectory_analytical() {
        let traj = Trajectory {
            target_id: 0,
            initial_position: [0.0, 0.0, 1000.0],
            initial_velocity: [250.0, 0.0, 0.0],
            segments: vec![Segment {
                segment_type: SegmentType::Cv,
                duration: 10.0,
            }],
            dt: 1.0,
        };
        let wps = traj.generate();
        // At t=10, x should be 250*10 = 2500
        let last = &wps[wps.len() - 1];
        assert!((last.position[0] - 2500.0).abs() < 1.0);
        assert!((last.position[2] - 1000.0).abs() < 1e-8);
    }

    #[test]
    fn ca_trajectory() {
        let traj = Trajectory {
            target_id: 1,
            initial_position: [0.0, 0.0, 0.0],
            initial_velocity: [0.0, 0.0, 0.0],
            segments: vec![Segment {
                segment_type: SegmentType::Ca {
                    acceleration: [10.0, 0.0, 0.0],
                },
                duration: 10.0,
            }],
            dt: 0.1,
        };
        let wps = traj.generate();
        let last = &wps[wps.len() - 1];
        // x = 0.5*10*100 = 500
        assert!((last.position[0] - 500.0).abs() < 5.0);
    }

    #[test]
    fn ballistic_falls() {
        let traj = Trajectory {
            target_id: 2,
            initial_position: [0.0, 0.0, 10000.0],
            initial_velocity: [500.0, 0.0, 0.0],
            segments: vec![Segment {
                segment_type: SegmentType::Ballistic {
                    drag_coefficient: 0.0,
                },
                duration: 10.0,
            }],
            dt: 0.1,
        };
        let wps = traj.generate();
        let last = &wps[wps.len() - 1];
        // z should decrease due to gravity
        assert!(last.position[2] < 10000.0);
        // z = 10000 - 0.5*g*t^2 = 10000 - 490.5 ≈ 9509.5
        assert!((last.position[2] - 9509.5).abs() < 10.0);
    }

    #[test]
    fn bicycle_zero_steering_is_straight_acceleration() {
        // δ=0: in-lane acceleration, x = v0·t + a·t²/2.
        let traj = Trajectory {
            target_id: 10,
            initial_position: [0.0, 0.0, 0.0],
            initial_velocity: [10.0, 0.0, 0.0],
            segments: vec![Segment {
                segment_type: SegmentType::KinematicBicycle {
                    steering_angle: 0.0,
                    acceleration: 2.0,
                    wheelbase: 2.7,
                },
                duration: 5.0,
            }],
            dt: 0.01,
        };
        let last = traj.generate().pop().unwrap();
        // x = 10*5 + 0.5*2*25 = 75; v = 10 + 2*5 = 20
        assert!(
            (last.position[0] - 75.0).abs() < 0.1,
            "{}",
            last.position[0]
        );
        assert!(last.position[1].abs() < 1e-9);
        assert!((last.velocity[0] - 20.0).abs() < 1e-6);
    }

    #[test]
    fn bicycle_constant_steering_traces_circle() {
        // Constant speed v and steering δ trace a circle of radius L/tan(δ):
        // after time T the vehicle is at angle θ = vT/R around the circle.
        let (v, delta, wheelbase) = (10.0, 0.1_f64, 2.7);
        let radius = wheelbase / delta.tan();
        let traj = Trajectory {
            target_id: 11,
            initial_position: [0.0, 0.0, 0.0],
            initial_velocity: [v, 0.0, 0.0],
            segments: vec![Segment {
                segment_type: SegmentType::KinematicBicycle {
                    steering_angle: delta,
                    acceleration: 0.0,
                    wheelbase,
                },
                duration: 5.0,
            }],
            dt: 0.001,
        };
        let wps = traj.generate();
        let last = wps.last().unwrap();
        // Circle centred at (0, R) starting eastbound and turning left.
        let theta = v * 5.0 / radius;
        let expect = [radius * theta.sin(), radius * (1.0 - theta.cos())];
        assert!(
            (last.position[0] - expect[0]).abs() < 0.05,
            "x: {} vs {}",
            last.position[0],
            expect[0]
        );
        assert!(
            (last.position[1] - expect[1]).abs() < 0.05,
            "y: {} vs {}",
            last.position[1],
            expect[1]
        );
        // Speed is preserved with zero acceleration.
        let sp = (last.velocity[0].powi(2) + last.velocity[1].powi(2)).sqrt();
        assert!((sp - v).abs() < 1e-6);
    }

    #[test]
    fn bicycle_braking_clamps_at_standstill() {
        // Hard braking from 5 m/s for 10 s must stop, not reverse.
        let traj = Trajectory {
            target_id: 12,
            initial_position: [0.0, 0.0, 0.0],
            initial_velocity: [5.0, 0.0, 0.0],
            segments: vec![Segment {
                segment_type: SegmentType::KinematicBicycle {
                    steering_angle: 0.0,
                    acceleration: -2.0,
                    wheelbase: 2.7,
                },
                duration: 10.0,
            }],
            dt: 0.01,
        };
        let wps = traj.generate();
        let last = wps.last().unwrap();
        let sp = (last.velocity[0].powi(2) + last.velocity[1].powi(2)).sqrt();
        assert!(sp < 1e-9, "vehicle should be stopped, speed {sp}");
        // Distance to stop: v²/(2a) = 25/4 = 6.25 m.
        assert!(
            (last.position[0] - 6.25).abs() < 0.1,
            "{}",
            last.position[0]
        );
        // x must be monotonically non-decreasing (never reverses).
        for w in wps.windows(2) {
            assert!(w[1].position[0] >= w[0].position[0] - 1e-9);
        }
    }

    #[test]
    fn multi_segment() {
        let traj = Trajectory {
            target_id: 3,
            initial_position: [0.0, 0.0, 5000.0],
            initial_velocity: [200.0, 0.0, 0.0],
            segments: vec![
                Segment {
                    segment_type: SegmentType::Cv,
                    duration: 5.0,
                },
                Segment {
                    segment_type: SegmentType::Ctrv { turn_rate: 0.1 },
                    duration: 5.0,
                },
            ],
            dt: 0.5,
        };
        let wps = traj.generate();
        assert!(wps.len() > 20);
    }
}
