//! Trajectory generation: CV, CA, CTRV, ballistic segments with stitching.

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
