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
            let n_steps = (seg.duration / self.dt).ceil() as usize;
            for _ in 0..n_steps {
                match &seg.segment_type {
                    SegmentType::Cv => {
                        pos += vel * self.dt;
                    }
                    SegmentType::Ca { acceleration } => {
                        let acc = Vector3::from_row_slice(acceleration);
                        pos += vel * self.dt + acc * 0.5 * self.dt * self.dt;
                        vel += acc * self.dt;
                    }
                    SegmentType::Ctrv { turn_rate } => {
                        let omega = *turn_rate;
                        let speed = (vel.x * vel.x + vel.y * vel.y).sqrt();
                        let heading = vel.y.atan2(vel.x);
                        let new_heading = heading + omega * self.dt;

                        if omega.abs() < 1e-8 {
                            pos += vel * self.dt;
                        } else {
                            let r = speed / omega;
                            pos.x += r * ((new_heading).sin() - heading.sin());
                            pos.y += r * (-(new_heading).cos() + heading.cos());
                            pos.z += vel.z * self.dt;
                        }
                        vel.x = speed * new_heading.cos();
                        vel.y = speed * new_heading.sin();
                    }
                    SegmentType::Ballistic { drag_coefficient } => {
                        let drag = *drag_coefficient;
                        let speed = vel.norm();
                        let drag_force = if speed > 1e-10 {
                            vel * (-drag * speed)
                        } else {
                            Vector3::zeros()
                        };
                        let acc = drag_force + Vector3::new(0.0, 0.0, -GRAVITY);
                        pos += vel * self.dt + acc * 0.5 * self.dt * self.dt;
                        vel += acc * self.dt;
                    }
                }
                time += self.dt;
                waypoints.push(Waypoint {
                    time,
                    position: [pos.x, pos.y, pos.z],
                    velocity: [vel.x, vel.y, vel.z],
                });
            }
        }
        waypoints
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
