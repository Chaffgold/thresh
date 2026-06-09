//! Road-scenario presets for automotive tracking (automotive-tracking-pipeline,
//! tasks 3.2/3.3).
//!
//! Each preset returns a set of [`RoadAgent`]s — a class-tagged
//! [`Trajectory`] per road user — in a local world ENU frame (metres; `z = 0`
//! is the road plane). The presets are deterministic and physically plausible
//! by construction; [`plausibility`] provides the per-class speed/turn-rate
//! bounds used by the validation tests and available to downstream consumers.

use thresh_core::track::TargetClass;

use crate::trajectory::{Segment, SegmentType, Trajectory, Waypoint};

/// A class-tagged road user in a scenario.
#[derive(Debug, Clone)]
pub struct RoadAgent {
    /// Stable per-scenario agent id.
    pub agent_id: u32,
    /// Object class (drives per-class tracker heads downstream).
    pub class: TargetClass,
    /// The agent's ground-truth trajectory.
    pub trajectory: Trajectory,
}

impl RoadAgent {
    /// Generate the agent's ground-truth waypoints.
    pub fn waypoints(&self) -> Vec<Waypoint> {
        self.trajectory.generate()
    }
}

/// Typical car wheelbase in metres.
pub const CAR_WHEELBASE_M: f64 = 2.7;
/// Ground-truth sample interval for the presets, seconds.
pub const ROAD_DT_S: f64 = 0.1;

/// Lane-follow preset: `n_vehicles` cars in a single lane, 30 m headway,
/// cruising at 13 m/s (~47 km/h); the lead car brakes to a stop mid-scenario
/// while the followers continue (tests longitudinal dynamics + track
/// continuity through closing gaps).
pub fn lane_follow(n_vehicles: u32) -> Vec<RoadAgent> {
    (0..n_vehicles)
        .map(|i| {
            let lead = i == 0;
            let segments = if lead {
                vec![
                    Segment {
                        segment_type: SegmentType::KinematicBicycle {
                            steering_angle: 0.0,
                            acceleration: 0.0,
                            wheelbase: CAR_WHEELBASE_M,
                        },
                        duration: 5.0,
                    },
                    Segment {
                        // Brake at 3 m/s² to a stop (clamped at standstill).
                        segment_type: SegmentType::KinematicBicycle {
                            steering_angle: 0.0,
                            acceleration: -3.0,
                            wheelbase: CAR_WHEELBASE_M,
                        },
                        duration: 15.0,
                    },
                ]
            } else {
                vec![Segment {
                    segment_type: SegmentType::KinematicBicycle {
                        steering_angle: 0.0,
                        acceleration: 0.0,
                        wheelbase: CAR_WHEELBASE_M,
                    },
                    duration: 20.0,
                }]
            };
            RoadAgent {
                agent_id: i,
                class: TargetClass::Car,
                trajectory: Trajectory {
                    target_id: i,
                    // 30 m headway behind the lead car.
                    initial_position: [-(f64::from(i)) * 30.0, 0.0, 0.0],
                    initial_velocity: [13.0, 0.0, 0.0],
                    segments,
                    dt: ROAD_DT_S,
                },
            }
        })
        .collect()
}

/// Intersection preset: a car crossing eastbound, a truck crossing northbound,
/// a car turning left through the junction (bicycle steering), and a
/// pedestrian on the crosswalk — four classes of motion through a shared
/// region around the origin.
pub fn intersection() -> Vec<RoadAgent> {
    let straight_car = RoadAgent {
        agent_id: 0,
        class: TargetClass::Car,
        trajectory: Trajectory {
            target_id: 0,
            initial_position: [-80.0, -2.0, 0.0],
            initial_velocity: [12.0, 0.0, 0.0],
            segments: vec![Segment {
                segment_type: SegmentType::KinematicBicycle {
                    steering_angle: 0.0,
                    acceleration: 0.0,
                    wheelbase: CAR_WHEELBASE_M,
                },
                duration: 14.0,
            }],
            dt: ROAD_DT_S,
        },
    };
    let crossing_truck = RoadAgent {
        agent_id: 1,
        class: TargetClass::Truck,
        trajectory: Trajectory {
            target_id: 1,
            initial_position: [2.0, -70.0, 0.0],
            initial_velocity: [0.0, 9.0, 0.0],
            segments: vec![Segment {
                segment_type: SegmentType::KinematicBicycle {
                    steering_angle: 0.0,
                    acceleration: 0.0,
                    wheelbase: 4.5,
                },
                duration: 14.0,
            }],
            dt: ROAD_DT_S,
        },
    };
    // Approaches eastbound, slows, then turns left (north) through the box.
    let turning_car = RoadAgent {
        agent_id: 2,
        class: TargetClass::Car,
        trajectory: Trajectory {
            target_id: 2,
            initial_position: [-60.0, 2.0, 0.0],
            initial_velocity: [10.0, 0.0, 0.0],
            segments: vec![
                Segment {
                    // Decelerate on approach.
                    segment_type: SegmentType::KinematicBicycle {
                        steering_angle: 0.0,
                        acceleration: -1.0,
                        wheelbase: CAR_WHEELBASE_M,
                    },
                    duration: 4.0,
                },
                Segment {
                    // Left turn: δ such that the 90° arc completes in ~5 s at
                    // ~6 m/s (ω = v·tan(δ)/L ≈ 0.31 rad/s).
                    segment_type: SegmentType::KinematicBicycle {
                        steering_angle: 0.14,
                        acceleration: 0.0,
                        wheelbase: CAR_WHEELBASE_M,
                    },
                    duration: 5.0,
                },
                Segment {
                    // Accelerate out of the turn.
                    segment_type: SegmentType::KinematicBicycle {
                        steering_angle: 0.0,
                        acceleration: 1.5,
                        wheelbase: CAR_WHEELBASE_M,
                    },
                    duration: 5.0,
                },
            ],
            dt: ROAD_DT_S,
        },
    };
    let pedestrian = RoadAgent {
        agent_id: 3,
        class: TargetClass::Pedestrian,
        trajectory: Trajectory {
            target_id: 3,
            initial_position: [8.0, -6.0, 0.0],
            initial_velocity: [0.0, 1.4, 0.0],
            segments: vec![Segment {
                segment_type: SegmentType::Cv,
                duration: 14.0,
            }],
            dt: ROAD_DT_S,
        },
    };
    vec![straight_car, crossing_truck, turning_car, pedestrian]
}

/// Multi-agent mixed preset: lane traffic plus the intersection's vulnerable
/// road users and a cyclist along the shoulder — a denser scene mixing five
/// classes for association stress.
pub fn multi_agent() -> Vec<RoadAgent> {
    let mut agents = lane_follow(3);
    let base = agents.len() as u32;
    for (offset, mut agent) in intersection().into_iter().enumerate() {
        agent.agent_id = base + offset as u32;
        agent.trajectory.target_id = agent.agent_id;
        // Shift the intersection cluster 60 m north so the scenes overlap
        // without colliding.
        agent.trajectory.initial_position[1] += 60.0;
        agents.push(agent);
    }
    let id = agents.len() as u32;
    agents.push(RoadAgent {
        agent_id: id,
        class: TargetClass::Bicycle,
        trajectory: Trajectory {
            target_id: id,
            initial_position: [-40.0, 5.0, 0.0],
            initial_velocity: [5.0, 0.0, 0.0],
            segments: vec![Segment {
                segment_type: SegmentType::KinematicBicycle {
                    steering_angle: 0.02,
                    acceleration: 0.0,
                    wheelbase: 1.1,
                },
                duration: 20.0,
            }],
            dt: ROAD_DT_S,
        },
    });
    agents
}

/// Physical-plausibility bounds per class, used by the validation tests and
/// available to downstream consumers (e.g. gating sanity checks).
pub mod plausibility {
    use thresh_core::track::TargetClass;

    /// Maximum plausible speed (m/s) for a class in urban road scenarios.
    pub fn max_speed_mps(class: TargetClass) -> f64 {
        match class {
            TargetClass::Pedestrian => 3.0,
            TargetClass::Bicycle => 8.0,
            TargetClass::Motorcycle => 40.0,
            TargetClass::Car => 40.0,
            TargetClass::Truck | TargetClass::Bus => 30.0,
            // Aerospace / unknown: no road bound.
            _ => f64::INFINITY,
        }
    }

    /// Maximum plausible turn rate (rad/s) for a class in urban scenarios.
    pub fn max_turn_rate_radps(class: TargetClass) -> f64 {
        match class {
            TargetClass::Pedestrian => 3.0,
            TargetClass::Bicycle => 1.5,
            TargetClass::Motorcycle => 1.5,
            TargetClass::Car => 1.0,
            TargetClass::Truck | TargetClass::Bus => 0.7,
            _ => f64::INFINITY,
        }
    }
}

/// Peak planar speed over a waypoint sequence, m/s.
pub fn max_speed(waypoints: &[Waypoint]) -> f64 {
    waypoints
        .iter()
        .map(|w| (w.velocity[0].powi(2) + w.velocity[1].powi(2)).sqrt())
        .fold(0.0, f64::max)
}

/// Peak heading rate over a waypoint sequence, rad/s. Heading is undefined at
/// standstill, so steps where either endpoint is slower than 0.5 m/s are
/// skipped.
pub fn max_turn_rate(waypoints: &[Waypoint]) -> f64 {
    let mut max_rate = 0.0_f64;
    for w in waypoints.windows(2) {
        let dt = w[1].time - w[0].time;
        if dt <= 0.0 {
            continue;
        }
        let speed0 = (w[0].velocity[0].powi(2) + w[0].velocity[1].powi(2)).sqrt();
        let speed1 = (w[1].velocity[0].powi(2) + w[1].velocity[1].powi(2)).sqrt();
        if speed0 < 0.5 || speed1 < 0.5 {
            continue;
        }
        let h0 = w[0].velocity[1].atan2(w[0].velocity[0]);
        let h1 = w[1].velocity[1].atan2(w[1].velocity[0]);
        // Wrap the heading delta to (-π, π] in closed form.
        let dh = (h1 - h0).sin().atan2((h1 - h0).cos());
        max_rate = max_rate.max((dh / dt).abs());
    }
    max_rate
}

#[cfg(test)]
mod tests {
    use super::*;

    fn all_presets() -> Vec<(&'static str, Vec<RoadAgent>)> {
        vec![
            ("lane_follow", lane_follow(3)),
            ("intersection", intersection()),
            ("multi_agent", multi_agent()),
        ]
    }

    /// Task 3.3: every preset trajectory stays within its class's physical
    /// speed and turn-rate bounds.
    #[test]
    fn presets_are_physically_plausible() {
        for (name, agents) in all_presets() {
            assert!(!agents.is_empty());
            for agent in &agents {
                let wps = agent.waypoints();
                assert!(
                    wps.len() > 10,
                    "{name}/{}: too few waypoints",
                    agent.agent_id
                );
                let speed = max_speed(&wps);
                let turn = max_turn_rate(&wps);
                assert!(
                    speed <= plausibility::max_speed_mps(agent.class),
                    "{name}/agent {} ({:?}): speed {speed:.1} m/s exceeds class bound",
                    agent.agent_id,
                    agent.class,
                );
                assert!(
                    turn <= plausibility::max_turn_rate_radps(agent.class),
                    "{name}/agent {} ({:?}): turn rate {turn:.2} rad/s exceeds class bound",
                    agent.agent_id,
                    agent.class,
                );
            }
        }
    }

    /// Ground vehicles stay on the road plane.
    #[test]
    fn presets_stay_on_road_plane() {
        for (name, agents) in all_presets() {
            for agent in &agents {
                for w in agent.waypoints() {
                    assert!(
                        w.position[2].abs() < 1e-9,
                        "{name}/agent {}: left the road plane (z = {})",
                        agent.agent_id,
                        w.position[2]
                    );
                }
            }
        }
    }

    #[test]
    fn lane_follow_lead_car_stops_followers_do_not() {
        let agents = lane_follow(3);
        let lead = agents[0].waypoints();
        let last = lead.last().unwrap();
        let lead_final_speed = (last.velocity[0].powi(2) + last.velocity[1].powi(2)).sqrt();
        assert!(
            lead_final_speed < 1e-6,
            "lead must stop, got {lead_final_speed}"
        );
        for follower in &agents[1..] {
            let wps = follower.waypoints();
            let lf = wps.last().unwrap();
            let sp = (lf.velocity[0].powi(2) + lf.velocity[1].powi(2)).sqrt();
            assert!((sp - 13.0).abs() < 1e-6, "follower keeps cruising");
        }
    }

    #[test]
    fn intersection_turning_car_ends_up_northbound() {
        let agents = intersection();
        let turner = agents.iter().find(|a| a.agent_id == 2).unwrap();
        let wps = turner.waypoints();
        let last = wps.last().unwrap();
        let heading = last.velocity[1].atan2(last.velocity[0]);
        // After the 90° left turn the car should head ~north (π/2).
        assert!(
            (heading - std::f64::consts::FRAC_PI_2).abs() < 0.2,
            "final heading {heading:.2} rad should be ~π/2"
        );
    }

    #[test]
    fn multi_agent_ids_are_unique() {
        let agents = multi_agent();
        let mut ids: Vec<u32> = agents.iter().map(|a| a.agent_id).collect();
        ids.sort_unstable();
        ids.dedup();
        assert_eq!(ids.len(), agents.len(), "agent ids must be unique");
        // And the scene mixes at least 4 distinct classes.
        let mut classes: Vec<_> = agents.iter().map(|a| a.class).collect();
        classes.sort_by_key(|c| format!("{c:?}"));
        classes.dedup();
        assert!(classes.len() >= 4, "expected a mixed-class scene");
    }
}
