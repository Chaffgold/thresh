//! Multi-target scenario composition and serialization.

use serde::{Deserialize, Serialize};
use thresh_core::measurement::Measurement;

use crate::measurement_gen::{RadarConfig, generate_radar};
use crate::trajectory::Trajectory;

/// A complete scenario: multiple targets + sensor configurations.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Scenario {
    pub name: String,
    pub trajectories: Vec<Trajectory>,
}

/// Ground truth entry for a single target at a single time.
#[derive(Debug, Clone)]
pub struct GroundTruth {
    pub target_id: u32,
    pub time: f64,
    pub position: [f64; 3],
    pub velocity: [f64; 3],
}

/// Time-ordered measurement with source info.
#[derive(Debug, Clone)]
pub struct TimedMeasurement {
    pub time: f64,
    pub measurement: Measurement,
}

/// Generate all ground truth and measurements for a scenario.
pub fn run_scenario(
    scenario: &Scenario,
    radar_config: &RadarConfig,
) -> (Vec<GroundTruth>, Vec<TimedMeasurement>) {
    let mut rng = rand::rng();
    let mut all_gt = Vec::new();
    let mut all_meas = Vec::new();

    for traj in &scenario.trajectories {
        let waypoints = traj.generate();

        for wp in &waypoints {
            all_gt.push(GroundTruth {
                target_id: traj.target_id,
                time: wp.time,
                position: wp.position,
                velocity: wp.velocity,
            });

            if let Some(m) = generate_radar(wp, radar_config, &mut rng) {
                all_meas.push(TimedMeasurement {
                    time: wp.time,
                    measurement: m,
                });
            }
        }
    }

    // Sort measurements by time
    all_meas.sort_by(|a, b| a.time.partial_cmp(&b.time).unwrap());
    all_gt.sort_by(|a, b| a.time.partial_cmp(&b.time).unwrap());

    (all_gt, all_meas)
}

/// Serialize a scenario to JSON.
pub fn serialize_scenario(scenario: &Scenario) -> String {
    serde_json::to_string_pretty(scenario).expect("Scenario serialization failed")
}

/// Deserialize a scenario from JSON.
pub fn deserialize_scenario(json: &str) -> Scenario {
    serde_json::from_str(json).expect("Scenario deserialization failed")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::trajectory::{Segment, SegmentType};

    fn make_scenario(n_targets: u32) -> Scenario {
        let trajectories = (0..n_targets)
            .map(|i| Trajectory {
                target_id: i,
                initial_position: [i as f64 * 10000.0, 0.0, 5000.0],
                initial_velocity: [250.0, 50.0 * i as f64, 0.0],
                segments: vec![Segment {
                    segment_type: SegmentType::Cv,
                    duration: 10.0,
                }],
                dt: 1.0,
            })
            .collect();

        Scenario {
            name: "test".into(),
            trajectories,
        }
    }

    #[test]
    fn multi_target_scenario() {
        let scenario = make_scenario(50);
        let radar = RadarConfig {
            p_detection: 0.9,
            ..Default::default()
        };
        let (gt, meas) = run_scenario(&scenario, &radar);

        // 50 targets * ~11 waypoints each
        assert!(gt.len() >= 500);
        // Should have many measurements
        assert!(meas.len() > 100);
        // Measurements are time-sorted
        for i in 1..meas.len() {
            assert!(meas[i].time >= meas[i - 1].time);
        }
    }

    #[test]
    fn scenario_serialization_roundtrip() {
        let scenario = make_scenario(3);
        let json = serialize_scenario(&scenario);
        let restored = deserialize_scenario(&json);
        assert_eq!(restored.trajectories.len(), 3);
        assert_eq!(restored.name, "test");
    }
}
