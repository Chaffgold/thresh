//! Frame and ground-truth types for dataset iteration.

use serde::{Deserialize, Serialize};
use thresh_core::measurement::Measurement;
use thresh_core::track::TargetClass;

/// A single frame of sensor data with optional ground truth.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Frame {
    /// Timestamp in seconds.
    pub timestamp: f64,
    /// Measurements captured in this frame.
    pub measurements: Vec<Measurement>,
    /// Ground-truth entries, if available.
    pub ground_truth: Option<Vec<GroundTruthEntry>>,
    /// Metadata about the sensor that produced this frame.
    pub sensor_metadata: Option<SensorInfo>,
}

/// A single ground-truth entry for one target.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GroundTruthEntry {
    /// Unique identifier for this target.
    pub target_id: u64,
    /// Position as [x, y, z].
    pub position: [f64; 3],
    /// Velocity as [vx, vy, vz], if known.
    pub velocity: Option<[f64; 3]>,
    /// Target classification, if known.
    pub class: Option<TargetClass>,
}

/// Metadata about the sensor that produced a frame.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SensorInfo {
    /// Sensor identifier.
    pub sensor_id: u32,
    /// Human-readable sensor type (e.g. "radar", "eo/ir").
    pub sensor_type: String,
}
