//! Dataset trait and metadata types.

use serde::{Deserialize, Serialize};

use crate::frame::Frame;

/// Coordinate reference frame for a dataset.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CoordinateFrame {
    /// East-North-Up local tangent plane.
    Enu,
    /// Earth-Centered Earth-Fixed.
    Ecef,
    /// Earth-Centered Inertial.
    Eci,
    /// WGS-84 geodetic (latitude, longitude, altitude).
    Wgs84,
}

/// Metadata describing a dataset.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetMetadata {
    /// Human-readable name.
    pub name: String,
    /// Data source identifier (e.g. "opensky", "nuscenes").
    pub source: String,
    /// Number of distinct targets, if known.
    pub target_count: Option<usize>,
    /// Time span as (start, end) in seconds, if known.
    pub time_span: Option<(f64, f64)>,
    /// Coordinate reference frame.
    pub coordinate_frame: CoordinateFrame,
}

/// A dataset that can be iterated frame-by-frame.
pub trait Dataset {
    /// Return metadata about this dataset.
    fn metadata(&self) -> DatasetMetadata;
    /// Iterate over measurement frames.
    fn frames(&self) -> Box<dyn Iterator<Item = Frame> + '_>;
    /// Iterate over ground-truth frames, if available.
    fn ground_truth(&self) -> Option<Box<dyn Iterator<Item = Frame> + '_>>;
}
