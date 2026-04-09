//! Timestamp type for tracking.

use serde::{Deserialize, Serialize};

/// Timestamp in seconds (f64 for sub-second precision).
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Serialize, Deserialize)]
pub struct Timestamp(pub f64);

impl Timestamp {
    /// Time difference in seconds.
    pub fn dt(&self, other: &Timestamp) -> f64 {
        self.0 - other.0
    }
}
