//! Track identity and lifecycle types.

use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicU64, Ordering};

/// Global monotonic counter for track IDs.
static NEXT_TRACK_ID: AtomicU64 = AtomicU64::new(1);

/// Globally unique track identifier (never reused).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TrackId(pub u64);

impl TrackId {
    /// Allocate a new globally unique track ID.
    pub fn new() -> Self {
        Self(NEXT_TRACK_ID.fetch_add(1, Ordering::Relaxed))
    }
}

impl Default for TrackId {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Display for TrackId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "T{}", self.0)
    }
}

/// Track lifecycle state.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TrackState {
    /// Track has been created but not yet confirmed.
    Tentative,
    /// Track has been confirmed by M-of-N hits.
    Confirmed,
    /// Track has not received updates for some frames.
    Coasting,
    /// Track is scheduled for deletion.
    Deleted,
}

/// Target classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TargetClass {
    /// Fixed-wing aircraft (subsonic–supersonic).
    Aircraft,
    /// Ballistic missile (boost, midcourse, terminal phases).
    Ballistic,
    /// Rotary-wing or multi-rotor unmanned aerial vehicle.
    Uav,
    /// Orbital object or satellite.
    Orbital,
    /// Unknown or unclassified target.
    Unknown,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn track_ids_are_unique() {
        let id1 = TrackId::new();
        let id2 = TrackId::new();
        let id3 = TrackId::new();
        assert_ne!(id1, id2);
        assert_ne!(id2, id3);
        assert!(id2.0 > id1.0);
    }

    #[test]
    fn track_state_default_lifecycle() {
        let state = TrackState::Tentative;
        assert_eq!(state, TrackState::Tentative);
    }

    #[test]
    fn track_id_display() {
        let id = TrackId(42);
        assert_eq!(format!("{id}"), "T42");
    }
}
