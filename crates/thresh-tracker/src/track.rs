//! Track struct and identity management.

use nalgebra::{DMatrix, DVector};
use thresh_core::track::{TargetClass, TrackId, TrackState};

/// A single tracked object.
#[derive(Debug, Clone)]
pub struct Track {
    /// Globally unique track ID.
    pub id: TrackId,
    /// Current state estimate.
    pub state: DVector<f64>,
    /// Current covariance estimate.
    pub covariance: DMatrix<f64>,
    /// Lifecycle state.
    pub lifecycle: TrackState,
    /// Target classification.
    pub class: TargetClass,
    /// Number of consecutive hits (measurements associated).
    pub hit_streak: usize,
    /// Total number of hits since creation.
    pub total_hits: usize,
    /// Number of consecutive frames without measurement.
    pub coast_count: usize,
    /// Total age in frames.
    pub age: usize,
    /// History of state estimates (last N).
    pub history: Vec<DVector<f64>>,
    /// Maximum history length.
    pub max_history: usize,
    /// Dominant IMM mode index (populated only in IMM mode).
    pub dominant_mode: Option<usize>,
    /// IMM mode probabilities (populated only in IMM mode).
    pub mode_probabilities: Option<DVector<f64>>,
    /// Key into the tracker's `imm_filters` map (populated only in IMM mode).
    pub(crate) imm_key: Option<usize>,
}

impl Track {
    /// Create a new tentative track from an initial detection.
    pub fn new(state: DVector<f64>, covariance: DMatrix<f64>, class: TargetClass) -> Self {
        Self {
            id: TrackId::new(),
            state,
            covariance,
            lifecycle: TrackState::Tentative,
            class,
            hit_streak: 1,
            total_hits: 1,
            coast_count: 0,
            age: 1,
            history: Vec::new(),
            max_history: 50,
            dominant_mode: None,
            mode_probabilities: None,
            imm_key: None,
        }
    }

    /// Record a measurement association (hit).
    pub fn record_hit(&mut self) {
        self.hit_streak += 1;
        self.total_hits += 1;
        self.coast_count = 0;
    }

    /// Record a missed detection (coast).
    pub fn record_miss(&mut self) {
        self.hit_streak = 0;
        self.coast_count += 1;
    }

    /// Advance age and save current state to history.
    pub fn advance(&mut self) {
        self.age += 1;
        if self.history.len() >= self.max_history {
            self.history.remove(0);
        }
        self.history.push(self.state.clone());
    }

    /// Check if this track is alive (not deleted).
    pub fn is_alive(&self) -> bool {
        self.lifecycle != TrackState::Deleted
    }
}

impl crate::cost_matrix::LinearTrack for Track {
    fn is_alive(&self) -> bool {
        self.is_alive()
    }
    fn state(&self) -> &DVector<f64> {
        &self.state
    }
    fn state_mut(&mut self) -> &mut DVector<f64> {
        &mut self.state
    }
    fn covariance(&self) -> &DMatrix<f64> {
        &self.covariance
    }
    fn covariance_mut(&mut self) -> &mut DMatrix<f64> {
        &mut self.covariance
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_track_is_tentative() {
        let t = Track::new(
            DVector::zeros(6),
            DMatrix::identity(6, 6),
            TargetClass::Aircraft,
        );
        assert_eq!(t.lifecycle, TrackState::Tentative);
        assert_eq!(t.hit_streak, 1);
        assert_eq!(t.coast_count, 0);
    }

    #[test]
    fn hit_miss_tracking() {
        let mut t = Track::new(
            DVector::zeros(6),
            DMatrix::identity(6, 6),
            TargetClass::Unknown,
        );
        t.record_hit();
        assert_eq!(t.hit_streak, 2);
        assert_eq!(t.total_hits, 2);
        t.record_miss();
        assert_eq!(t.hit_streak, 0);
        assert_eq!(t.coast_count, 1);
        t.record_hit();
        assert_eq!(t.hit_streak, 1);
        assert_eq!(t.coast_count, 0);
    }
}
