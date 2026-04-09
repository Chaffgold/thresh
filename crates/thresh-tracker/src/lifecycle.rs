//! Track lifecycle policies: M-of-N confirmation, max-coast deletion.

use thresh_core::track::TrackState;

use crate::track::Track;

/// M-of-N confirmation policy.
#[derive(Debug, Clone, Copy)]
pub struct ConfirmationPolicy {
    /// Number of hits required.
    pub m: usize,
    /// Window of frames to consider.
    pub n: usize,
}

impl Default for ConfirmationPolicy {
    fn default() -> Self {
        Self { m: 3, n: 5 }
    }
}

impl ConfirmationPolicy {
    pub fn new(m: usize, n: usize) -> Self {
        assert!(m <= n, "M must be <= N");
        Self { m, n }
    }

    /// Check if a track should be confirmed.
    pub fn should_confirm(&self, track: &Track) -> bool {
        track.lifecycle == TrackState::Tentative && track.total_hits >= self.m
    }

    /// Check if a tentative track should be deleted (failed to confirm in N frames).
    pub fn should_delete_tentative(&self, track: &Track) -> bool {
        track.lifecycle == TrackState::Tentative && track.age > self.n && track.total_hits < self.m
    }
}

/// Max-coast-age deletion policy.
#[derive(Debug, Clone, Copy)]
pub struct DeletionPolicy {
    /// Maximum frames without measurement before deletion.
    pub max_coast_age: usize,
}

impl Default for DeletionPolicy {
    fn default() -> Self {
        Self { max_coast_age: 5 }
    }
}

impl DeletionPolicy {
    pub fn new(max_coast_age: usize) -> Self {
        Self { max_coast_age }
    }

    /// Check if a confirmed/coasting track should be deleted.
    pub fn should_delete(&self, track: &Track) -> bool {
        track.coast_count >= self.max_coast_age
    }
}

/// Apply lifecycle transitions to a track.
pub fn update_lifecycle(
    track: &mut Track,
    was_associated: bool,
    confirm: &ConfirmationPolicy,
    delete: &DeletionPolicy,
) {
    if was_associated {
        track.record_hit();
    } else {
        track.record_miss();
    }
    track.advance();

    match track.lifecycle {
        TrackState::Tentative => {
            if confirm.should_confirm(track) {
                track.lifecycle = TrackState::Confirmed;
            } else if confirm.should_delete_tentative(track) {
                track.lifecycle = TrackState::Deleted;
            }
        }
        TrackState::Confirmed => {
            if !was_associated {
                track.lifecycle = TrackState::Coasting;
            }
        }
        TrackState::Coasting => {
            if was_associated {
                track.lifecycle = TrackState::Confirmed;
            } else if delete.should_delete(track) {
                track.lifecycle = TrackState::Deleted;
            }
        }
        TrackState::Deleted => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{DMatrix, DVector};
    use thresh_core::track::TargetClass;

    #[test]
    fn m_of_n_3_of_5() {
        let confirm = ConfirmationPolicy::new(3, 5);
        let delete = DeletionPolicy::default();
        let mut track = Track::new(
            DVector::zeros(6),
            DMatrix::identity(6, 6),
            TargetClass::Aircraft,
        );

        // Frame 2: hit
        update_lifecycle(&mut track, true, &confirm, &delete);
        assert_eq!(track.lifecycle, TrackState::Tentative);

        // Frame 3: hit -> 3 hits in 3 frames, should confirm
        update_lifecycle(&mut track, true, &confirm, &delete);
        assert_eq!(track.lifecycle, TrackState::Confirmed);
    }

    #[test]
    fn tentative_deletion() {
        let confirm = ConfirmationPolicy::new(3, 5);
        let delete = DeletionPolicy::default();
        let mut track = Track::new(
            DVector::zeros(6),
            DMatrix::identity(6, 6),
            TargetClass::Unknown,
        );

        // Miss for 6 frames (age > n=5, hits=1 < m=3)
        for _ in 0..6 {
            update_lifecycle(&mut track, false, &confirm, &delete);
        }
        assert_eq!(track.lifecycle, TrackState::Deleted);
    }

    #[test]
    fn coast_then_delete() {
        let confirm = ConfirmationPolicy::new(1, 1);
        let delete = DeletionPolicy::new(3);
        let mut track = Track::new(
            DVector::zeros(6),
            DMatrix::identity(6, 6),
            TargetClass::Ballistic,
        );

        // Confirm immediately (m=1, n=1)
        update_lifecycle(&mut track, true, &confirm, &delete);
        assert_eq!(track.lifecycle, TrackState::Confirmed);

        // Miss -> coasting
        update_lifecycle(&mut track, false, &confirm, &delete);
        assert_eq!(track.lifecycle, TrackState::Coasting);

        // Miss again
        update_lifecycle(&mut track, false, &confirm, &delete);
        assert_eq!(track.lifecycle, TrackState::Coasting);

        // 3rd miss -> deleted
        update_lifecycle(&mut track, false, &confirm, &delete);
        assert_eq!(track.lifecycle, TrackState::Deleted);
    }

    #[test]
    fn coast_recovery() {
        let confirm = ConfirmationPolicy::new(1, 1);
        let delete = DeletionPolicy::new(5);
        let mut track = Track::new(DVector::zeros(6), DMatrix::identity(6, 6), TargetClass::Uav);

        // Confirm
        update_lifecycle(&mut track, true, &confirm, &delete);
        assert_eq!(track.lifecycle, TrackState::Confirmed);

        // Coast
        update_lifecycle(&mut track, false, &confirm, &delete);
        assert_eq!(track.lifecycle, TrackState::Coasting);

        // Re-associate -> back to confirmed
        update_lifecycle(&mut track, true, &confirm, &delete);
        assert_eq!(track.lifecycle, TrackState::Confirmed);
    }
}
