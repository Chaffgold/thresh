//! Motion-mode classification labels shared across the tracking and
//! training-data pipelines.
//!
//! These labels are the ground-truth target for the Track B IMM mode
//! classifier (see the `flight-data-training-pipeline` change). They are
//! derived analytically from trajectory kinematics, never from filter output.

use serde::{Deserialize, Serialize};

/// Analytic motion-mode label for a trajectory timestep.
///
/// The variants are index-aligned with the 4-model IMM bank produced by
/// `thresh_filter::imm::ImmConfig::cv_ca_ctrv_ct` (`0 = CV`, `1 = CA`,
/// `2 = CTRV`, `3 = CoordTurn`), so a label can be compared directly against
/// the IMM's `dominant_mode` index.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MotionModeLabel {
    /// Constant velocity — no significant acceleration or turn.
    Cv,
    /// Constant acceleration along a (roughly) straight line.
    Ca,
    /// Constant turn rate, near-level — turning but not banked past the
    /// coordinated-turn threshold.
    Ctrv,
    /// Coordinated (banked) turn — a sustained turn whose implied bank angle
    /// exceeds the coordinated-turn threshold.
    CoordTurn,
}

impl MotionModeLabel {
    /// IMM mode index this label maps to in the canonical `cv_ca_ctrv_ct` bank.
    pub fn as_index(self) -> usize {
        match self {
            Self::Cv => 0,
            Self::Ca => 1,
            Self::Ctrv => 2,
            Self::CoordTurn => 3,
        }
    }

    /// Construct from a `cv_ca_ctrv_ct` IMM mode index, if in range `[0, 4)`.
    pub fn from_index(index: usize) -> Option<Self> {
        match index {
            0 => Some(Self::Cv),
            1 => Some(Self::Ca),
            2 => Some(Self::Ctrv),
            3 => Some(Self::CoordTurn),
            _ => None,
        }
    }

    /// Stable lowercase string label (used in dataset columns and reports).
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Cv => "cv",
            Self::Ca => "ca",
            Self::Ctrv => "ctrv",
            Self::CoordTurn => "coord_turn",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn index_roundtrip_is_stable() {
        for idx in 0..4 {
            let label = MotionModeLabel::from_index(idx).unwrap();
            assert_eq!(label.as_index(), idx);
        }
        assert_eq!(MotionModeLabel::from_index(4), None);
    }

    #[test]
    fn labels_align_with_cv_ca_ctrv_ct_bank() {
        // Mode ordering MUST match ImmConfig::cv_ca_ctrv_ct.
        assert_eq!(MotionModeLabel::Cv.as_index(), 0);
        assert_eq!(MotionModeLabel::Ca.as_index(), 1);
        assert_eq!(MotionModeLabel::Ctrv.as_index(), 2);
        assert_eq!(MotionModeLabel::CoordTurn.as_index(), 3);
    }

    #[test]
    fn string_labels_are_snake_case() {
        assert_eq!(MotionModeLabel::Cv.as_str(), "cv");
        assert_eq!(MotionModeLabel::CoordTurn.as_str(), "coord_turn");
    }
}
