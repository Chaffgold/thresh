//! Track management: lifecycle state machine, M-of-N confirmation, class-specific heads.

pub mod cost_matrix;
pub mod ecef_tracker;
pub mod great_circle_tracker;
pub mod heads;
pub mod lifecycle;
pub mod othr_integration;
pub mod recentered_enu_tracker;
pub mod stereographic_tracker;
pub mod track;
pub mod tracker;
pub mod tracker_variant;

#[cfg(feature = "streaming")]
pub mod streaming;
