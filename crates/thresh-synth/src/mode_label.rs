//! Analytic motion-mode labelling from trajectory kinematics (Track B,
//! task 5.4).
//!
//! Labels are derived **only** from the ground-truth trajectory — never from
//! filter output — so they form an unbiased target for the IMM mode
//! classifier. Each label is computed from the velocity sequence of a target's
//! [`Waypoint`]s via finite differences.
//!
//! # Classification rule and precedence
//!
//! The design lists four conditions (accel > 1 m/s² → CA; turn rate > 3°/s →
//! CTRV; sustained turn > 3°/s with bank > 15° → coord_turn; else CV). These
//! overlap: a turn carries centripetal acceleration `a = ω·v`, which can
//! exceed the CA threshold even though the motion is a turn, not a straight-line
//! acceleration. We therefore evaluate **turning first**:
//!
//! 1. If `|turn_rate| > 3°/s`:
//!    - if the turn is *sustained* (the whole local window exceeds the
//!      threshold) **and** the implied coordinated-turn bank exceeds 15° →
//!      [`MotionModeLabel::CoordTurn`];
//!    - otherwise → [`MotionModeLabel::Ctrv`].
//! 2. Else if `|acceleration| > 1 m/s²` (straight-line) →
//!    [`MotionModeLabel::Ca`].
//! 3. Else → [`MotionModeLabel::Cv`].
//!
//! Bank is not stored in the trajectory, so it is derived from the
//! coordinated-turn relation for a balanced turn:
//! `bank = atan2(|ω|·speed, g)`.

use crate::trajectory::Waypoint;
use thresh_core::motion_mode::MotionModeLabel;

/// Acceleration magnitude (m/s²) above which straight-line motion is `CA`.
pub const ACCEL_THRESHOLD_MPS2: f64 = 1.0;
/// Turn rate (deg/s) above which motion is considered turning.
pub const TURN_RATE_THRESHOLD_DEG_S: f64 = 3.0;
/// Implied bank angle (deg) above which a sustained turn is `coord_turn`.
pub const BANK_THRESHOLD_DEG: f64 = 15.0;
/// Gravitational acceleration used in the coordinated-turn bank relation.
pub const GRAVITY_MPS2: f64 = 9.81;
/// Half-width (in waypoints) of the window used for the "sustained" test.
pub const SUSTAINED_HALF_WINDOW: usize = 2;

const TURN_RATE_THRESHOLD_RAD_S: f64 = TURN_RATE_THRESHOLD_DEG_S * std::f64::consts::PI / 180.0;
const BANK_THRESHOLD_RAD: f64 = BANK_THRESHOLD_DEG * std::f64::consts::PI / 180.0;

/// Horizontal speed (m/s) from a velocity triple `[vx, vy, vz]`.
fn horizontal_speed(velocity: &[f64; 3]) -> f64 {
    (velocity[0] * velocity[0] + velocity[1] * velocity[1]).sqrt()
}

/// Heading (rad) of the horizontal velocity, measured from +x toward +y.
fn heading(velocity: &[f64; 3]) -> f64 {
    velocity[1].atan2(velocity[0])
}

/// Wrap an angle difference into `(-π, π]`.
///
/// Branch-free: a single modulo brings the value into `(-2π, 2π)` and one
/// conditional adjustment folds it into `(-π, π]`. (Heading differences are
/// already within `(-2π, 2π)`, so one adjustment always suffices.)
fn wrap_angle(a: f64) -> f64 {
    use std::f64::consts::{PI, TAU};
    let r = a % TAU;
    if r > PI {
        r - TAU
    } else if r <= -PI {
        r + TAU
    } else {
        r
    }
}

/// Indices `(lo, hi)` bracketing waypoint `i` for a central difference, falling
/// back to a one-sided pair at the ends. Returns `None` if fewer than two
/// waypoints exist.
fn diff_bracket(len: usize, i: usize) -> Option<(usize, usize)> {
    if len < 2 {
        return None;
    }
    let lo = if i == 0 { 0 } else { i - 1 };
    let hi = if i + 1 >= len { len - 1 } else { i + 1 };
    if lo == hi { None } else { Some((lo, hi)) }
}

/// Instantaneous acceleration magnitude (m/s²) at waypoint `i`, from a finite
/// difference of velocity. Returns `0.0` if it cannot be computed.
pub fn accel_magnitude_mps2(waypoints: &[Waypoint], i: usize) -> f64 {
    let Some((lo, hi)) = diff_bracket(waypoints.len(), i) else {
        return 0.0;
    };
    let dt = waypoints[hi].time - waypoints[lo].time;
    if dt.abs() < 1e-9 {
        return 0.0;
    }
    let (a, b) = (&waypoints[lo].velocity, &waypoints[hi].velocity);
    let dvx = b[0] - a[0];
    let dvy = b[1] - a[1];
    let dvz = b[2] - a[2];
    (dvx * dvx + dvy * dvy + dvz * dvz).sqrt() / dt
}

/// Instantaneous turn rate (rad/s) at waypoint `i`, from a finite difference of
/// horizontal heading (angle-wrapped). Returns `0.0` if it cannot be computed.
pub fn turn_rate_rad_s(waypoints: &[Waypoint], i: usize) -> f64 {
    let Some((lo, hi)) = diff_bracket(waypoints.len(), i) else {
        return 0.0;
    };
    let dt = waypoints[hi].time - waypoints[lo].time;
    if dt.abs() < 1e-9 {
        return 0.0;
    }
    let dheading = wrap_angle(heading(&waypoints[hi].velocity) - heading(&waypoints[lo].velocity));
    dheading / dt
}

/// Implied bank angle (rad) of a balanced coordinated turn at the given turn
/// rate and horizontal speed: `atan2(|ω|·speed, g)`.
pub fn bank_angle_rad(turn_rate_rad_s: f64, horizontal_speed_mps: f64) -> f64 {
    (turn_rate_rad_s.abs() * horizontal_speed_mps).atan2(GRAVITY_MPS2)
}

/// Whether the turn at waypoint `i` is *sustained*: every waypoint in the local
/// window `[i ± SUSTAINED_HALF_WINDOW]` (clamped to the slice) exceeds the turn
/// rate threshold.
fn is_sustained_turn(waypoints: &[Waypoint], i: usize) -> bool {
    let lo = i.saturating_sub(SUSTAINED_HALF_WINDOW);
    let hi = (i + SUSTAINED_HALF_WINDOW).min(waypoints.len().saturating_sub(1));
    (lo..=hi).all(|j| turn_rate_rad_s(waypoints, j).abs() > TURN_RATE_THRESHOLD_RAD_S)
}

/// Derive the analytic motion-mode label at waypoint `i` from trajectory
/// kinematics. See the module docs for the rule and its precedence.
pub fn analytic_mode_label(waypoints: &[Waypoint], i: usize) -> MotionModeLabel {
    let turn_rate = turn_rate_rad_s(waypoints, i);
    if turn_rate.abs() > TURN_RATE_THRESHOLD_RAD_S {
        let speed = horizontal_speed(&waypoints[i].velocity);
        let bank = bank_angle_rad(turn_rate, speed);
        if bank > BANK_THRESHOLD_RAD && is_sustained_turn(waypoints, i) {
            MotionModeLabel::CoordTurn
        } else {
            MotionModeLabel::Ctrv
        }
    } else if accel_magnitude_mps2(waypoints, i) > ACCEL_THRESHOLD_MPS2 {
        MotionModeLabel::Ca
    } else {
        MotionModeLabel::Cv
    }
}

/// Label every waypoint in a trajectory. Convenience wrapper over
/// [`analytic_mode_label`].
pub fn label_trajectory(waypoints: &[Waypoint]) -> Vec<MotionModeLabel> {
    (0..waypoints.len())
        .map(|i| analytic_mode_label(waypoints, i))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::trajectory::{Segment, SegmentType, Trajectory};

    fn fraction(labels: &[MotionModeLabel], target: MotionModeLabel) -> f64 {
        let n = labels.iter().filter(|&&l| l == target).count();
        n as f64 / labels.len() as f64
    }

    /// Task 5.6: a constant-velocity trajectory yields 100% `CV` labels.
    #[test]
    fn cv_trajectory_yields_all_cv_labels() {
        let traj = Trajectory {
            target_id: 0,
            initial_position: [0.0, 0.0, 1000.0],
            initial_velocity: [120.0, 0.0, 0.0],
            segments: vec![Segment {
                segment_type: SegmentType::Cv,
                duration: 60.0,
            }],
            dt: 1.0,
        };
        let wps = traj.generate();
        let labels = label_trajectory(&wps);
        assert_eq!(
            fraction(&labels, MotionModeLabel::Cv),
            1.0,
            "constant velocity must be 100% CV, got {labels:?}"
        );
    }

    /// Task 5.6: a coordinated-turn trajectory yields ≥80% `coord_turn` during
    /// the turn segment. A 0.1 rad/s (≈5.7°/s) turn at 100 m/s implies a bank
    /// of atan2(10, 9.81) ≈ 45.6° > 15°, so it should classify as a banked
    /// coordinated turn.
    #[test]
    fn coordinated_turn_yields_mostly_coord_turn() {
        let traj = Trajectory {
            target_id: 0,
            initial_position: [0.0, 0.0, 5000.0],
            initial_velocity: [100.0, 0.0, 0.0],
            segments: vec![Segment {
                segment_type: SegmentType::Ctrv { turn_rate: 0.1 },
                duration: 120.0,
            }],
            dt: 1.0,
        };
        let wps = traj.generate();
        // Skip the first/last few steps (transient one-sided differences).
        let interior: Vec<_> = (5..wps.len() - 5)
            .map(|i| analytic_mode_label(&wps, i))
            .collect();
        let coord_frac = fraction(&interior, MotionModeLabel::CoordTurn);
        assert!(
            coord_frac >= 0.80,
            "sustained banked turn should be ≥80% coord_turn, got {:.2} ({interior:?})",
            coord_frac
        );
    }

    /// A straight-line constant-acceleration segment classifies as `CA`.
    #[test]
    fn straight_acceleration_yields_ca() {
        let traj = Trajectory {
            target_id: 0,
            initial_position: [0.0, 0.0, 1000.0],
            initial_velocity: [50.0, 0.0, 0.0],
            segments: vec![Segment {
                segment_type: SegmentType::Ca {
                    acceleration: [3.0, 0.0, 0.0],
                },
                duration: 30.0,
            }],
            dt: 1.0,
        };
        let wps = traj.generate();
        let interior: Vec<_> = (2..wps.len() - 2)
            .map(|i| analytic_mode_label(&wps, i))
            .collect();
        assert!(
            fraction(&interior, MotionModeLabel::Ca) >= 0.90,
            "straight acceleration should be ≥90% CA, got {interior:?}"
        );
    }

    /// A slow, low-speed turn banks below 15° and classifies as `CTRV`, not
    /// `coord_turn` — the bank threshold is what separates the two.
    #[test]
    fn slow_shallow_turn_yields_ctrv_not_coord_turn() {
        // ω = 0.06 rad/s (≈3.4°/s, just over threshold), speed 30 m/s →
        // bank = atan2(1.8, 9.81) ≈ 10.4° < 15°.
        let traj = Trajectory {
            target_id: 0,
            initial_position: [0.0, 0.0, 500.0],
            initial_velocity: [30.0, 0.0, 0.0],
            segments: vec![Segment {
                segment_type: SegmentType::Ctrv { turn_rate: 0.06 },
                duration: 60.0,
            }],
            dt: 1.0,
        };
        let wps = traj.generate();
        let interior: Vec<_> = (5..wps.len() - 5)
            .map(|i| analytic_mode_label(&wps, i))
            .collect();
        assert!(
            fraction(&interior, MotionModeLabel::Ctrv) >= 0.80,
            "shallow turn should be ≥80% CTRV (bank < 15°), got {interior:?}"
        );
        assert_eq!(
            fraction(&interior, MotionModeLabel::CoordTurn),
            0.0,
            "shallow turn must never be coord_turn"
        );
    }

    #[test]
    fn bank_relation_matches_hand_calc() {
        // 0.1 rad/s at 100 m/s → atan2(10, 9.81) ≈ 0.7954 rad ≈ 45.6°.
        let bank = bank_angle_rad(0.1, 100.0);
        assert!((bank - 0.7954).abs() < 1e-3, "bank = {bank}");
    }

    #[test]
    fn wrap_angle_covers_all_three_branches() {
        use std::f64::consts::PI;
        // In-range value passes through unchanged.
        assert!((wrap_angle(0.5) - 0.5).abs() < 1e-12);
        // > π folds down by 2π.
        assert!((wrap_angle(1.5 * PI) - (-0.5 * PI)).abs() < 1e-9);
        // ≤ -π folds up by 2π.
        assert!((wrap_angle(-1.5 * PI) - (0.5 * PI)).abs() < 1e-9);
    }

    #[test]
    fn derivatives_handle_degenerate_inputs() {
        let wp = |t: f64, vx: f64| Waypoint {
            time: t,
            position: [0.0, 0.0, 0.0],
            velocity: [vx, 0.0, 0.0],
        };
        // Single waypoint: no difference bracket → zero.
        let one = vec![wp(0.0, 10.0)];
        assert!(accel_magnitude_mps2(&one, 0).abs() < 1e-12);
        assert!(turn_rate_rad_s(&one, 0).abs() < 1e-12);
        // Two waypoints sharing a timestamp: divide-by-zero guard → zero.
        let same_time = vec![wp(1.0, 10.0), wp(1.0, 20.0)];
        assert!(accel_magnitude_mps2(&same_time, 0).abs() < 1e-12);
        assert!(turn_rate_rad_s(&same_time, 0).abs() < 1e-12);
    }
}
