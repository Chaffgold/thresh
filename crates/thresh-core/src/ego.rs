//! Ego-platform pose and motion types (automotive-tracking-pipeline, Phase 2).
//!
//! Automotive sensors ride on a moving platform, so measurements arrive in the
//! ego/body frame while tracking happens in a fixed world ENU frame. These
//! types carry the platform state needed to lift measurements into the world
//! frame (and, later, to feed platform motion into prediction):
//!
//! - [`EgoPose`] — the ego→world transform at an instant (translation in
//!   metres + orientation quaternion).
//! - [`EgoMotion`] — a pose plus linear (m/s) and angular (rad/s) velocity.
//!
//! Conventions: the pose maps **ego/body coordinates to world ENU** —
//! `p_world = R(q) * p_ego + t`. Quaternions are `[w, x, y, z]` (scalar
//! first), matching the nuScenes `ego_pose` record. Velocities are expressed
//! in the **world** frame. All of this is pure math with no I/O, so dataset
//! bridges (e.g. the PyO3-gated nuScenes module) stay thin and the testable
//! logic lives here.

use serde::{Deserialize, Serialize};

/// Ego→world pose at an instant: `p_world = R(rotation) * p_ego + translation`.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct EgoPose {
    /// Ego origin in world ENU coordinates, metres.
    pub translation_m: [f64; 3],
    /// Ego→world rotation as a unit quaternion `[w, x, y, z]` (scalar first).
    pub rotation_wxyz: [f64; 4],
    /// Timestamp in seconds.
    pub time_s: f64,
}

impl EgoPose {
    /// Identity pose at `time_s`: ego frame coincides with the world frame.
    pub fn identity(time_s: f64) -> Self {
        Self {
            translation_m: [0.0; 3],
            rotation_wxyz: [1.0, 0.0, 0.0, 0.0],
            time_s,
        }
    }

    /// Rotate an ego-frame vector into the world frame (no translation).
    pub fn rotate_to_world(&self, v_ego: [f64; 3]) -> [f64; 3] {
        rotate(normalize(self.rotation_wxyz), v_ego)
    }

    /// Transform an ego-frame point into the world frame (rotate + translate).
    pub fn transform_to_world(&self, p_ego: [f64; 3]) -> [f64; 3] {
        let r = self.rotate_to_world(p_ego);
        [
            r[0] + self.translation_m[0],
            r[1] + self.translation_m[1],
            r[2] + self.translation_m[2],
        ]
    }

    /// Heading (yaw) of the ego x-axis in the world frame, radians in
    /// `(-π, π]`. ENU convention: 0 = East, `π/2` = North.
    pub fn yaw_rad(&self) -> f64 {
        let x_world = self.rotate_to_world([1.0, 0.0, 0.0]);
        x_world[1].atan2(x_world[0])
    }
}

/// Ego pose plus world-frame linear and angular velocity.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct EgoMotion {
    /// The ego→world pose.
    pub pose: EgoPose,
    /// Linear velocity of the ego origin in the world frame, m/s.
    pub linear_velocity_mps: [f64; 3],
    /// Angular velocity in the world frame, rad/s (axis-angle rate).
    pub angular_velocity_radps: [f64; 3],
}

impl EgoMotion {
    /// A stationary platform at `pose`.
    pub fn stationary(pose: EgoPose) -> Self {
        Self {
            pose,
            linear_velocity_mps: [0.0; 3],
            angular_velocity_radps: [0.0; 3],
        }
    }

    /// Derive ego motion from two consecutive poses by finite differences.
    ///
    /// Linear velocity is the world-frame translation delta over `dt`; angular
    /// velocity is the axis-angle of the relative rotation (in the world
    /// frame) over `dt`. The returned motion carries `cur` as its pose.
    /// Returns `None` if the poses are not strictly ordered in time.
    pub fn from_pose_pair(prev: &EgoPose, cur: &EgoPose) -> Option<Self> {
        let dt = cur.time_s - prev.time_s;
        if dt <= 0.0 {
            return None;
        }
        let linear = [
            (cur.translation_m[0] - prev.translation_m[0]) / dt,
            (cur.translation_m[1] - prev.translation_m[1]) / dt,
            (cur.translation_m[2] - prev.translation_m[2]) / dt,
        ];
        // Relative rotation in the world frame: q_rel * q_prev = q_cur.
        let q_rel = multiply(
            normalize(cur.rotation_wxyz),
            conjugate(normalize(prev.rotation_wxyz)),
        );
        let (axis, angle) = axis_angle(q_rel);
        let rate = angle / dt;
        Some(Self {
            pose: *cur,
            linear_velocity_mps: linear,
            angular_velocity_radps: [axis[0] * rate, axis[1] * rate, axis[2] * rate],
        })
    }
}

// --- minimal quaternion helpers (scalar-first `[w, x, y, z]`) ---------------

fn normalize(q: [f64; 4]) -> [f64; 4] {
    let n = (q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]).sqrt();
    // Non-finite norms (NaN sails past `n < 1e-12`; ±inf would divide to an
    // all-zero "quaternion") degrade to identity instead of poisoning every
    // downstream transform.
    if !n.is_finite() || n < 1e-12 {
        return [1.0, 0.0, 0.0, 0.0];
    }
    [q[0] / n, q[1] / n, q[2] / n, q[3] / n]
}

fn conjugate(q: [f64; 4]) -> [f64; 4] {
    [q[0], -q[1], -q[2], -q[3]]
}

fn multiply(a: [f64; 4], b: [f64; 4]) -> [f64; 4] {
    [
        a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3],
        a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2],
        a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1],
        a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0],
    ]
}

/// Rotate vector `v` by unit quaternion `q`: `q * (0, v) * q⁻¹`.
fn rotate(q: [f64; 4], v: [f64; 3]) -> [f64; 3] {
    let p = [0.0, v[0], v[1], v[2]];
    let r = multiply(multiply(q, p), conjugate(q));
    [r[1], r[2], r[3]]
}

/// Decompose a unit quaternion into a rotation axis and angle in `[0, π]`.
/// The identity rotation returns axis `[0, 0, 1]` with angle 0.
fn axis_angle(q: [f64; 4]) -> ([f64; 3], f64) {
    let q = normalize(q);
    // Force the scalar part non-negative so the angle is the short way round.
    let q = if q[0] < 0.0 {
        [-q[0], -q[1], -q[2], -q[3]]
    } else {
        q
    };
    let sin_half = (q[1] * q[1] + q[2] * q[2] + q[3] * q[3]).sqrt();
    if sin_half < 1e-12 {
        return ([0.0, 0.0, 1.0], 0.0);
    }
    let angle = 2.0 * sin_half.atan2(q[0]);
    ([q[1] / sin_half, q[2] / sin_half, q[3] / sin_half], angle)
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f64 = 1e-10;

    fn yaw_quat(yaw: f64) -> [f64; 4] {
        [(yaw / 2.0).cos(), 0.0, 0.0, (yaw / 2.0).sin()]
    }

    #[test]
    fn identity_pose_is_a_no_op() {
        let pose = EgoPose::identity(0.0);
        let p = pose.transform_to_world([3.0, -2.0, 1.0]);
        assert!((p[0] - 3.0).abs() < EPS);
        assert!((p[1] + 2.0).abs() < EPS);
        assert!((p[2] - 1.0).abs() < EPS);
        assert!(pose.yaw_rad().abs() < EPS);
    }

    #[test]
    fn translation_only_shifts_points() {
        let pose = EgoPose {
            translation_m: [100.0, 50.0, 2.0],
            rotation_wxyz: [1.0, 0.0, 0.0, 0.0],
            time_s: 0.0,
        };
        let p = pose.transform_to_world([1.0, 0.0, 0.0]);
        assert!((p[0] - 101.0).abs() < EPS);
        assert!((p[1] - 50.0).abs() < EPS);
        assert!((p[2] - 2.0).abs() < EPS);
    }

    #[test]
    fn ninety_degree_yaw_rotates_x_to_y() {
        let pose = EgoPose {
            translation_m: [0.0; 3],
            rotation_wxyz: yaw_quat(std::f64::consts::FRAC_PI_2),
            time_s: 0.0,
        };
        // A point 10 m ahead of the ego (its +x) lands on world +y.
        let p = pose.transform_to_world([10.0, 0.0, 0.0]);
        assert!(p[0].abs() < 1e-9, "x should vanish, got {}", p[0]);
        assert!((p[1] - 10.0).abs() < 1e-9);
        assert!((pose.yaw_rad() - std::f64::consts::FRAC_PI_2).abs() < EPS);
    }

    #[test]
    fn stationary_object_is_frame_invariant_across_ego_motion() {
        // The spec scenario: a fixed world point seen from two different ego
        // poses must lift to the same world coordinates from both.
        let world_point = [50.0, 20.0, 0.0];
        let pose_a = EgoPose {
            translation_m: [0.0, 0.0, 0.0],
            rotation_wxyz: yaw_quat(0.0),
            time_s: 0.0,
        };
        let pose_b = EgoPose {
            translation_m: [10.0, 5.0, 0.0],
            rotation_wxyz: yaw_quat(0.3),
            time_s: 1.0,
        };
        for pose in [pose_a, pose_b] {
            // What the sensor would report: the world point in ego coords
            // (inverse transform), then lifted back to world.
            let q = normalize(pose.rotation_wxyz);
            let d = [
                world_point[0] - pose.translation_m[0],
                world_point[1] - pose.translation_m[1],
                world_point[2] - pose.translation_m[2],
            ];
            let p_ego = rotate(conjugate(q), d);
            let lifted = pose.transform_to_world(p_ego);
            for i in 0..3 {
                assert!(
                    (lifted[i] - world_point[i]).abs() < 1e-9,
                    "axis {i}: {} vs {}",
                    lifted[i],
                    world_point[i]
                );
            }
        }
    }

    #[test]
    fn from_pose_pair_recovers_linear_and_angular_velocity() {
        let prev = EgoPose {
            translation_m: [0.0, 0.0, 0.0],
            rotation_wxyz: yaw_quat(0.0),
            time_s: 10.0,
        };
        let cur = EgoPose {
            translation_m: [10.0, -4.0, 0.2],
            rotation_wxyz: yaw_quat(0.5),
            time_s: 12.0,
        };
        let m = EgoMotion::from_pose_pair(&prev, &cur).expect("dt > 0");
        assert!((m.linear_velocity_mps[0] - 5.0).abs() < EPS);
        assert!((m.linear_velocity_mps[1] + 2.0).abs() < EPS);
        assert!((m.linear_velocity_mps[2] - 0.1).abs() < EPS);
        // Yaw advanced 0.5 rad in 2 s about +z.
        assert!((m.angular_velocity_radps[2] - 0.25).abs() < 1e-9);
        assert!(m.angular_velocity_radps[0].abs() < 1e-9);
        assert!(m.pose == cur);
    }

    #[test]
    fn from_pose_pair_rejects_non_positive_dt() {
        let a = EgoPose::identity(5.0);
        let b = EgoPose::identity(5.0);
        assert!(EgoMotion::from_pose_pair(&a, &b).is_none());
        let c = EgoPose::identity(4.0);
        assert!(EgoMotion::from_pose_pair(&a, &c).is_none());
    }

    #[test]
    fn ego_pose_serde_roundtrip() {
        let pose = EgoPose {
            translation_m: [1.0, 2.0, 3.0],
            rotation_wxyz: yaw_quat(1.0),
            time_s: 42.0,
        };
        let m = EgoMotion {
            pose,
            linear_velocity_mps: [1.0, 0.0, 0.0],
            angular_velocity_radps: [0.0, 0.0, 0.1],
        };
        let json = serde_json::to_string(&m).expect("serialize");
        let m2: EgoMotion = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(m, m2);
    }

    #[test]
    fn non_unit_quaternion_is_normalized() {
        let pose = EgoPose {
            translation_m: [0.0; 3],
            // 2x the identity quaternion — must behave as identity.
            rotation_wxyz: [2.0, 0.0, 0.0, 0.0],
            time_s: 0.0,
        };
        let p = pose.transform_to_world([1.0, 1.0, 1.0]);
        for (i, want) in [1.0, 1.0, 1.0].iter().enumerate() {
            assert!((p[i] - want).abs() < EPS);
        }
    }

    #[test]
    fn non_finite_quaternion_degrades_to_identity_not_nan() {
        for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let pose = EgoPose {
                translation_m: [5.0, 0.0, 0.0],
                rotation_wxyz: [bad, 0.0, 0.0, 0.0],
                time_s: 0.0,
            };
            let p = pose.transform_to_world([1.0, 2.0, 3.0]);
            assert!(
                p.iter().all(|v| v.is_finite()),
                "{bad} must not propagate: {p:?}"
            );
            // Degrades to the identity rotation: translate only.
            assert!((p[0] - 6.0).abs() < EPS);
            assert!((p[1] - 2.0).abs() < EPS);
        }
    }
}
