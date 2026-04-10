//! OTHR (Over-The-Horizon Radar) tracker integration helpers.
//!
//! OTHR measurements are nonlinear in range/azimuth and lack a direct altitude
//! observable, which makes them a poor fit for the linear Cartesian tracker in
//! this crate. This module bridges the gap by converting OTHR observations to
//! Cartesian ENU detections (via Vincenty on the ellipsoid), which can then be
//! fed directly into [`crate::tracker::MultiObjectTracker`].
//!
//! A nonlinear observation Jacobian is also provided for future EKF use.

use nalgebra::{DMatrix, DVector};
use thresh_core::measurement::Measurement;
use thresh_core::othr::{OthrSensorRegistration, othr_to_enu};

/// Convert an OTHR measurement to a Cartesian ENU detection vector `[x, y, z]`.
///
/// Returns `None` if the measurement is not an [`Measurement::Othr`] variant.
///
/// The `estimated_alt_m` argument is the assumed target altitude (since OTHR
/// has no elevation observable). A typical value is `10_000.0` m for
/// aircraft-class targets.
pub fn othr_to_cartesian(
    measurement: &Measurement,
    registration: &OthrSensorRegistration,
    estimated_alt_m: f64,
    ref_lat_rad: f64,
    ref_lon_rad: f64,
    ref_alt_m: f64,
) -> Option<DVector<f64>> {
    match measurement {
        Measurement::Othr {
            ground_range_m,
            azimuth_rad,
            ..
        } => {
            let enu = othr_to_enu(
                registration,
                *ground_range_m,
                *azimuth_rad,
                estimated_alt_m,
                ref_lat_rad,
                ref_lon_rad,
                ref_alt_m,
            );
            Some(DVector::from_column_slice(&[enu.x, enu.y, enu.z]))
        }
        _ => None,
    }
}

/// Compute the observation Jacobian for an OTHR measurement at a given state.
///
/// Maps state `[x, vx, y, vy, z, vz]` (length 6) to observation
/// `[ground_range, azimuth, doppler]` (length 3), where the transmitter is at
/// `transmitter_enu = [tx, ty, tz]` in the same local ENU frame as the state.
///
/// This is a flat-Earth approximation suitable for linearization around a
/// state estimate. It is provided as a helper for a future nonlinear (EKF)
/// OTHR filter; the integration test in this crate uses the Cartesian tracker
/// with [`othr_to_cartesian`] directly.
///
/// Returns a 3x6 Jacobian matrix.
pub fn othr_observation_jacobian(state: &DVector<f64>, transmitter_enu: &[f64; 3]) -> DMatrix<f64> {
    assert_eq!(
        state.len(),
        6,
        "othr_observation_jacobian expects a 6-dim state [x, vx, y, vy, z, vz]"
    );

    let x = state[0];
    let vx = state[1];
    let y = state[2];
    let vy = state[3];
    // z and vz are unused by the ground-plane observables but kept in the state.

    let dx = x - transmitter_enu[0];
    let dy = y - transmitter_enu[1];
    let r2 = dx * dx + dy * dy;
    let r = r2.sqrt().max(1e-9);

    // ground_range = sqrt(dx^2 + dy^2)
    //   d/dx = dx/r,  d/dy = dy/r
    let drdx = dx / r;
    let drdy = dy / r;

    // azimuth = atan2(dx, dy)  (clockwise from north)
    //   d(atan2(dx, dy))/dx =  dy / r^2
    //   d(atan2(dx, dy))/dy = -dx / r^2
    let dazdx = dy / r2;
    let dazdy = -dx / r2;

    // doppler = (dx * vx + dy * vy) / r
    //   d/dx  = vx/r - (dx*vx+dy*vy)*dx / r^3
    //   d/dy  = vy/r - (dx*vx+dy*vy)*dy / r^3
    //   d/dvx = dx/r
    //   d/dvy = dy/r
    let num = dx * vx + dy * vy;
    let ddop_dx = vx / r - num * dx / (r2 * r);
    let ddop_dy = vy / r - num * dy / (r2 * r);
    let ddop_dvx = dx / r;
    let ddop_dvy = dy / r;

    let mut h = DMatrix::<f64>::zeros(3, 6);

    // Row 0: ground_range
    h[(0, 0)] = drdx;
    h[(0, 2)] = drdy;

    // Row 1: azimuth
    h[(1, 0)] = dazdx;
    h[(1, 2)] = dazdy;

    // Row 2: doppler
    h[(2, 0)] = ddop_dx;
    h[(2, 1)] = ddop_dvx;
    h[(2, 2)] = ddop_dy;
    h[(2, 3)] = ddop_dvy;

    h
}

/// Build a Cartesian measurement noise matrix for an OTHR detection.
///
/// OTHR has km-scale ground-range uncertainty and degree-scale azimuth
/// uncertainty, which translates to a range-dependent cross-range error of
/// `ground_range * azimuth_sigma_rad`. This helper returns a 3x3 isotropic
/// covariance whose standard deviation is the larger of the two, which is a
/// conservative but positive-definite choice suitable for the existing
/// Cartesian tracker.
///
/// For EKF-style filters that operate directly on (range, azimuth, doppler),
/// prefer constructing the noise matrix in measurement space instead.
pub fn othr_cartesian_noise(
    ground_range_m: f64,
    range_sigma_m: f64,
    azimuth_sigma_rad: f64,
) -> DMatrix<f64> {
    let cross_range = ground_range_m.abs() * azimuth_sigma_rad;
    let sigma = range_sigma_m.max(cross_range);
    DMatrix::identity(3, 3) * (sigma * sigma)
}

#[cfg(test)]
mod tests {
    use super::*;
    use thresh_core::measurement::PropagationMode;

    fn test_registration() -> OthrSensorRegistration {
        OthrSensorRegistration {
            transmitter_lat_rad: 40.0_f64.to_radians(),
            transmitter_lon_rad: (-74.0_f64).to_radians(),
            transmitter_alt_m: 0.0,
            operating_freq_mhz: 15.0,
        }
    }

    #[test]
    fn othr_to_cartesian_returns_none_for_non_othr() {
        let reg = test_registration();
        let m = Measurement::Radar {
            range: 10_000.0,
            azimuth: 0.5,
            elevation: 0.1,
            range_rate: None,
            time: 0.0,
            sensor_id: 0,
        };
        assert!(
            othr_to_cartesian(
                &m,
                &reg,
                10_000.0,
                reg.transmitter_lat_rad,
                reg.transmitter_lon_rad,
                reg.transmitter_alt_m,
            )
            .is_none()
        );
    }

    #[test]
    fn othr_to_cartesian_basic() {
        let reg = test_registration();
        // Target 100 km due north of transmitter.
        let m = Measurement::Othr {
            ground_range_m: 100_000.0,
            azimuth_rad: 0.0,
            doppler_m_s: 0.0,
            propagation_mode: PropagationMode::FLayer,
            time: 0.0,
            sensor_id: 0,
        };

        let det = othr_to_cartesian(
            &m,
            &reg,
            0.0,
            reg.transmitter_lat_rad,
            reg.transmitter_lon_rad,
            reg.transmitter_alt_m,
        )
        .expect("othr measurement should convert");

        assert_eq!(det.len(), 3);
        // East component near zero, north ~100 km.
        assert!(det[0].abs() < 100.0, "east should be ~0, got {}", det[0]);
        assert!(
            (det[1] - 100_000.0).abs() < 1000.0,
            "north should be ~100 km, got {}",
            det[1]
        );
    }

    #[test]
    fn jacobian_has_correct_dimensions() {
        let state = DVector::from_column_slice(&[100_000.0, 200.0, 50_000.0, -50.0, 0.0, 0.0]);
        let tx = [0.0_f64, 0.0, 0.0];
        let h = othr_observation_jacobian(&state, &tx);
        assert_eq!(h.nrows(), 3);
        assert_eq!(h.ncols(), 6);
    }

    #[test]
    fn jacobian_radial_velocity_case() {
        // Target at (0, R) with velocity (0, vy): all motion is radial.
        // Then d(range)/dy = 1, d(azimuth)/dy = 0, d(doppler)/dvy = 1.
        let r = 500_000.0_f64;
        let vy = 150.0;
        let state = DVector::from_column_slice(&[0.0, 0.0, r, vy, 0.0, 0.0]);
        let tx = [0.0, 0.0, 0.0];
        let h = othr_observation_jacobian(&state, &tx);

        assert!((h[(0, 2)] - 1.0).abs() < 1e-9, "d range / dy");
        assert!(h[(0, 0)].abs() < 1e-9, "d range / dx");
        assert!(h[(1, 2)].abs() < 1e-9, "d az / dy");
        assert!((h[(2, 3)] - 1.0).abs() < 1e-9, "d doppler / dvy");
    }

    #[test]
    fn cartesian_noise_is_positive_definite() {
        let r = othr_cartesian_noise(2_000_000.0, 20_000.0, 0.017);
        assert_eq!(r.nrows(), 3);
        assert_eq!(r.ncols(), 3);
        for i in 0..3 {
            assert!(r[(i, i)] > 0.0, "diagonal must be positive");
            for j in 0..3 {
                if i != j {
                    assert_eq!(r[(i, j)], 0.0, "off-diagonal must be zero");
                }
            }
        }
        // Cholesky existence => positive-definite.
        assert!(r.clone().cholesky().is_some());
    }

    #[test]
    fn cartesian_noise_cross_range_dominates_at_long_range() {
        // At 2000 km with 1 deg azimuth, cross-range sigma ~ 34 km > 20 km range sigma.
        let r = othr_cartesian_noise(2_000_000.0, 20_000.0, 0.017);
        let cross = 2_000_000.0_f64 * 0.017;
        let expected = cross * cross;
        assert!((r[(0, 0)] - expected).abs() / expected < 1e-9);
    }
}
