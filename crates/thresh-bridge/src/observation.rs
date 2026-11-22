//! Rust-only observation normalization. Context validation precedes this phase.

use nalgebra::{DMatrix, DVector, Matrix3, Rotation3, Vector3};
use thresh_core::{geodetic, measurement::Measurement, sensor::SensorRegistration};

use crate::context::{
    CommonFrame, DetectionContext, DopplerSign, ObservationProfile, VelocityFrame,
};
use crate::covariance::validate_covariance;
use crate::error::{BridgeError, BridgeResult};

pub(crate) struct Observation {
    pub values: DVector<f64>,
    pub covariance: DMatrix<f64>,
    pub model: Model,
}

pub(crate) enum Model {
    Spherical {
        class: &'static str,
        position: Vector3<f64>,
        rotation: Vector3<f64>,
        doppler: bool,
    },
    Cartesian {
        mapping: Vec<usize>,
    },
}

/// Requires a matching, validated measurement/context pair. Still checks every
/// derived numeric result: finite input can overflow during normalization.
pub(crate) fn normalize(
    measurement: &Measurement,
    context: &DetectionContext,
) -> BridgeResult<Observation> {
    let observation = match measurement {
        Measurement::Radar { .. } | Measurement::EoIr { .. } => {
            normalize_spherical(measurement, context)?
        }
        Measurement::AdsB { .. } => normalize_adsb(measurement, context)?,
        _ => return Err(invalid("unsupported measurement variant")),
    };
    if !observation.values.iter().all(|v| v.is_finite()) {
        return Err(invalid("nonfinite normalized observation"));
    }
    validate_covariance(&observation.covariance, observation.values.len())?;
    validate_model_numbers(&observation.model)?;
    Ok(observation)
}

fn validate_model_numbers(model: &Model) -> BridgeResult<()> {
    if let Model::Spherical {
        position, rotation, ..
    } = model
        && !position
            .iter()
            .chain(rotation.iter())
            .all(|v| v.is_finite())
    {
        return Err(invalid("nonfinite normalized sensor pose"));
    }
    Ok(())
}

fn invalid(message: &str) -> BridgeError {
    BridgeError::ConversionError(message.to_owned())
}

fn normalize_spherical(
    measurement: &Measurement,
    context: &DetectionContext,
) -> BridgeResult<Observation> {
    let sensor = context
        .profile
        .sensor()
        .ok_or_else(|| invalid("spherical observation requires sensor pose"))?;
    let (values, order, signs, class) = spherical_components(measurement, &context.profile)?;
    let covariance = DMatrix::from_fn(order.len(), order.len(), |row, col| {
        context.covariance[(order[row], order[col])] * signs[row] * signs[col]
    });
    Ok(Observation {
        model: spherical_model(sensor, class, values.len() == 4)?,
        values: DVector::from_vec(values),
        covariance,
    })
}

type SphericalComponents = (Vec<f64>, Vec<usize>, Vec<f64>, &'static str);

fn spherical_components(
    measurement: &Measurement,
    profile: &ObservationProfile,
) -> BridgeResult<SphericalComponents> {
    match measurement {
        Measurement::Radar {
            range,
            azimuth,
            elevation,
            range_rate,
            ..
        } => radar_components(*range, *azimuth, *elevation, *range_rate, profile),
        Measurement::EoIr {
            azimuth, elevation, ..
        } => Ok((
            vec![*elevation, normalize_azimuth(*azimuth)],
            vec![1, 0],
            vec![1.0; 2],
            "CartesianToElevationBearing",
        )),
        _ => Err(invalid("expected radar or EO/IR measurement")),
    }
}

fn radar_components(
    range: f64,
    azimuth: f64,
    elevation: f64,
    range_rate: Option<f64>,
    profile: &ObservationProfile,
) -> BridgeResult<SphericalComponents> {
    let mut values = vec![elevation, normalize_azimuth(azimuth), range];
    let mut order = vec![2, 1, 0];
    let mut signs = vec![1.0; 3];
    let Some(rate) = range_rate else {
        return Ok((values, order, signs, "CartesianToElevationBearingRange"));
    };
    let sign = match profile {
        ObservationProfile::Radar4 {
            doppler_sign: DopplerSign::PositiveAway,
            ..
        } => 1.0,
        ObservationProfile::Radar4 {
            doppler_sign: DopplerSign::PositiveTowards,
            ..
        } => -1.0,
        _ => {
            return Err(invalid(
                "radar Doppler requires an explicit sign convention",
            ));
        }
    };
    values.push(sign * rate);
    order.push(3);
    signs.push(sign);
    Ok((values, order, signs, "CartesianToElevationBearingRangeRate"))
}

fn normalize_azimuth(angle: f64) -> f64 {
    // Reduce before shifting: adding pi first could overflow a finite input.
    let wrapped = angle.rem_euclid(std::f64::consts::TAU);
    if wrapped >= std::f64::consts::PI {
        wrapped - std::f64::consts::TAU
    } else {
        wrapped
    }
}

fn spherical_model(
    sensor: &SensorRegistration,
    class: &'static str,
    doppler: bool,
) -> BridgeResult<Model> {
    // nalgebra: sensor->common = Rz(yaw) Ry(pitch) Rx(roll).
    // Stone Soup: common->sensor = Rx(-roll) Ry(offset_y) Rz(-yaw).
    // Hence its offset vector is [roll, -pitch, yaw], including gimbal lock.
    let angles = checked_sensor_angles(&sensor.rotation)?;
    Ok(Model::Spherical {
        class,
        position: sensor.position,
        rotation: Vector3::new(angles.x, -angles.y, angles.z),
        doppler,
    })
}

/// Requires the proper-rotation validation performed by DetectionContext.
/// Retain the small cosine terms near pitch lock: asin(-r20) loses them when
/// r20 rounds to +/-1, or magnifies tiny tolerated matrix errors into angle errors.
fn checked_sensor_angles(rotation: &Matrix3<f64>) -> BridgeResult<Vector3<f64>> {
    if !rotation.iter().all(|value| value.is_finite()) {
        return Err(invalid("nonfinite sensor rotation before Euler conversion"));
    }
    let angles = sensor_euler_angles(rotation);
    validate_rotation_roundtrip(rotation, &angles)?;
    Ok(angles)
}

fn sensor_euler_angles(rotation: &Matrix3<f64>) -> Vector3<f64> {
    let horizontal = rotation[(0, 0)].hypot(rotation[(1, 0)]);
    let pitch = (-rotation[(2, 0)]).atan2(horizontal);
    if horizontal > 8.0 * f64::EPSILON {
        return Vector3::new(
            rotation[(2, 1)].atan2(rotation[(2, 2)]),
            pitch,
            rotation[(1, 0)].atan2(rotation[(0, 0)]),
        );
    }
    // At lock, roll and yaw are coupled. Choose yaw=0 and recover the coupled
    // angle from the well-conditioned entries; this also handles negative lock.
    let roll = if rotation[(2, 0)] < 0.0 {
        rotation[(0, 1)].atan2(rotation[(0, 2)])
    } else {
        (-rotation[(0, 1)]).atan2(-rotation[(0, 2)])
    };
    Vector3::new(roll, pitch, 0.0)
}

fn validate_rotation_roundtrip(rotation: &Matrix3<f64>, angles: &Vector3<f64>) -> BridgeResult<()> {
    if !angles.iter().all(|value| value.is_finite()) {
        return Err(invalid("nonfinite Euler angles for sensor rotation"));
    }
    let rebuilt = Rotation3::from_euler_angles(angles.x, angles.y, angles.z);
    // Match the context's documented proper-rotation tolerance. Do not project
    // or silently accept a pose which Euler encoding changes beyond that bound.
    const ROTATION_TOLERANCE: f64 = 1e-10;
    if (rebuilt.matrix() - rotation)
        .iter()
        .any(|difference| !difference.is_finite() || difference.abs() > ROTATION_TOLERANCE)
    {
        return Err(invalid(
            "sensor rotation cannot be preserved by model Euler offsets",
        ));
    }
    Ok(())
}

fn normalize_adsb(
    measurement: &Measurement,
    context: &DetectionContext,
) -> BridgeResult<Observation> {
    let Measurement::AdsB {
        lat,
        lon,
        alt,
        velocity,
        ..
    } = measurement
    else {
        return Err(invalid("expected ADS-B measurement"));
    };
    let CommonFrame::Wgs84Enu(origin) = &context.frame else {
        return Err(invalid("ADS-B requires fixed WGS84 ENU frame"));
    };
    let datum = context
        .profile
        .datum()
        .ok_or_else(|| invalid("missing ADS-B datum"))?;
    let height = alt + datum.geoid_separation_m;
    if !height.is_finite() {
        return Err(invalid("MSL to ellipsoid height overflow"));
    }
    let latitude = lat.to_radians();
    let longitude = lon.to_radians();
    let position = geodetic::wgs84_to_enu(
        latitude,
        longitude,
        height,
        origin.latitude_rad,
        origin.longitude_rad,
        origin.ellipsoid_height_m,
    );
    let common_rotation = ecef_to_enu_rotation(origin.latitude_rad, origin.longitude_rad);
    let position_rotation = common_rotation * ecef_to_enu_rotation(latitude, longitude).transpose();
    let (values, transform, mapping) = adsb_components(
        position,
        *velocity,
        &context.profile,
        common_rotation,
        position_rotation,
    )?;
    let first = &transform * &context.covariance;
    if !first.iter().all(|v| v.is_finite()) {
        return Err(invalid("ADS-B covariance transform overflow"));
    }
    let covariance = first * transform.transpose();
    Ok(Observation {
        values,
        covariance,
        model: Model::Cartesian { mapping },
    })
}

fn adsb_components(
    position: Vector3<f64>,
    velocity: Option<[f64; 3]>,
    profile: &ObservationProfile,
    common_rotation: Matrix3<f64>,
    position_rotation: Matrix3<f64>,
) -> BridgeResult<(DVector<f64>, DMatrix<f64>, Vec<usize>)> {
    let mut values = position.as_slice().to_vec();
    let Some(velocity) = velocity else {
        return Ok((
            DVector::from_vec(values),
            DMatrix::from_column_slice(3, 3, position_rotation.as_slice()),
            vec![0, 2, 4],
        ));
    };
    let velocity_rotation = match profile {
        ObservationProfile::Adsb6 {
            velocity_frame: VelocityFrame::Ecef,
            ..
        } => common_rotation,
        ObservationProfile::Adsb6 {
            velocity_frame: VelocityFrame::EnuAtReport,
            ..
        } => position_rotation,
        _ => return Err(invalid("ADS-B velocity requires a declared frame")),
    };
    let normalized_velocity = velocity_rotation * Vector3::from_column_slice(&velocity);
    values.extend_from_slice(normalized_velocity.as_slice());
    let mut transform = DMatrix::zeros(6, 6);
    transform
        .view_mut((0, 0), (3, 3))
        .copy_from(&position_rotation);
    transform
        .view_mut((3, 3), (3, 3))
        .copy_from(&velocity_rotation);
    Ok((DVector::from_vec(values), transform, vec![0, 2, 4, 1, 3, 5]))
}

fn ecef_to_enu_rotation(latitude: f64, longitude: f64) -> Matrix3<f64> {
    let (slat, clat) = latitude.sin_cos();
    let (slon, clon) = longitude.sin_cos();
    Matrix3::new(
        -slon,
        clon,
        0.0,
        -slat * clon,
        -slat * slon,
        clat,
        clat * clon,
        clat * slon,
        slat,
    )
}

#[cfg(test)]
mod pose_tests {
    use super::*;

    fn assert_preserved(rotation: Matrix3<f64>, tolerance: f64) {
        let sensor = SensorRegistration::with_rotation(Vector3::zeros(), rotation, 7);
        let Model::Spherical {
            rotation: offsets, ..
        } = spherical_model(&sensor, "CartesianToElevationBearingRange", false).unwrap()
        else {
            panic!("expected spherical model");
        };
        assert!(offsets.iter().all(|value| value.is_finite()));
        let rebuilt = Rotation3::from_euler_angles(offsets.x, -offsets.y, offsets.z);
        assert!(
            (rebuilt.matrix() - rotation)
                .iter()
                .all(|difference| difference.abs() <= tolerance)
        );
    }

    #[test]
    fn pose_euler_preserves_multiaxis_rotations() {
        for (roll, pitch, yaw) in [(0.3, -0.4, 0.7), (-1.1, 0.9, -2.0), (2.1, -2.0, 0.6)] {
            assert_preserved(
                Rotation3::from_euler_angles(roll, pitch, yaw).into_inner(),
                1e-14,
            );
        }
    }

    #[test]
    fn pose_euler_preserves_both_exact_gimbal_locks() {
        for pitch in [std::f64::consts::FRAC_PI_2, -std::f64::consts::FRAC_PI_2] {
            for (roll, yaw) in [(0.0, 0.4), (0.7, -0.9), (-2.1, 1.8)] {
                assert_preserved(
                    Rotation3::from_euler_angles(roll, pitch, yaw).into_inner(),
                    1e-14,
                );
            }
        }
    }

    #[test]
    fn pose_euler_preserves_both_near_gimbal_locks() {
        for pitch in [
            std::f64::consts::FRAC_PI_2 - 1e-8,
            -std::f64::consts::FRAC_PI_2 + 1e-8,
        ] {
            for (roll, yaw) in [(0.0, 0.4), (0.7, -0.9), (-2.1, 1.8)] {
                assert_preserved(
                    Rotation3::from_euler_angles(roll, pitch, yaw).into_inner(),
                    1e-14,
                );
            }
        }
    }

    #[test]
    fn pose_euler_does_not_amplify_tolerated_near_lock_matrix_error() {
        for sign in [1.0_f64, -1.0] {
            let rotation = Matrix3::new(
                0.0,
                0.0,
                sign,
                0.0,
                1.0,
                0.0,
                -sign * (1.0 - 2e-11),
                0.0,
                0.0,
            );
            assert!(
                (rotation.transpose() * rotation - Matrix3::identity())
                    .iter()
                    .all(|v| v.abs() < 1e-10)
            );
            assert!((rotation.determinant() - 1.0).abs() < 1e-10);
            assert_preserved(rotation, 2.1e-11);
        }
    }

    #[test]
    fn pose_euler_rejects_nonfinite_or_unpreservable_rotation() {
        for rotation in [
            Matrix3::from_element(f64::NAN),
            Matrix3::from_element(f64::INFINITY),
            Matrix3::identity() * 1.01,
            Matrix3::zeros(),
        ] {
            assert!(matches!(
                checked_sensor_angles(&rotation),
                Err(BridgeError::ConversionError(_))
            ));
        }
    }
}
