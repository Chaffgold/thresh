//! Explicit coordinate, time, geometry, and noise contracts for observations.
//!
//! Required information has no implicit default. See the executable examples
//! below for all five profiles and compile-time rejection of missing fields.
#![doc = include_str!("../../../docs/bridge-context.md")]

use std::f64::consts::{FRAC_PI_2, PI};

use nalgebra::{DMatrix, Matrix3};
use thresh_core::{measurement::Measurement, sensor::SensorRegistration, time::Epoch};

use crate::covariance::validate_covariance;
use crate::error::{BridgeError, BridgeResult};
pub use crate::time::{ObservationTime, UtcDateTime};

/// WGS84 origin of the fixed common east-north-up frame.
#[derive(Debug, Clone, Copy)]
pub struct EnuOrigin {
    /// Geodetic latitude in radians, within [-pi/2, pi/2].
    pub latitude_rad: f64,
    /// Longitude in radians, within [-pi, pi].
    pub longitude_rad: f64,
    /// Height above the WGS84 ellipsoid in metres, not MSL height.
    pub ellipsoid_height_m: f64,
}

/// Declared common coordinate frame. Only the fixed WGS84 ENU profile is supported.
#[derive(Debug, Clone, Copy)]
pub enum CommonFrame {
    /// Fixed local frame with state order `[x,vx,y,vy,z,vz]` in metres and m/s.
    Wgs84Enu(EnuOrigin),
    /// Explicit unsupported frame, rejected without assuming an ENU origin.
    Inertial,
    /// Explicit missing frame, rejected rather than assigned Cartesian units.
    Unspecified,
}

/// Sensor pose and motion declaration; moving platforms are not yet supported.
#[derive(Debug, Clone)]
pub enum SensorPose {
    /// Sensor is stationary relative to the fixed common ENU frame.
    Stationary(SensorRegistration),
    /// Unsupported moving sensor; no zero sensor velocity may be inferred.
    Moving(SensorRegistration),
}

/// Sign convention of a measured radar range rate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DopplerSign {
    /// Positive means range is increasing.
    PositiveAway,
    /// Positive means range is decreasing.
    PositiveTowards,
}

/// Axes for an ADS-B report's optional velocity and corresponding noise block.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VelocityFrame {
    /// Earth-centred, Earth-fixed Cartesian axes.
    Ecef,
    /// Local east, north, up axes at the reported position, not the frame origin.
    EnuAtReport,
}

/// Position-error convention for ADS-B source covariance.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AdsbNoiseConvention {
    /// Metric east/north/up position errors at the reported geodetic position.
    MetricEnuAtReport,
    /// Unsupported degree-valued position errors; never interpreted as metres.
    GeodeticDegrees,
}

/// Information absent from core's ADS-B type and therefore supplied by its caller.
#[derive(Debug, Clone, Copy)]
pub struct AdsbDatum {
    /// Ellipsoid height minus MSL height, in metres, at the reported position.
    pub geoid_separation_m: f64,
    /// Explicit units and axes of the source position-error covariance.
    pub noise: AdsbNoiseConvention,
}

/// A complete source observation profile; optional observations select dimensions.
#[derive(Debug, Clone)]
pub enum ObservationProfile {
    /// Source order `[range, azimuth, elevation]` (metres, radians, radians).
    Radar3 {
        /// Required fixed sensor registration.
        sensor: SensorPose,
    },
    /// Source order `[range, azimuth, elevation, range_rate]`.
    Radar4 {
        /// Required fixed sensor registration.
        sensor: SensorPose,
        /// Explicit sign convention of the measured Doppler component.
        doppler_sign: DopplerSign,
    },
    /// Source order `[azimuth, elevation]` in radians.
    Eoir2 {
        /// Required fixed sensor registration.
        sensor: SensorPose,
    },
    /// Position-only cooperative report; noise order is metric report-local ENU.
    Adsb3 {
        /// Required vertical datum correction and source noise convention.
        datum: AdsbDatum,
    },
    /// Position and velocity; noise order is report-local ENU position then velocity.
    Adsb6 {
        /// Required vertical datum correction and source noise convention.
        datum: AdsbDatum,
        /// Required axes for measured velocity and the velocity noise block.
        velocity_frame: VelocityFrame,
    },
}

impl ObservationProfile {
    /// Number of measured components; absent observations are not filled with zero.
    pub fn dimension(&self) -> usize {
        match self {
            Self::Radar3 { .. } | Self::Adsb3 { .. } => 3,
            Self::Radar4 { .. } => 4,
            Self::Eoir2 { .. } => 2,
            Self::Adsb6 { .. } => 6,
        }
    }

    /// Sensor registration, when applicable. Validate before using its pose.
    pub fn sensor(&self) -> Option<&SensorRegistration> {
        let pose = match self {
            Self::Radar3 { sensor } | Self::Radar4 { sensor, .. } | Self::Eoir2 { sensor } => {
                sensor
            }
            Self::Adsb3 { .. } | Self::Adsb6 { .. } => return None,
        };
        match pose {
            SensorPose::Stationary(sensor) | SensorPose::Moving(sensor) => Some(sensor),
        }
    }

    /// ADS-B datum/noise declaration, when applicable.
    pub fn datum(&self) -> Option<&AdsbDatum> {
        match self {
            Self::Adsb3 { datum } | Self::Adsb6 { datum, .. } => Some(datum),
            _ => None,
        }
    }
}

/// Required inputs for conversion; no origin, geometry, covariance, or datum defaults.
#[derive(Debug, Clone)]
pub struct DetectionContext {
    /// Absolute origin of the measurement's relative physical seconds.
    pub origin: Epoch,
    /// Optional separate arrival epoch; never substitutes for observation time.
    pub arrival: Option<Epoch>,
    /// Fixed common frame with interleaved six-state mapping `[x,vx,y,vy,z,vz]`.
    pub frame: CommonFrame,
    /// Source-specific geometry, datum, and optional-observation contract.
    pub profile: ObservationProfile,
    /// Full covariance in the profile's source component order, including cross terms.
    pub covariance: DMatrix<f64>,
}

impl DetectionContext {
    /// Validate context, source values and time entirely in Rust.
    ///
    /// Unsupported measurement variants win over other invalid context. Source
    /// covariance is checked but never repaired, symmetrized, or defaulted.
    /// Coordinate/model construction must additionally check derived outputs.
    pub fn validate(&self, measurement: &Measurement) -> BridgeResult<ObservationTime> {
        validate_supported_measurement(measurement)?;
        validate_frame(self.frame)?;
        validate_profile(&self.profile, measurement)?;
        validate_source_values(measurement)?;
        validate_covariance(&self.covariance, self.profile.dimension())?;
        crate::time::observation_time(self.origin, measurement.time(), self.arrival)
    }
}

fn context_error(message: impl Into<String>) -> BridgeError {
    BridgeError::ConversionError(format!("observation context: {}", message.into()))
}

fn validate_supported_measurement(measurement: &Measurement) -> BridgeResult<()> {
    let unsupported = match measurement {
        Measurement::Othr { .. } => "OTHR",
        Measurement::Lidar { .. } => "LiDAR",
        Measurement::Camera { .. } => "Camera",
        _ => return Ok(()),
    };
    Err(context_error(format!(
        "unsupported measurement: {unsupported}"
    )))
}

fn validate_frame(frame: CommonFrame) -> BridgeResult<()> {
    let CommonFrame::Wgs84Enu(origin) = frame else {
        return Err(context_error(
            "only an explicit fixed WGS84 ENU frame is supported",
        ));
    };
    validate_finite(&[
        origin.latitude_rad,
        origin.longitude_rad,
        origin.ellipsoid_height_m,
    ])?;
    if !(-FRAC_PI_2..=FRAC_PI_2).contains(&origin.latitude_rad)
        || !(-PI..=PI).contains(&origin.longitude_rad)
    {
        return Err(context_error(
            "ENU origin latitude/longitude are outside their radian ranges",
        ));
    }
    Ok(())
}

fn validate_profile(profile: &ObservationProfile, measurement: &Measurement) -> BridgeResult<()> {
    match (profile, measurement) {
        (
            ObservationProfile::Radar3 { sensor },
            Measurement::Radar {
                range_rate: None,
                sensor_id,
                ..
            },
        )
        | (
            ObservationProfile::Radar4 { sensor, .. },
            Measurement::Radar {
                range_rate: Some(_),
                sensor_id,
                ..
            },
        )
        | (ObservationProfile::Eoir2 { sensor }, Measurement::EoIr { sensor_id, .. }) => {
            validate_sensor(sensor, *sensor_id)
        }
        (
            ObservationProfile::Adsb3 { datum },
            Measurement::AdsB {
                velocity: None,
                alt,
                ..
            },
        )
        | (
            ObservationProfile::Adsb6 { datum, .. },
            Measurement::AdsB {
                velocity: Some(_),
                alt,
                ..
            },
        ) => validate_datum(*datum, *alt),
        _ => Err(context_error(
            "measurement and profile disagree, including optional Doppler/velocity",
        )),
    }
}

fn validate_sensor(pose: &SensorPose, sensor_id: u32) -> BridgeResult<()> {
    let SensorPose::Stationary(sensor) = pose else {
        return Err(context_error("moving sensor geometry is not supported"));
    };
    if sensor.sensor_id != sensor_id {
        return Err(context_error(
            "measurement sensor identity conflicts with its registration",
        ));
    }
    validate_finite(sensor.position.as_slice())?;
    validate_rotation(&sensor.rotation)
}

fn validate_rotation(rotation: &Matrix3<f64>) -> BridgeResult<()> {
    validate_finite(rotation.as_slice())?;
    const TOLERANCE: f64 = 1e-10;
    // Reject extreme entries before matrix multiplication can overflow.
    if rotation.iter().any(|entry| entry.abs() > 1.0 + TOLERANCE) {
        return Err(context_error(
            "sensor rotation is not a proper orthogonal matrix",
        ));
    }
    let orthogonality = rotation.transpose() * rotation - Matrix3::identity();
    if orthogonality.iter().any(|entry| entry.abs() > TOLERANCE)
        || (rotation.determinant() - 1.0).abs() > TOLERANCE
    {
        return Err(context_error(
            "sensor rotation is not a proper orthogonal matrix",
        ));
    }
    Ok(())
}

fn validate_datum(datum: AdsbDatum, msl_height: f64) -> BridgeResult<()> {
    validate_finite(&[datum.geoid_separation_m, msl_height])?;
    if datum.noise != AdsbNoiseConvention::MetricEnuAtReport {
        return Err(context_error(
            "ADS-B noise must be metric ENU at the report position, not degrees",
        ));
    }
    if !(msl_height + datum.geoid_separation_m).is_finite() {
        return Err(context_error(
            "ADS-B MSL-to-ellipsoid datum addition overflowed",
        ));
    }
    Ok(())
}

fn validate_source_values(measurement: &Measurement) -> BridgeResult<()> {
    validate_finite(measurement.to_vector().as_slice())?;
    match measurement {
        Measurement::Radar {
            range, elevation, ..
        } => {
            if *range < 0.0 {
                return Err(context_error("radar range must be nonnegative"));
            }
            validate_elevation(*elevation)
        }
        Measurement::EoIr { elevation, .. } => validate_elevation(*elevation),
        Measurement::AdsB { lat, lon, .. } => validate_latitude_longitude(*lat, *lon),
        _ => Ok(()),
    }
}

fn validate_elevation(elevation: f64) -> BridgeResult<()> {
    if !(-FRAC_PI_2..=FRAC_PI_2).contains(&elevation) {
        return Err(context_error(
            "elevation must be within [-pi/2, pi/2] radians",
        ));
    }
    Ok(())
}

fn validate_latitude_longitude(latitude: f64, longitude: f64) -> BridgeResult<()> {
    if !(-90.0..=90.0).contains(&latitude) || !(-180.0..=180.0).contains(&longitude) {
        return Err(context_error(
            "ADS-B latitude/longitude are outside their degree ranges",
        ));
    }
    Ok(())
}

fn validate_finite(values: &[f64]) -> BridgeResult<()> {
    if values.iter().any(|value| !value.is_finite()) {
        return Err(context_error(
            "numeric source/context values must be finite",
        ));
    }
    Ok(())
}
