//! Pure-Rust context failures must not require an initialized interpreter.
#![cfg(feature = "stonesoup")]

use nalgebra::{DMatrix, Matrix3, Vector3};
use thresh_bridge::{context::*, error::BridgeError};
use thresh_core::{measurement::Measurement, sensor::SensorRegistration, time::Epoch};

fn sensor() -> SensorPose {
    SensorPose::Stationary(SensorRegistration::new(Vector3::new(10.0, 20.0, 30.0), 7))
}

fn datum() -> AdsbDatum {
    AdsbDatum {
        geoid_separation_m: 20.0,
        noise: AdsbNoiseConvention::MetricEnuAtReport,
    }
}

fn context(profile: ObservationProfile) -> DetectionContext {
    let dimension = profile.dimension();
    DetectionContext {
        origin: Epoch::from_iso8601("2020-01-01T00:00:00 UTC").unwrap(),
        arrival: None,
        frame: CommonFrame::Wgs84Enu(EnuOrigin {
            latitude_rad: 0.5,
            longitude_rad: -0.3,
            ellipsoid_height_m: 50.0,
        }),
        profile,
        covariance: DMatrix::identity(dimension, dimension),
    }
}

fn radar(range_rate: Option<f64>) -> Measurement {
    Measurement::Radar {
        range: 100.0,
        azimuth: 0.2,
        elevation: 0.1,
        range_rate,
        time: 0.0,
        sensor_id: 7,
    }
}

fn adsb(velocity: Option<[f64; 3]>) -> Measurement {
    Measurement::AdsB {
        lat: 10.0,
        lon: 20.0,
        alt: 100.0,
        velocity,
        time: 0.0,
    }
}

fn assert_conversion_error(context: &DetectionContext, measurement: &Measurement) {
    assert!(matches!(
        context.validate(measurement),
        Err(BridgeError::ConversionError(_))
    ));
}

#[test]
fn all_five_context_profiles_validate_without_python() {
    let cases = [
        (ObservationProfile::Radar3 { sensor: sensor() }, radar(None)),
        (
            ObservationProfile::Radar4 {
                sensor: sensor(),
                doppler_sign: DopplerSign::PositiveAway,
            },
            radar(Some(-3.0)),
        ),
        (
            ObservationProfile::Eoir2 { sensor: sensor() },
            Measurement::EoIr {
                azimuth: 0.2,
                elevation: 0.1,
                time: 0.0,
                sensor_id: 7,
            },
        ),
        (ObservationProfile::Adsb3 { datum: datum() }, adsb(None)),
        (
            ObservationProfile::Adsb6 {
                datum: datum(),
                velocity_frame: VelocityFrame::Ecef,
            },
            adsb(Some([1.0, 2.0, 3.0])),
        ),
    ];
    for (profile, measurement) in cases {
        let context = context(profile);
        assert_eq!(context.profile.dimension(), measurement.dim());
        assert_eq!(context.validate(&measurement).unwrap().calendar.year, 2020);
    }
}

#[test]
fn optional_observations_must_match_the_explicit_profile() {
    let radar3 = context(ObservationProfile::Radar3 { sensor: sensor() });
    let radar4 = context(ObservationProfile::Radar4 {
        sensor: sensor(),
        doppler_sign: DopplerSign::PositiveTowards,
    });
    let adsb3 = context(ObservationProfile::Adsb3 { datum: datum() });
    let adsb6 = context(ObservationProfile::Adsb6 {
        datum: datum(),
        velocity_frame: VelocityFrame::EnuAtReport,
    });
    assert_conversion_error(&radar3, &radar(Some(1.0)));
    assert_conversion_error(&radar4, &radar(None));
    assert_conversion_error(&adsb3, &adsb(Some([0.0; 3])));
    assert_conversion_error(&adsb6, &adsb(None));
    assert_conversion_error(&radar3, &adsb(None));
}

#[test]
fn unsupported_frames_and_motion_are_explicit_errors() {
    let mut context = context(ObservationProfile::Radar3 { sensor: sensor() });
    for frame in [CommonFrame::Inertial, CommonFrame::Unspecified] {
        context.frame = frame;
        assert_conversion_error(&context, &radar(None));
    }
    let mut moving = super_context();
    moving.profile = ObservationProfile::Radar3 {
        sensor: SensorPose::Moving(SensorRegistration::new(Vector3::zeros(), 7)),
    };
    assert_conversion_error(&moving, &radar(None));
}

fn super_context() -> DetectionContext {
    context(ObservationProfile::Radar3 { sensor: sensor() })
}

#[test]
fn sensor_identity_and_nonfinite_position_are_rejected() {
    let invalid_sensors = [
        SensorRegistration::new(Vector3::zeros(), 8),
        SensorRegistration::new(Vector3::new(f64::NAN, 0.0, 0.0), 7),
        SensorRegistration::new(Vector3::new(0.0, f64::INFINITY, 0.0), 7),
    ];
    for registration in invalid_sensors {
        let context = context(ObservationProfile::Radar3 {
            sensor: SensorPose::Stationary(registration),
        });
        assert_conversion_error(&context, &radar(None));
    }
}

#[test]
fn rotations_must_be_proper_finite_and_orthogonal() {
    let invalid = [
        Matrix3::new(-1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
        Matrix3::identity() * 2.0,
        Matrix3::from_element(f64::MAX),
        Matrix3::from_element(f64::NAN),
        Matrix3::zeros(),
    ];
    for rotation in invalid {
        let registration = SensorRegistration::with_rotation(Vector3::zeros(), rotation, 7);
        let context = context(ObservationProfile::Radar3 {
            sensor: SensorPose::Stationary(registration),
        });
        assert_conversion_error(&context, &radar(None));
    }
    let rotated = SensorRegistration::with_rotation(
        Vector3::zeros(),
        Matrix3::new(0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0),
        7,
    );
    assert!(
        context(ObservationProfile::Radar3 {
            sensor: SensorPose::Stationary(rotated)
        })
        .validate(&radar(None))
        .is_ok()
    );
}

#[test]
fn source_ranges_and_nonfinite_observations_are_rejected() {
    let invalid = [
        (-1.0, 0.2, 0.1),
        (100.0, f64::NAN, 0.1),
        (100.0, 0.2, 2.0),
        (f64::INFINITY, 0.2, 0.1),
        (100.0, 0.2, f64::NEG_INFINITY),
    ];
    let context = super_context();
    for (range, azimuth, elevation) in invalid {
        let measurement = Measurement::Radar {
            range,
            azimuth,
            elevation,
            range_rate: None,
            time: 0.0,
            sensor_id: 7,
        };
        assert_conversion_error(&context, &measurement);
    }
    let wrapped = Measurement::Radar {
        range: 0.0,
        azimuth: 10_000.0,
        elevation: -std::f64::consts::FRAC_PI_2,
        range_rate: None,
        time: 0.0,
        sensor_id: 7,
    };
    assert!(context.validate(&wrapped).is_ok());
}

#[test]
fn adsb_requires_metric_noise_finite_datum_and_valid_degrees() {
    for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let context = context(ObservationProfile::Adsb3 {
            datum: AdsbDatum {
                geoid_separation_m: value,
                ..datum()
            },
        });
        assert_conversion_error(&context, &adsb(None));
    }
    let invalid_noise = context(ObservationProfile::Adsb3 {
        datum: AdsbDatum {
            noise: AdsbNoiseConvention::GeodeticDegrees,
            ..datum()
        },
    });
    assert_conversion_error(&invalid_noise, &adsb(None));
    let context = context(ObservationProfile::Adsb3 { datum: datum() });
    for (lat, lon, alt) in [
        (91.0, 0.0, 0.0),
        (0.0, -181.0, 0.0),
        (f64::NAN, 0.0, 0.0),
        (0.0, 0.0, f64::INFINITY),
    ] {
        assert_conversion_error(
            &context,
            &Measurement::AdsB {
                lat,
                lon,
                alt,
                velocity: None,
                time: 0.0,
            },
        );
    }
}

#[test]
fn overflowing_finite_datum_addition_and_nonfinite_velocity_are_rejected() {
    let context = context(ObservationProfile::Adsb3 {
        datum: AdsbDatum {
            geoid_separation_m: f64::MAX,
            ..datum()
        },
    });
    let overflow = Measurement::AdsB {
        lat: 0.0,
        lon: 0.0,
        alt: f64::MAX,
        velocity: None,
        time: 0.0,
    };
    assert_conversion_error(&context, &overflow);
    let velocity_context = self::context(ObservationProfile::Adsb6 {
        datum: datum(),
        velocity_frame: VelocityFrame::Ecef,
    });
    assert_conversion_error(&velocity_context, &adsb(Some([0.0, f64::NAN, 0.0])));
}

#[test]
fn frame_origin_must_be_finite_and_have_valid_radian_ranges() {
    let mut context = super_context();
    for (latitude_rad, longitude_rad, ellipsoid_height_m) in [
        (2.0, 0.0, 0.0),
        (0.0, 4.0, 0.0),
        (f64::NAN, 0.0, 0.0),
        (0.0, 0.0, f64::MAX * 2.0),
    ] {
        context.frame = CommonFrame::Wgs84Enu(EnuOrigin {
            latitude_rad,
            longitude_rad,
            ellipsoid_height_m,
        });
        assert_conversion_error(&context, &radar(None));
    }
}

#[test]
fn covariance_is_required_and_never_defaulted() {
    let mut context = super_context();
    context.covariance = DMatrix::zeros(0, 0);
    assert_conversion_error(&context, &radar(None));
    context.covariance = DMatrix::identity(2, 2);
    assert_conversion_error(&context, &radar(None));
    context.covariance = DMatrix::identity(3, 3);
    context.covariance[(0, 0)] = -1.0;
    assert_conversion_error(&context, &radar(None));
}

#[test]
fn delayed_arrival_preserves_independent_metadata_and_observation_order() {
    let mut context = super_context();
    context.arrival = Some(context.origin + 20.0);
    let mut newer = radar(None);
    if let Measurement::Radar { time, .. } = &mut newer {
        *time = 10.0;
    }
    assert_eq!(context.validate(&newer).unwrap().calendar.second, 10);
    let delayed = context.validate(&radar(None)).unwrap();
    assert_eq!(delayed.calendar.second, 0);
    assert_eq!(delayed.relative_seconds, 0.0);
    assert_eq!(delayed.origin_utc, context.origin.to_string());
    assert_eq!(
        delayed.arrival_utc,
        Some((context.origin + 20.0).to_string())
    );
}
