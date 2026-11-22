//! End-to-end, wholly invented observations against the pinned Stone Soup API.
#![cfg(feature = "stonesoup")]

use nalgebra::{DMatrix, DVector, Matrix3, Rotation3, Vector3};
use pyo3::prelude::*;
use thresh_bridge::{context::*, convert, detection::measurement_to_detection, error::BridgeError};
use thresh_core::{
    measurement::{Measurement, PropagationMode},
    sensor::SensorRegistration,
    time::Epoch,
    track::TargetClass,
};

fn registration(rotation: Matrix3<f64>) -> SensorRegistration {
    SensorRegistration::with_rotation(Vector3::new(10.0, 20.0, 30.0), rotation, 7)
}

fn sensor() -> SensorPose {
    SensorPose::Stationary(registration(Matrix3::identity()))
}

fn datum() -> AdsbDatum {
    AdsbDatum {
        geoid_separation_m: 20.0,
        noise: AdsbNoiseConvention::MetricEnuAtReport,
    }
}

fn context(profile: ObservationProfile) -> DetectionContext {
    let n = profile.dimension();
    DetectionContext {
        origin: Epoch::from_iso8601("2020-01-01T00:00:00 UTC").unwrap(),
        arrival: None,
        frame: CommonFrame::Wgs84Enu(EnuOrigin {
            latitude_rad: 0.0,
            longitude_rad: 0.0,
            ellipsoid_height_m: 0.0,
        }),
        profile,
        covariance: DMatrix::identity(n, n),
    }
}

fn radar(rate: Option<f64>, time: f64) -> Measurement {
    Measurement::Radar {
        range: 100.0,
        azimuth: 0.2,
        elevation: 0.1,
        range_rate: rate,
        time,
        sensor_id: 7,
    }
}

fn adsb(longitude: f64, velocity: Option<[f64; 3]>) -> Measurement {
    Measurement::AdsB {
        lat: 0.0,
        lon: longitude,
        alt: 100.0,
        velocity,
        time: 1.25,
    }
}

fn detect(measurement: &Measurement, context: &DetectionContext) -> Py<PyAny> {
    Python::initialize();
    measurement_to_detection(measurement, context).unwrap()
}

fn values(py: Python<'_>, detection: &Bound<'_, PyAny>) -> DVector<f64> {
    convert::numpy_to_dvector(py, &detection.getattr("state_vector").unwrap()).unwrap()
}

fn model_covariance(py: Python<'_>, detection: &Bound<'_, PyAny>) -> DMatrix<f64> {
    let model = detection.getattr("measurement_model").unwrap();
    convert::numpy_to_dmatrix(py, &model.call_method0("covar").unwrap()).unwrap()
}

fn assert_common(
    py: Python<'_>,
    detection: &Bound<'_, PyAny>,
    expected: &[f64],
    mapping: &[usize],
) {
    assert!((values(py, detection) - DVector::from_column_slice(expected)).norm() < 1e-8);
    let vector = detection.getattr("state_vector").unwrap();
    assert_eq!(
        vector
            .getattr("shape")
            .unwrap()
            .extract::<(usize, usize)>()
            .unwrap(),
        (expected.len(), 1)
    );
    assert_eq!(
        vector
            .getattr("dtype")
            .unwrap()
            .str()
            .unwrap()
            .to_str()
            .unwrap(),
        "float64"
    );
    let model = detection.getattr("measurement_model").unwrap();
    assert_eq!(
        model
            .getattr("ndim_state")
            .unwrap()
            .extract::<usize>()
            .unwrap(),
        6
    );
    assert_eq!(
        model
            .getattr("ndim_meas")
            .unwrap()
            .extract::<usize>()
            .unwrap(),
        expected.len()
    );
    assert_eq!(
        model
            .getattr("mapping")
            .unwrap()
            .extract::<Vec<usize>>()
            .unwrap(),
        mapping
    );
    let timestamp = detection.getattr("timestamp").unwrap();
    assert_eq!(
        timestamp
            .call_method0("utcoffset")
            .unwrap()
            .call_method0("total_seconds")
            .unwrap()
            .extract::<f64>()
            .unwrap(),
        0.0
    );
}

#[test]
fn radar3_has_absolute_time_model_and_correlated_covariance() {
    let mut context = context(ObservationProfile::Radar3 { sensor: sensor() });
    context.covariance =
        DMatrix::from_row_slice(3, 3, &[9.0, 0.2, 0.3, 0.2, 4.0, 0.4, 0.3, 0.4, 1.0]);
    let detection = detect(&radar(None, 1.25), &context);
    Python::attach(|py| {
        let d = detection.bind(py);
        assert_common(py, d, &[0.1, 0.2, 100.0], &[0, 2, 4]);
        assert_eq!(
            model_covariance(py, d),
            DMatrix::from_row_slice(3, 3, &[1.0, 0.4, 0.3, 0.4, 4.0, 0.2, 0.3, 0.2, 9.0])
        );
        assert_eq!(
            d.getattr("timestamp")
                .unwrap()
                .call_method0("isoformat")
                .unwrap()
                .extract::<String>()
                .unwrap(),
            "2020-01-01T00:00:01.250000+00:00"
        );
    });
}

#[test]
fn radar4_preserves_pose_velocity_mapping_and_both_doppler_signs() {
    let rotation = Rotation3::from_euler_angles(0.3, -0.4, 0.7).into_inner();
    for (sign, rate) in [
        (DopplerSign::PositiveAway, -3.0),
        (DopplerSign::PositiveTowards, 3.0),
    ] {
        let mut context = context(ObservationProfile::Radar4 {
            sensor: SensorPose::Stationary(registration(rotation)),
            doppler_sign: sign,
        });
        context.covariance = DMatrix::from_diagonal(&DVector::from_vec(vec![9.0, 4.0, 1.0, 16.0]));
        context.covariance[(0, 3)] = 0.5;
        context.covariance[(3, 0)] = 0.5;
        let detection = detect(&radar(Some(-3.0), 0.0), &context);
        Python::attach(|py| {
            let d = detection.bind(py);
            assert_common(py, d, &[0.1, 0.2, 100.0, rate], &[0, 2, 4]);
            assert_radar_pose(py, d, &rotation);
            let expected_cross = if rate == -3.0 { 0.5 } else { -0.5 };
            let covariance = model_covariance(py, d);
            assert_eq!(covariance[(2, 3)], expected_cross);
            assert_eq!(covariance[(3, 2)], expected_cross);
            assert_eq!(
                covariance.diagonal(),
                DVector::from_vec(vec![1.0, 4.0, 9.0, 16.0])
            );
        });
    }
}

fn assert_radar_pose(py: Python<'_>, detection: &Bound<'_, PyAny>, rotation: &Matrix3<f64>) {
    let model = detection.getattr("measurement_model").unwrap();
    let actual = convert::numpy_to_dmatrix(py, &model.getattr("rotation_matrix").unwrap()).unwrap();
    let expected = DMatrix::from_column_slice(3, 3, rotation.transpose().as_slice());
    assert!((actual - expected).norm() < 1e-12);
    let position =
        convert::numpy_to_dvector(py, &model.getattr("translation_offset").unwrap()).unwrap();
    assert_eq!(position, DVector::from_vec(vec![10.0, 20.0, 30.0]));
    assert_eq!(
        model
            .getattr("velocity_mapping")
            .unwrap()
            .extract::<Vec<usize>>()
            .unwrap(),
        [1, 3, 5]
    );
    assert_eq!(
        convert::numpy_to_dvector(py, &model.getattr("velocity").unwrap()).unwrap(),
        DVector::zeros(3)
    );
}

fn model_prediction(py: Python<'_>, detection: &Bound<'_, PyAny>, state: &[f64]) -> DVector<f64> {
    let vector = convert::dvector_to_numpy(py, &DVector::from_column_slice(state)).unwrap();
    let state = py
        .import("stonesoup.types.state")
        .unwrap()
        .getattr("State")
        .unwrap()
        .call1((vector,))
        .unwrap();
    let prediction = detection
        .getattr("measurement_model")
        .unwrap()
        .call_method1("function", (state,))
        .unwrap();
    // Nonlinear models return angle objects; explicit test-side cast lets us
    // compare their physical values without weakening bridge dtype validation.
    let numeric = prediction.call_method1("astype", ("float64",)).unwrap();
    convert::numpy_to_dvector(py, &numeric).unwrap()
}

#[test]
fn eoir_model_applies_nonzero_translation_and_ninety_degree_rotation() {
    // Sensor +x maps to common +y; sensor +y maps to common -x.
    let rotation = Matrix3::new(0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0);
    let mut context = context(ObservationProfile::Eoir2 {
        sensor: SensorPose::Stationary(registration(rotation)),
    });
    context.covariance = DMatrix::from_row_slice(2, 2, &[4.0, 0.3, 0.3, 1.0]);
    let measurement = Measurement::EoIr {
        azimuth: 0.2,
        elevation: 0.1,
        time: 0.0,
        sensor_id: 7,
    };
    let detection = detect(&measurement, &context);
    let x = 100.0 * 0.1_f64.cos() * 0.2_f64.cos();
    let y = 100.0 * 0.1_f64.cos() * 0.2_f64.sin();
    let z = 100.0 * 0.1_f64.sin();
    Python::attach(|py| {
        let d = detection.bind(py);
        assert_common(py, d, &[0.1, 0.2], &[0, 2, 4]);
        let prediction = model_prediction(py, d, &[10.0 - y, 0.0, 20.0 + x, 0.0, 30.0 + z, 0.0]);
        assert!((prediction - DVector::from_vec(vec![0.1, 0.2])).norm() < 1e-12);
        assert_eq!(
            model_covariance(py, d),
            DMatrix::from_row_slice(2, 2, &[1.0, 0.3, 0.3, 4.0])
        );
    });
}

#[test]
fn spherical_pose_supports_pitch_gimbal_lock_without_nonfinite_offsets() {
    let rotation = Rotation3::from_euler_angles(0.0, std::f64::consts::FRAC_PI_2, 0.4).into_inner();
    let context = context(ObservationProfile::Radar4 {
        sensor: SensorPose::Stationary(registration(rotation)),
        doppler_sign: DopplerSign::PositiveAway,
    });
    let detection = detect(&radar(Some(-3.0), 0.0), &context);
    Python::attach(|py| assert_radar_pose(py, detection.bind(py), &rotation));
}

#[test]
fn adsb3_converts_msl_height_without_fabricating_velocity() {
    let context = context(ObservationProfile::Adsb3 { datum: datum() });
    let detection = detect(&adsb(0.0, None), &context);
    Python::attach(|py| {
        let d = detection.bind(py);
        assert_common(py, d, &[0.0, 0.0, 120.0], &[0, 2, 4]);
        let metadata = d.getattr("metadata").unwrap();
        assert_eq!(
            metadata
                .get_item("provenance")
                .unwrap()
                .extract::<String>()
                .unwrap(),
            "cooperative_report"
        );
        assert!(metadata.get_item("velocity").unwrap().is_none());
        assert_eq!(model_covariance(py, d), DMatrix::identity(3, 3));
    });
}

#[test]
fn adsb3_nonzero_origin_rotates_full_position_noise() {
    let mut context = context(ObservationProfile::Adsb3 { datum: datum() });
    context.frame = CommonFrame::Wgs84Enu(EnuOrigin {
        latitude_rad: 0.0,
        longitude_rad: std::f64::consts::FRAC_PI_2,
        ellipsoid_height_m: 0.0,
    });
    context.covariance =
        DMatrix::from_row_slice(3, 3, &[4.0, 0.2, 0.3, 0.2, 9.0, 0.4, 0.3, 0.4, 16.0]);
    let detection = detect(&adsb(0.0, None), &context);
    Python::attach(|py| {
        let d = detection.bind(py);
        assert_common(py, d, &[-6_378_257.0, 0.0, -6_378_137.0], &[0, 2, 4]);
        let expected =
            DMatrix::from_row_slice(3, 3, &[16.0, -0.4, -0.3, -0.4, 9.0, 0.2, -0.3, 0.2, 4.0]);
        assert!((model_covariance(py, d) - expected).norm() < 1e-12);
    });
}

fn correlated_six_noise() -> DMatrix<f64> {
    let mut covariance =
        DMatrix::from_diagonal(&DVector::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]));
    covariance[(0, 4)] = 0.25;
    covariance[(4, 0)] = 0.25;
    covariance[(2, 3)] = 0.5;
    covariance[(3, 2)] = 0.5;
    covariance
}

fn independent_nontrivial_rotations() -> (Matrix3<f64>, Matrix3<f64>) {
    // Closed-form ECEF->ENU bases at (45deg,-30deg) and (30deg,60deg).
    // These constants do not call the production geodetic/rotation helpers.
    let s2 = 2.0_f64.sqrt();
    let s3 = 3.0_f64.sqrt();
    let s6 = 6.0_f64.sqrt();
    (
        Matrix3::new(
            0.5,
            s3 / 2.0,
            0.0,
            -s6 / 4.0,
            s2 / 4.0,
            s2 / 2.0,
            s6 / 4.0,
            -s2 / 4.0,
            s2 / 2.0,
        ),
        Matrix3::new(
            -s3 / 2.0,
            0.5,
            0.0,
            -0.25,
            -s3 / 4.0,
            s3 / 2.0,
            s3 / 4.0,
            0.75,
            0.5,
        ),
    )
}

#[test]
fn adsb6_both_velocity_frames_preserve_full_covariance_at_nonzero_latitude() {
    let (common, report) = independent_nontrivial_rotations();
    let ecef_velocity = Vector3::new(4.0, -2.0, 1.0);
    let expected_velocity = common * ecef_velocity;
    let cases = [
        (VelocityFrame::Ecef, ecef_velocity, common),
        (
            VelocityFrame::EnuAtReport,
            report * ecef_velocity,
            common * report.transpose(),
        ),
    ];
    for (velocity_frame, velocity, velocity_rotation) in cases {
        let mut context = context(ObservationProfile::Adsb6 {
            datum: datum(),
            velocity_frame,
        });
        context.frame = CommonFrame::Wgs84Enu(EnuOrigin {
            latitude_rad: 45.0_f64.to_radians(),
            longitude_rad: -30.0_f64.to_radians(),
            ellipsoid_height_m: 0.0,
        });
        context.covariance = correlated_six_noise() + DMatrix::from_element(6, 6, 0.05);
        let measurement = Measurement::AdsB {
            lat: 30.0,
            lon: 60.0,
            alt: 100.0,
            velocity: Some([velocity.x, velocity.y, velocity.z]),
            time: 1.25,
        };
        let mut transform = DMatrix::zeros(6, 6);
        transform
            .view_mut((0, 0), (3, 3))
            .copy_from(&(common * report.transpose()));
        transform
            .view_mut((3, 3), (3, 3))
            .copy_from(&velocity_rotation);
        let expected_covariance = &transform * &context.covariance * transform.transpose();
        let detection = detect(&measurement, &context);
        Python::attach(|py| {
            let d = detection.bind(py);
            // WGS84 closed-form ECEF positions differenced and projected into
            // the independent basis above; not a production-helper round trip.
            assert_common(
                py,
                d,
                &[
                    5_528_360.562_341_29,
                    2_263_219.849_197_183_7,
                    -4_125_618.440_923_963_7,
                    expected_velocity.x,
                    expected_velocity.y,
                    expected_velocity.z,
                ],
                &[0, 2, 4, 1, 3, 5],
            );
            assert!((model_covariance(py, d) - &expected_covariance).norm() < 1e-12);
        });
    }
}

#[test]
fn adsb6_ecef_velocity_and_cross_covariance_use_common_enu() {
    let mut context = context(ObservationProfile::Adsb6 {
        datum: datum(),
        velocity_frame: VelocityFrame::Ecef,
    });
    context.covariance = correlated_six_noise();
    let detection = detect(&adsb(0.0, Some([1.0, 2.0, 3.0])), &context);
    Python::attach(|py| {
        let d = detection.bind(py);
        assert_common(
            py,
            d,
            &[0.0, 0.0, 120.0, 2.0, 3.0, 1.0],
            &[0, 2, 4, 1, 3, 5],
        );
        let cov = model_covariance(py, d);
        assert_eq!(
            cov.diagonal(),
            DVector::from_vec(vec![1.0, 2.0, 3.0, 5.0, 6.0, 4.0])
        );
        assert_eq!(cov[(0, 3)], 0.25);
        assert_eq!(cov[(3, 0)], 0.25);
        assert_eq!(cov[(2, 5)], 0.5);
        assert_eq!(cov[(5, 2)], 0.5);
        assert_eq!(cov.iter().filter(|x| **x != 0.0).count(), 10);
    });
}

#[test]
fn adsb6_report_local_velocity_rotates_both_covariance_blocks() {
    let mut context = context(ObservationProfile::Adsb6 {
        datum: datum(),
        velocity_frame: VelocityFrame::EnuAtReport,
    });
    context.covariance = correlated_six_noise();
    let detection = detect(&adsb(90.0, Some([1.0, 2.0, 3.0])), &context);
    Python::attach(|py| {
        let d = detection.bind(py);
        assert_common(
            py,
            d,
            &[6_378_257.0, 0.0, -6_378_137.0, 3.0, 2.0, -1.0],
            &[0, 2, 4, 1, 3, 5],
        );
        let cov = model_covariance(py, d);
        assert!(
            (cov.diagonal() - DVector::from_vec(vec![3.0, 2.0, 1.0, 6.0, 5.0, 4.0])).norm() < 1e-12
        );
        assert!((cov[(2, 4)] + 0.25).abs() < 1e-12);
        assert!((cov[(0, 5)] + 0.5).abs() < 1e-12);
        assert!((cov.clone() - cov.transpose()).norm() < 1e-12);
    });
}

#[test]
fn observation_time_is_not_arrival_time_or_call_order() {
    let mut context = context(ObservationProfile::Radar3 { sensor: sensor() });
    context.arrival = Some(context.origin + 50.0);
    for (offset, iso) in [
        (1.25, "2020-01-01T00:00:01.250000+00:00"),
        (-0.5, "2019-12-31T23:59:59.500000+00:00"),
        (0.0, "2020-01-01T00:00:00+00:00"),
    ] {
        let detection = detect(&radar(None, offset), &context);
        Python::attach(|py| {
            let d = detection.bind(py);
            assert_eq!(
                d.getattr("timestamp")
                    .unwrap()
                    .call_method0("isoformat")
                    .unwrap()
                    .extract::<String>()
                    .unwrap(),
                iso
            );
            let metadata = d.getattr("metadata").unwrap();
            assert_eq!(
                metadata.get_item("time").unwrap().extract::<f64>().unwrap(),
                offset
            );
            assert_eq!(
                metadata
                    .get_item("origin_utc")
                    .unwrap()
                    .extract::<String>()
                    .unwrap(),
                context.origin.to_string()
            );
            assert_eq!(
                metadata
                    .get_item("observation_utc_unrounded")
                    .unwrap()
                    .extract::<String>()
                    .unwrap(),
                (context.origin + offset).to_string()
            );
            assert_eq!(
                metadata
                    .get_item("arrival_utc")
                    .unwrap()
                    .extract::<String>()
                    .unwrap(),
                context.arrival.unwrap().to_string()
            );
        });
    }
}

#[test]
fn rounded_timestamp_collisions_keep_distinct_original_times() {
    let context = context(ObservationProfile::Radar3 { sensor: sensor() });
    let first = detect(&radar(None, 0.000_000_1), &context);
    let second = detect(&radar(None, 0.000_000_4), &context);
    Python::attach(|py| {
        assert!(
            first
                .bind(py)
                .getattr("timestamp")
                .unwrap()
                .eq(second.bind(py).getattr("timestamp").unwrap())
                .unwrap()
        );
        let a: f64 = first
            .bind(py)
            .getattr("metadata")
            .unwrap()
            .get_item("time")
            .unwrap()
            .extract()
            .unwrap();
        let b: f64 = second
            .bind(py)
            .getattr("metadata")
            .unwrap()
            .get_item("time")
            .unwrap()
            .extract()
            .unwrap();
        assert_ne!(a, b);
    });
}

#[test]
fn derived_overflow_is_rejected_before_python_access() {
    let mut context = context(ObservationProfile::Adsb3 {
        datum: AdsbDatum {
            geoid_separation_m: f64::MAX,
            noise: AdsbNoiseConvention::MetricEnuAtReport,
        },
    });
    let measurement = Measurement::AdsB {
        lat: 0.0,
        lon: 0.0,
        alt: f64::MAX,
        velocity: None,
        time: 0.0,
    };
    assert!(matches!(
        measurement_to_detection(&measurement, &context),
        Err(BridgeError::ConversionError(_))
    ));
    context.profile = ObservationProfile::Adsb6 {
        datum: datum(),
        velocity_frame: VelocityFrame::Ecef,
    };
    context.covariance = DMatrix::identity(6, 6);
    context.frame = CommonFrame::Wgs84Enu(EnuOrigin {
        latitude_rad: 0.0,
        longitude_rad: std::f64::consts::FRAC_PI_4,
        ellipsoid_height_m: 0.0,
    });
    let huge_velocity = adsb(0.0, Some([f64::MAX, -f64::MAX, 0.0]));
    assert!(matches!(
        measurement_to_detection(&huge_velocity, &context),
        Err(BridgeError::ConversionError(_))
    ));
}

fn unsupported_measurements() -> Vec<Measurement> {
    vec![
        Measurement::Othr {
            ground_range_m: 10.0,
            azimuth_rad: 0.0,
            doppler_m_s: 0.0,
            propagation_mode: PropagationMode::FLayer,
            time: 0.0,
            sensor_id: 7,
        },
        Measurement::Lidar {
            position: [0.0; 3],
            extent: [1.0; 3],
            yaw: 0.0,
            class: TargetClass::Unknown,
            velocity: None,
            time: 0.0,
            sensor_id: 7,
        },
        Measurement::Camera {
            center_px: [0.0; 2],
            extent_px: [1.0; 2],
            position: None,
            yaw: None,
            class: TargetClass::Unknown,
            time: 0.0,
            sensor_id: 7,
        },
    ]
}

fn run_child(case: &str) {
    let result = std::process::Command::new(std::env::current_exe().unwrap())
        .args(["--exact", "detection_process_child", "--nocapture"])
        .env("THRESH_DETECTION_CONTRACT_CASE", case)
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "child {case}: {}\n{}",
        String::from_utf8_lossy(&result.stdout),
        String::from_utf8_lossy(&result.stderr)
    );
    assert!(String::from_utf8_lossy(&result.stdout).contains("1 passed; 0 failed; 0 ignored"));
}

#[test]
fn validation_precedes_runtime_and_missing_packages() {
    run_child("uninitialized");
    run_child("missing_packages");
}

#[test]
fn finite_position_and_covariance_overflow_precede_interpreter_access() {
    run_child("derived_overflow");
}

fn assert_derived_overflow_without_python() {
    assert!(Python::try_attach(|_| ()).is_none());
    let mut context = context(ObservationProfile::Adsb3 { datum: datum() });
    context.frame = CommonFrame::Wgs84Enu(EnuOrigin {
        latitude_rad: 0.0,
        longitude_rad: 0.0,
        ellipsoid_height_m: -f64::MAX,
    });
    let measurement = Measurement::AdsB {
        lat: 0.0,
        lon: 0.0,
        alt: f64::MAX,
        velocity: None,
        time: 0.0,
    };
    assert!(context.validate(&measurement).is_ok());
    assert!(matches!(
        measurement_to_detection(&measurement, &context),
        Err(BridgeError::ConversionError(_))
    ));
    context.frame = CommonFrame::Wgs84Enu(EnuOrigin {
        latitude_rad: 0.0,
        longitude_rad: std::f64::consts::FRAC_PI_4,
        ellipsoid_height_m: 0.0,
    });
    let m = f64::MAX * 0.75;
    context.covariance = DMatrix::from_row_slice(3, 3, &[m, 0.0, m, 0.0, m, 0.0, m, 0.0, m]);
    let ordinary = adsb(0.0, None);
    assert!(context.validate(&ordinary).is_ok());
    assert!(matches!(
        measurement_to_detection(&ordinary, &context),
        Err(BridgeError::ConversionError(_))
    ));
    assert!(Python::try_attach(|_| ()).is_none());
}

#[test]
fn detection_process_child() {
    let Ok(case) = std::env::var("THRESH_DETECTION_CONTRACT_CASE") else {
        return;
    };
    if case == "derived_overflow" {
        assert_derived_overflow_without_python();
        return;
    }
    if case == "missing_packages" {
        Python::initialize();
        Python::attach(|py| {
            py.run(pyo3::ffi::c_str!("import sys\nclass BlockAll:\n def find_spec(self, fullname, path=None, target=None):\n  raise AssertionError('unexpected import: ' + fullname)\nsys.meta_path.insert(0, BlockAll())"), None, None).unwrap()
        });
    }
    let context = context(ObservationProfile::Radar3 { sensor: sensor() });
    for measurement in unsupported_measurements() {
        let error = measurement_to_detection(&measurement, &context).unwrap_err();
        assert!(matches!(error, BridgeError::ConversionError(_)));
        assert!(error.to_string().contains("unsupported measurement"));
    }
    let mut invalid = context.clone();
    invalid.frame = CommonFrame::Unspecified;
    assert!(matches!(
        measurement_to_detection(&radar(None, 0.0), &invalid),
        Err(BridgeError::ConversionError(_))
    ));
    if case == "uninitialized" {
        assert!(matches!(
            measurement_to_detection(&radar(None, 0.0), &context),
            Err(BridgeError::PythonNotAvailable(_))
        ));
    }
}
