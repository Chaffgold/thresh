//! Fresh-process lifecycle and structured-diagnostic contracts; not algorithm parity.

#![cfg(feature = "stonesoup")]

use std::process::Command;

use nalgebra::{DMatrix, Matrix3, Vector3};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use thresh_bridge::context::{
    CommonFrame, DetectionContext, EnuOrigin, ObservationProfile, SensorPose,
};
use thresh_bridge::detection::measurement_to_detection;
use thresh_bridge::error::{BridgeError, PythonException};
use thresh_bridge::jpda::{JpdaAssociator, JpdaConfig};
use thresh_bridge::phd::{PhdConfig, PhdFilter};
use thresh_core::{measurement::Measurement, sensor::SensorRegistration, time::Epoch};

fn fresh_process(name: &str, assertion: impl FnOnce()) {
    if std::env::var("THRESH_BRIDGE_RUNTIME_CASE").as_deref() == Ok(name) {
        assertion();
        return;
    }
    let output = Command::new(std::env::current_exe().unwrap())
        .args(["--exact", name, "--nocapture"])
        .env("THRESH_BRIDGE_RUNTIME_CASE", name)
        .output()
        .expect("launch isolated runtime assertion");
    assert!(
        output.status.success(),
        "child {name} failed:\n{}\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(String::from_utf8_lossy(&output.stdout).contains("1 passed"));
}

fn block_import(py: Python<'_>, target: &str, missing: &str) {
    let globals = PyDict::new(py);
    globals.set_item("target_module", target).unwrap();
    globals.set_item("missing_module", missing).unwrap();
    py.run(c"\
import sys
class BlockImport:
    def find_spec(self, fullname, path=None, target=None):
        if fullname == target_module:
            raise ModuleNotFoundError('synthetic import failure; unrelated display text', name=missing_module)
        return None
sys.meta_path.insert(0, BlockImport())
", Some(&globals), None).unwrap();
}

fn exception(error: &BridgeError) -> &PythonException {
    match error {
        BridgeError::StoneSoupNotInstalled(source)
        | BridgeError::MissingDependency { source, .. }
        | BridgeError::UnsupportedApi { source, .. }
        | BridgeError::PythonError(source) => source,
        _ => panic!("expected structured Python cause, got {error:?}"),
    }
}

fn assert_import_details(error: &BridgeError) {
    let source = exception(error);
    assert_eq!(source.exception_type, "ModuleNotFoundError");
    assert!(source.message.contains("synthetic import failure"));
    assert!(source.operation.starts_with("import "));
    assert!(
        source
            .traceback
            .as_ref()
            .is_some_and(|traceback| traceback.contains("find_spec"))
    );
}

#[test]
fn explicit_host_initialization_is_reused_without_mutation() {
    fresh_process(
        "explicit_host_initialization_is_reused_without_mutation",
        || {
            Python::initialize();
            Python::initialize();
            let before = Python::attach(|py| {
                block_import(py, "stonesoup", "stonesoup");
                environment_snapshot(py)
            });
            for _ in 0..3 {
                assert!(matches!(
                    jpda_dependency_error(),
                    BridgeError::StoneSoupNotInstalled(_)
                ));
            }
            let after = Python::try_attach(environment_snapshot).expect("host remains initialized");
            assert_eq!(before, after);
        },
    );
}

fn jpda_dependency_error() -> BridgeError {
    Python::attach(|py| {
        let placeholder = py.None();
        JpdaAssociator::new(&JpdaConfig::default(), &placeholder, &placeholder)
            .err()
            .expect("blocked dependency must fail before component use")
    })
}

/// Invented radar observation and context that pass every Rust-side check, so
/// conversion reaches its Python imports.
fn valid_detection_inputs() -> (Measurement, DetectionContext) {
    let sensor =
        SensorRegistration::with_rotation(Vector3::new(10.0, 20.0, 30.0), Matrix3::identity(), 7);
    let measurement = Measurement::Radar {
        range: 100.0,
        azimuth: 0.2,
        elevation: 0.1,
        range_rate: None,
        time: 0.0,
        sensor_id: 7,
    };
    let context = DetectionContext {
        origin: Epoch::from_iso8601("2020-01-01T00:00:00 UTC").unwrap(),
        arrival: None,
        frame: CommonFrame::Wgs84Enu(EnuOrigin {
            latitude_rad: 0.0,
            longitude_rad: 0.0,
            ellipsoid_height_m: 0.0,
        }),
        profile: ObservationProfile::Radar3 {
            sensor: SensorPose::Stationary(sensor),
        },
        covariance: DMatrix::identity(3, 3),
    };
    (measurement, context)
}

fn reference_context(py: Python<'_>) -> Bound<'_, PyDict> {
    let globals = PyDict::new(py);
    py.run(
        c"\
from datetime import datetime, timezone
from stonesoup.models.transition.linear import RandomWalk
from stonesoup.models.measurement.linear import LinearGaussian
from stonesoup.predictor.kalman import KalmanPredictor
from stonesoup.updater.kalman import KalmanUpdater
predictor = KalmanPredictor(RandomWalk(0.0))
updater = KalmanUpdater(LinearGaussian(ndim_state=1, mapping=[0], noise_covar=[[1.0]]))
timestamp = datetime(2020, 1, 1, tzinfo=timezone.utc)
tracks, detections = set(), set()
prior, births = [], []
",
        Some(&globals),
        None,
    )
    .unwrap();
    globals
}

#[test]
fn initialized_reference_calls_reuse_host_without_mutation() {
    fresh_process(
        "initialized_reference_calls_reuse_host_without_mutation",
        || {
            Python::initialize();
            Python::attach(|py| {
                let context = reference_context(py);
                let before = environment_snapshot(py);
                for _ in 0..3 {
                    assert_empty_reference_calls(py, &context);
                }
                assert_eq!(before, environment_snapshot(py));
            });
        },
    );
}

fn assert_empty_reference_calls(py: Python<'_>, context: &Bound<'_, PyDict>) {
    let item = |name| context.get_item(name).unwrap().unwrap().unbind();
    let predictor = item("predictor");
    let updater = item("updater");
    let timestamp = item("timestamp");
    let detections = item("detections");
    let jpda = JpdaAssociator::new(&JpdaConfig::default(), &predictor, &updater).unwrap();
    let associations = jpda
        .associate(&item("tracks"), &detections, &timestamp)
        .unwrap();
    assert_eq!(associations.bind(py).len().unwrap(), 0);
    let phd = PhdFilter::new(
        py,
        PhdConfig::default(),
        predictor.bind(py),
        updater.bind(py),
    )
    .unwrap();
    let posterior = phd
        .filter(&item("prior"), &item("births"), &detections, &timestamp)
        .unwrap();
    assert_eq!(
        posterior
            .bind(py)
            .getattr("components")
            .unwrap()
            .len()
            .unwrap(),
        0
    );
    assert!(posterior.bind(py).getattr("timestamp").unwrap().is_none());
}

fn environment_snapshot(py: Python<'_>) -> (Vec<String>, Vec<(String, String)>, usize) {
    let sys = py.import("sys").unwrap();
    let path = sys.getattr("path").unwrap().extract().unwrap();
    let environment = py
        .import("os")
        .unwrap()
        .getattr("environ")
        .unwrap()
        .call_method0("items")
        .unwrap();
    let values: Vec<(String, String)> = py
        .import("builtins")
        .unwrap()
        .getattr("sorted")
        .unwrap()
        .call1((environment,))
        .unwrap()
        .extract()
        .unwrap();
    (path, values, sys.as_ptr() as usize)
}

#[test]
fn missing_top_level_stonesoup_is_distinct() {
    fresh_process("missing_top_level_stonesoup_is_distinct", || {
        Python::initialize();
        Python::attach(|py| block_import(py, "stonesoup", "stonesoup"));
        let error = jpda_dependency_error();
        assert!(matches!(&error, BridgeError::StoneSoupNotInstalled(_)));
        assert_import_details(&error);
    });
}

#[test]
fn missing_numpy_is_a_dependency_not_an_interpreter_error() {
    fresh_process(
        "missing_numpy_is_a_dependency_not_an_interpreter_error",
        || {
            Python::initialize();
            // Importing Stone Soup does not load NumPy, so with a valid context
            // the converter's own NumPy import is the first blocked import.
            Python::attach(|py| block_import(py, "numpy", "numpy"));
            let (measurement, context) = valid_detection_inputs();
            let error = measurement_to_detection(&measurement, &context)
                .err()
                .unwrap();
            assert!(
                matches!(&error, BridgeError::MissingDependency { name, .. } if name == "numpy")
            );
            assert_import_details(&error);
            assert_eq!(exception(&error).operation, "import numpy");
        },
    );
}

#[test]
fn missing_transitive_dependency_uses_exception_identity() {
    fresh_process(
        "missing_transitive_dependency_uses_exception_identity",
        || {
            Python::initialize();
            Python::attach(|py| block_import(py, "stonesoup", "synthetic_dependency"));
            let error = jpda_dependency_error();
            assert!(
                matches!(&error, BridgeError::MissingDependency { name, .. } if name == "synthetic_dependency")
            );
            assert_import_details(&error);
        },
    );
}

#[test]
fn missing_stonesoup_submodule_is_an_api_error() {
    fresh_process("missing_stonesoup_submodule_is_an_api_error", || {
        Python::initialize();
        Python::attach(|py| {
            // The JPDA constructor's first import; placeholder components are
            // never used because the import fails first.
            block_import(
                py,
                "stonesoup.hypothesiser.probability",
                "stonesoup.hypothesiser.probability",
            );
        });
        let error = jpda_dependency_error();
        assert!(
            matches!(&error, BridgeError::UnsupportedApi { api, .. } if api == "stonesoup.hypothesiser.probability")
        );
        assert_import_details(&error);
    });
}

#[test]
fn missing_symbol_is_an_api_error() {
    fresh_process("missing_symbol_is_an_api_error", || {
        Python::initialize();
        let error = Python::attach(|py| {
            py.run(
                c"\
import sys, types, importlib.util
class EmptyModule:
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'stonesoup.dataassociator.probability':
            return importlib.util.spec_from_loader(fullname, self)
        return None
    def create_module(self, spec):
        return types.ModuleType(spec.name)
    def exec_module(self, module):
        pass
sys.meta_path.insert(0, EmptyModule())
",
                None,
                None,
            )
            .unwrap();
            // Real components, so the hypothesiser is built and only the
            // associator symbol lookup can fail.
            let context = reference_context(py);
            let item = |name| context.get_item(name).unwrap().unwrap().unbind();
            JpdaAssociator::new(&JpdaConfig::default(), &item("predictor"), &item("updater"))
                .err()
                .expect("empty associator module must fail the JPDA lookup")
        });
        assert!(matches!(&error, BridgeError::UnsupportedApi { api, .. } if api == "JPDA"));
        let source = exception(&error);
        assert_eq!(source.exception_type, "AttributeError");
        assert!(source.message.contains("JPDA"));
        assert_eq!(source.operation, "JPDA associator lookup");
    });
}

#[test]
fn ordinary_python_failures_keep_type_message_context_and_traceback() {
    fresh_process(
        "ordinary_python_failures_keep_type_message_context_and_traceback",
        || {
            Python::initialize();
            Python::attach(|py| {
                assert_ordinary_error(
                    py,
                    c"raise TypeError('ModuleNotFoundError: stonesoup numpy')",
                    "TypeError",
                );
                assert_ordinary_error(
                    py,
                    c"raise ValueError('stonesoup is mentioned but not missing')",
                    "ValueError",
                );
                assert_ordinary_error(
                    py,
                    c"raise AttributeError('execution failure, not lookup')",
                    "AttributeError",
                );
                assert_ordinary_error(
                    py,
                    c"raise ModuleNotFoundError('stonesoup; no structured name')",
                    "ModuleNotFoundError",
                );
            });
        },
    );
}

fn assert_ordinary_error(py: Python<'_>, source: &std::ffi::CStr, kind: &str) {
    let original = py.run(source, None, None).unwrap_err();
    let original_message = original.value(py).str().unwrap().to_string();
    let error = BridgeError::from_python(py, original, "synthetic execution");
    assert!(matches!(&error, BridgeError::PythonError(_)));
    let cause = exception(&error);
    assert_eq!(cause.exception_type, kind);
    assert_eq!(cause.message, original_message);
    assert_eq!(cause.operation, "synthetic execution");
    assert!(
        cause
            .traceback
            .as_ref()
            .is_some_and(|traceback| traceback.contains("<module>"))
    );
    assert!(std::error::Error::source(&error).is_some());
}
