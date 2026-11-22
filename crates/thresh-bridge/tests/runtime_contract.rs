//! Fresh-process lifecycle and structured-diagnostic contracts; not algorithm parity.

#![cfg(feature = "stonesoup")]

use std::process::Command;

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use thresh_bridge::error::{BridgeError, PythonException};
use thresh_bridge::imm::{ImmConfig, ImmFilter};
use thresh_bridge::jpda::{JpdaAssociator, JpdaConfig};
use thresh_bridge::mht::{MhtConfig, MhtTracker};
use thresh_bridge::phd::{PhdConfig, PhdFilter};

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
fn token_free_constructors_require_host_initialization() {
    fresh_process(
        "token_free_constructors_require_host_initialization",
        || {
            assert!(matches!(
                JpdaAssociator::new(&JpdaConfig::default()),
                Err(BridgeError::PythonNotAvailable(_))
            ));
            assert!(matches!(
                MhtTracker::new(&MhtConfig::default()),
                Err(BridgeError::PythonNotAvailable(_))
            ));
            assert!(Python::try_attach(|_| ()).is_none());
        },
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
                    JpdaAssociator::new(&JpdaConfig::default()),
                    Err(BridgeError::StoneSoupNotInstalled(_))
                ));
                assert!(matches!(
                    MhtTracker::new(&MhtConfig::default()),
                    Err(BridgeError::StoneSoupNotInstalled(_))
                ));
            }
            let after = Python::try_attach(environment_snapshot).expect("host remains initialized");
            assert_eq!(before, after);
        },
    );
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
        let error = JpdaAssociator::new(&JpdaConfig::default()).err().unwrap();
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
            Python::attach(|py| {
                block_import(py, "numpy", "numpy");
                let models = PyList::empty(py);
                let error = ImmFilter::new(py, &ImmConfig::default(), &models)
                    .err()
                    .unwrap();
                assert!(
                    matches!(&error, BridgeError::MissingDependency { name, .. } if name == "numpy")
                );
                assert_import_details(&error);
            });
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
            let error = JpdaAssociator::new(&JpdaConfig::default()).err().unwrap();
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
            block_import(
                py,
                "stonesoup.predictor.interacting",
                "stonesoup.predictor.interacting",
            );
            let models = PyList::empty(py);
            let error = ImmFilter::new(py, &ImmConfig::default(), &models)
                .err()
                .unwrap();
            assert!(
                matches!(&error, BridgeError::UnsupportedApi { api, .. } if api == "stonesoup.predictor.interacting")
            );
            assert_import_details(&error);
        });
    });
}

#[test]
fn missing_symbol_is_an_api_error() {
    fresh_process("missing_symbol_is_an_api_error", || {
        Python::initialize();
        Python::attach(|py| {
            py.run(
                c"\
import sys, types, importlib.util
class EmptyModule:
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'stonesoup.tracker.simple':
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
        });
        let error = MhtTracker::new(&MhtConfig::default()).err().unwrap();
        assert!(
            matches!(&error, BridgeError::UnsupportedApi { api, .. } if api == "MultiTargetMixtureTracker")
        );
        let source = exception(&error);
        assert_eq!(source.exception_type, "AttributeError");
        assert!(source.message.contains("MultiTargetMixtureTracker"));
        assert_eq!(source.operation, "MHT tracker lookup");
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

#[test]
fn initialized_token_constructors_report_existing_algorithm_failures() {
    fresh_process(
        "initialized_token_constructors_report_existing_algorithm_failures",
        || {
            Python::initialize();
            Python::attach(|py| {
                let models = PyList::empty(py);
                let imm_error = ImmFilter::new(py, &ImmConfig::default(), &models)
                    .err()
                    .unwrap();
                assert!(
                    matches!(&imm_error, BridgeError::UnsupportedApi { api, .. } if api == "stonesoup.predictor.interacting")
                );
                let placeholder = py.None();
                let phd_error = PhdFilter::new(
                    py,
                    PhdConfig::default(),
                    placeholder.bind(py),
                    placeholder.bind(py),
                )
                .err()
                .unwrap();
                assert!(matches!(&phd_error, BridgeError::PythonError(_)));
                assert_eq!(exception(&phd_error).exception_type, "TypeError");
            });
        },
    );
}
