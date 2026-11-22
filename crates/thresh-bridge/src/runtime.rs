//! Fallible access to an interpreter initialized and owned by the host.

#![doc = include_str!("../../../docs/bridge-runtime.md")]

use pyo3::prelude::*;
use pyo3::types::PyModule;

use crate::error::{BridgeError, BridgeResult};

/// Attach without initializing, finalizing, or altering the host environment.
pub(crate) fn with_python<T>(
    operation: &str,
    callback: impl for<'py> FnOnce(Python<'py>) -> BridgeResult<T>,
) -> BridgeResult<T> {
    Python::try_attach(callback).unwrap_or_else(|| {
        Err(BridgeError::PythonNotAvailable(format!(
            "{operation}: host must initialize Python before using the bridge; attachment is unavailable"
        )))
    })
}

/// Import with structured missing-module diagnostics using an existing token.
pub(crate) fn import_module<'py>(
    py: Python<'py>,
    name: &str,
) -> BridgeResult<Bound<'py, PyModule>> {
    py.import(name)
        .map_err(|error| BridgeError::from_python(py, error, &format!("import {name}")))
}

/// Retrieve an expected API symbol without treating arbitrary execution errors
/// as package-version mismatches.
pub(crate) fn required_attribute<'py>(
    object: &Bound<'py, PyAny>,
    name: &str,
    operation: &str,
) -> BridgeResult<Bound<'py, PyAny>> {
    object
        .getattr(name)
        .map_err(|error| BridgeError::from_python_api(object.py(), error, operation, name))
}

/// Test-only dispatch doubles prove boundary plumbing, not algorithm correctness.
#[cfg(test)]
pub(crate) mod step_test_support {
    use super::*;
    use pyo3::types::PyDict;

    type HostSnapshot = (Vec<String>, Vec<(String, String)>, usize);

    struct DispatchFixture {
        object: Py<PyAny>,
        arguments: Vec<Py<PyAny>>,
        result: Py<PyAny>,
        host: HostSnapshot,
    }

    /// Run a real wrapper step against a Python-owned, synthetic dispatch object.
    pub(crate) fn assert_step_contract<W>(
        method: &str,
        operation: &str,
        lookup_operation: &str,
        arity: usize,
        wrap: impl FnOnce(Py<PyAny>) -> W,
        invoke: impl Fn(&W, &[Py<PyAny>]) -> BridgeResult<Py<PyAny>>,
    ) {
        Python::initialize();
        Python::initialize();
        let fixture = Python::attach(|py| create_fixture(py, method, arity));
        let wrapper = wrap(Python::attach(|py| fixture.object.clone_ref(py)));
        assert_successful_dispatch(&wrapper, &invoke, &fixture);
        assert_python_failures(&wrapper, &invoke, &fixture, operation);
        assert_missing_method(&wrapper, &invoke, &fixture, method, lookup_operation);
        let host = Python::try_attach(host_snapshot).expect("host remains initialized");
        assert_eq!(fixture.host, host);
    }

    fn create_fixture(py: Python<'_>, method: &str, arity: usize) -> DispatchFixture {
        let globals = PyDict::new(py);
        py.run(
            c"\
class DispatchDouble:
    def __init__(self):
        self.calls = []
        self.result = object()
        self.failure = None
    def dispatch(self, *arguments):
        self.calls.append(arguments)
        if self.failure is not None:
            raise self.failure
        return self.result
",
            Some(&globals),
            None,
        )
        .unwrap();
        let object = globals
            .get_item("DispatchDouble")
            .unwrap()
            .unwrap()
            .call0()
            .unwrap();
        object
            .setattr(method, object.getattr("dispatch").unwrap())
            .unwrap();
        DispatchFixture {
            result: object.getattr("result").unwrap().unbind(),
            object: object.unbind(),
            arguments: (0..arity)
                .map(|_| PyDict::new(py).into_any().unbind())
                .collect(),
            host: host_snapshot(py),
        }
    }

    fn assert_successful_dispatch<W>(
        wrapper: &W,
        invoke: &impl Fn(&W, &[Py<PyAny>]) -> BridgeResult<Py<PyAny>>,
        fixture: &DispatchFixture,
    ) {
        for _ in 0..2 {
            // The call must attach itself; no Python token is held here.
            let returned = invoke(wrapper, &fixture.arguments).unwrap();
            Python::try_attach(|py| assert!(returned.bind(py).is(fixture.result.bind(py))))
                .expect("step preserves the initialized host");
        }
        Python::attach(|py| {
            let calls = fixture.object.bind(py).getattr("calls").unwrap();
            assert_eq!(calls.len().unwrap(), 2);
            for index in 0..2 {
                let call = calls.get_item(index).unwrap();
                assert_eq!(call.len().unwrap(), fixture.arguments.len());
                for (position, argument) in fixture.arguments.iter().enumerate() {
                    assert!(call.get_item(position).unwrap().is(argument.bind(py)));
                }
            }
        });
    }

    fn assert_python_failures<W>(
        wrapper: &W,
        invoke: &impl Fn(&W, &[Py<PyAny>]) -> BridgeResult<Py<PyAny>>,
        fixture: &DispatchFixture,
        operation: &str,
    ) {
        const MESSAGE: &str = "synthetic step failure mentioning stonesoup numpy";
        for kind in ["ValueError", "TypeError"] {
            Python::attach(|py| {
                let error = py
                    .import("builtins")
                    .unwrap()
                    .getattr(kind)
                    .unwrap()
                    .call1((MESSAGE,))
                    .unwrap();
                fixture.object.bind(py).setattr("failure", error).unwrap();
            });
            let error = invoke(wrapper, &fixture.arguments).unwrap_err();
            let BridgeError::PythonError(source) = error else {
                panic!("expected ordinary Python execution error, got {error:?}");
            };
            assert_eq!(source.exception_type, kind);
            assert_eq!(source.message, MESSAGE);
            assert_eq!(source.operation, operation);
            assert!(
                source
                    .traceback
                    .as_ref()
                    .is_some_and(|text| text.contains("dispatch"))
            );
        }
    }

    fn assert_missing_method<W>(
        wrapper: &W,
        invoke: &impl Fn(&W, &[Py<PyAny>]) -> BridgeResult<Py<PyAny>>,
        fixture: &DispatchFixture,
        method: &str,
        lookup_operation: &str,
    ) {
        Python::attach(|py| fixture.object.bind(py).delattr(method).unwrap());
        let error = invoke(wrapper, &fixture.arguments).unwrap_err();
        let BridgeError::UnsupportedApi { api, source } = error else {
            panic!("expected missing-method API error, got {error:?}");
        };
        assert_eq!(api, method);
        assert_eq!(source.exception_type, "AttributeError");
        assert_eq!(source.operation, lookup_operation);
        assert!(source.message.contains(method));
    }

    fn host_snapshot(py: Python<'_>) -> HostSnapshot {
        let sys = py.import("sys").unwrap();
        let path = sys.getattr("path").unwrap().extract().unwrap();
        let environment = py
            .import("os")
            .unwrap()
            .getattr("environ")
            .unwrap()
            .call_method0("items")
            .unwrap();
        let values = py
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
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn uninitialized_shared_attachment_is_fallible() {
        const CASE: &str = "runtime::tests::uninitialized_shared_attachment_is_fallible";
        if std::env::var("THRESH_RUNTIME_UNIT_CHILD").as_deref() != Ok(CASE) {
            let output = std::process::Command::new(std::env::current_exe().unwrap())
                .args(["--exact", CASE, "--nocapture"])
                .env("THRESH_RUNTIME_UNIT_CHILD", CASE)
                .output()
                .unwrap();
            assert!(
                output.status.success(),
                "{}\n{}",
                String::from_utf8_lossy(&output.stdout),
                String::from_utf8_lossy(&output.stderr)
            );
            return;
        }
        let result = with_python("fresh-process probe", |_| -> BridgeResult<()> {
            panic!("callback must not run without host initialization")
        });
        assert!(matches!(result, Err(BridgeError::PythonNotAvailable(_))));
        assert!(Python::try_attach(|_| ()).is_none());
    }
}
