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
