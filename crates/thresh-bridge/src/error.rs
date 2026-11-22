//! Error types for the Stone Soup bridge.

use std::fmt;

use pyo3::exceptions::{PyAttributeError, PyModuleNotFoundError};
use pyo3::prelude::*;
use pyo3::types::{PyAnyMethods, PyTracebackMethods, PyTypeMethods};

/// Owned details captured while the caller already holds a Python token.
///
/// Formatting this value does not attach to or otherwise access Python.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PythonException {
    /// Bridge operation which raised the exception.
    pub operation: String,
    /// Original Python exception class name, without interpreting its message.
    pub exception_type: String,
    /// Original exception message, when its string representation is available.
    pub message: String,
    /// Original formatted traceback, when the exception has one.
    pub traceback: Option<String>,
}

impl PythonException {
    fn capture(py: Python<'_>, error: &PyErr, operation: &str) -> Self {
        Self {
            operation: operation.to_owned(),
            exception_type: error
                .get_type(py)
                .name()
                .map(|name| name.to_string())
                .unwrap_or_else(|_| "<unavailable exception type>".to_owned()),
            message: error
                .value(py)
                .str()
                .map(|message| message.to_string())
                .unwrap_or_else(|_| "<unavailable exception message>".to_owned()),
            traceback: error
                .traceback(py)
                .and_then(|traceback| traceback.format().ok()),
        }
    }
}

impl fmt::Display for PythonException {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}: {}: {}",
            self.operation, self.exception_type, self.message
        )
    }
}

impl std::error::Error for PythonException {}

/// Errors that can occur when interacting with the Python / Stone Soup bridge.
#[derive(Debug)]
pub enum BridgeError {
    /// Host-owned Python interpreter is not available for attachment.
    PythonNotAvailable(String),
    /// The `stonesoup` Python package is not installed.
    StoneSoupNotInstalled(PythonException),
    /// A dependency other than the top-level `stonesoup` package is missing.
    MissingDependency {
        /// Missing module identity from `ModuleNotFoundError.name`.
        name: String,
        /// Original Python exception details.
        source: PythonException,
    },
    /// A requested submodule or symbol is unavailable in an installed package.
    UnsupportedApi {
        /// Requested module or qualified symbol.
        api: String,
        /// Original Python exception details.
        source: PythonException,
    },
    /// A Python exception was raised during a bridge call.
    PythonError(PythonException),
    /// A numpy conversion failed (shape mismatch, dtype, etc.).
    ConversionError(String),
    /// An algorithm-specific error (bad config, divergence, etc.).
    AlgorithmError(String),
}

impl fmt::Display for BridgeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BridgeError::PythonNotAvailable(msg) => {
                write!(f, "Python not available: {msg}")
            }
            BridgeError::StoneSoupNotInstalled(msg) => {
                write!(f, "Stone Soup not installed: {msg}")
            }
            BridgeError::MissingDependency { name, source } => {
                write!(f, "Missing Python dependency {name}: {source}")
            }
            BridgeError::UnsupportedApi { api, source } => {
                write!(f, "Unsupported Python API {api}: {source}")
            }
            BridgeError::PythonError(msg) => {
                write!(f, "Python error: {msg}")
            }
            BridgeError::ConversionError(msg) => {
                write!(f, "Conversion error: {msg}")
            }
            BridgeError::AlgorithmError(msg) => {
                write!(f, "Algorithm error: {msg}")
            }
        }
    }
}

impl std::error::Error for BridgeError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::StoneSoupNotInstalled(source)
            | Self::MissingDependency { source, .. }
            | Self::UnsupportedApi { source, .. }
            | Self::PythonError(source) => Some(source),
            _ => None,
        }
    }
}

impl BridgeError {
    /// Convert an exception using an existing token; never attach implicitly.
    ///
    /// Only `ModuleNotFoundError.name`, not display text, determines missing
    /// dependency categories. Other exception types retain their original kind.
    pub fn from_python(py: Python<'_>, error: PyErr, operation: &str) -> Self {
        let source = PythonException::capture(py, &error, operation);
        match missing_module(py, &error) {
            Some(name) if name == "stonesoup" => Self::StoneSoupNotInstalled(source),
            Some(name) if name.starts_with("stonesoup.") => {
                Self::UnsupportedApi { api: name, source }
            }
            Some(name) => Self::MissingDependency { name, source },
            None => Self::PythonError(source),
        }
    }

    /// Convert failure to retrieve a specifically requested upstream API.
    ///
    /// An `AttributeError` is an API diagnostic only at this lookup boundary;
    /// ordinary Python execution still uses [`Self::from_python`].
    pub fn from_python_api(py: Python<'_>, error: PyErr, operation: &str, api: &str) -> Self {
        if error.is_instance_of::<PyAttributeError>(py) {
            return Self::UnsupportedApi {
                api: api.to_owned(),
                source: PythonException::capture(py, &error, operation),
            };
        }
        Self::from_python(py, error, operation)
    }
}

fn missing_module(py: Python<'_>, error: &PyErr) -> Option<String> {
    if !error.is_instance_of::<PyModuleNotFoundError>(py) {
        return None;
    }
    error.value(py).getattr("name").ok()?.extract().ok()
}

/// Convenience alias used throughout the bridge crate.
pub type BridgeResult<T> = Result<T, BridgeError>;
