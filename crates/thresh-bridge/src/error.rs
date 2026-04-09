//! Error types for the Stone Soup bridge.

use std::fmt;

/// Errors that can occur when interacting with the Python / Stone Soup bridge.
#[derive(Debug)]
pub enum BridgeError {
    /// Python interpreter is not available or could not be initialised.
    PythonNotAvailable(String),
    /// The `stonesoup` Python package is not installed.
    StoneSoupNotInstalled(String),
    /// A Python exception was raised during a bridge call.
    PythonError(String),
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

impl std::error::Error for BridgeError {}

#[cfg(feature = "stonesoup")]
impl From<pyo3::PyErr> for BridgeError {
    fn from(err: pyo3::PyErr) -> Self {
        let msg = pyo3::Python::with_gil(|_py| err.to_string());
        // Check for common import errors to provide better diagnostics.
        if msg.contains("ModuleNotFoundError") && msg.contains("stonesoup") {
            BridgeError::StoneSoupNotInstalled(msg)
        } else if msg.contains("ModuleNotFoundError") && msg.contains("numpy") {
            BridgeError::PythonNotAvailable(format!("numpy not installed: {msg}"))
        } else {
            BridgeError::PythonError(msg)
        }
    }
}

/// Convenience alias used throughout the bridge crate.
pub type BridgeResult<T> = Result<T, BridgeError>;
