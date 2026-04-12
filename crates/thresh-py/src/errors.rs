//! Custom Python exception types for thresh bindings.

use pyo3::prelude::*;

// Define a custom exception: thresh_py.ThreshError inherits from RuntimeError.
pyo3::create_exception!(thresh_py, ThreshError, pyo3::exceptions::PyRuntimeError);

/// Convert a string error message into a `PyErr` wrapping `ThreshError`.
pub fn thresh_err(msg: impl Into<String>) -> PyErr {
    ThreshError::new_err(msg.into())
}

#[cfg(test)]
mod tests {
    // ThreshError is a PyO3 exception type; instantiation requires the Python
    // runtime. We verify the module compiles and the helper builds a message.
    #[test]
    fn test_thresh_err_message_construction() {
        // Ensure the helper accepts both &str and String.
        let _f = || super::thresh_err("some error");
        let _g = || super::thresh_err(String::from("another error"));
    }
}
