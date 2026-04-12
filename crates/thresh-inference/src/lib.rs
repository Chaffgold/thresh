//! ONNX Runtime integration for modular transformer inference pipelines.
//!
//! The `onnx` feature must be enabled to use actual ONNX Runtime sessions.
//! Without it, only types and pipeline configuration are available.

pub mod detection;
pub mod pipeline;
pub mod preprocess;

#[cfg(feature = "onnx")]
pub mod session;
