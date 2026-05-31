//! ONNX Runtime session wrapper (requires `onnx` feature).
//!
//! A thin, reusable wrapper over [`ort::session::Session`] that loads an ONNX
//! model and runs single-input / single-output `f32` inference. It is the
//! shared ONNX execution primitive for the framework's learned components —
//! the Track B IMM mode classifier ([`thresh-filter`]'s `learned-imm` feature)
//! today, and the Track A detector in due course.

use std::path::Path;

use ort::session::Session;
use ort::value::Tensor;

/// A loaded ONNX model exposing a minimal `f32` run interface.
pub struct OnnxModel {
    session: Session,
}

impl OnnxModel {
    /// Load an ONNX model from a file at `path`.
    ///
    /// # Errors
    /// Returns an [`ort::Error`] if the model cannot be read or parsed.
    pub fn load(path: impl AsRef<Path>) -> Result<Self, ort::Error> {
        let session = Session::builder()?.commit_from_file(path.as_ref())?;
        Ok(Self { session })
    }

    /// Borrow the underlying session (e.g. to introspect input/output metadata).
    pub fn session(&self) -> &Session {
        &self.session
    }

    /// Run the model on a single row-major `f32` input tensor of the given
    /// `shape`, returning the first output's flattened `f32` data.
    ///
    /// # Errors
    /// Returns an [`ort::Error`] if tensor construction or inference fails.
    ///
    /// # Panics
    /// If `input.len()` does not equal the product of `shape`.
    pub fn run_f32(&mut self, input: &[f32], shape: &[i64]) -> Result<Vec<f32>, ort::Error> {
        let numel: i64 = shape.iter().product();
        assert_eq!(
            numel as usize,
            input.len(),
            "input length {} != product of shape {shape:?}",
            input.len(),
        );
        let tensor = Tensor::from_array((shape.to_vec(), input.to_vec()))?;
        let outputs = self.session.run(ort::inputs![tensor])?;
        let (_out_shape, out_data) = outputs[0].try_extract_tensor::<f32>()?;
        Ok(out_data.to_vec())
    }
}
