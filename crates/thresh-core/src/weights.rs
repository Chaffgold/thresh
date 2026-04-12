//! Weight metadata types for neural network model validation.
//!
//! Provides [`WeightManifest`] for declaring expected tensor names, shapes, and
//! data types, plus validation methods to check loaded tensors against the
//! manifest before using them.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// TensorShape
// ---------------------------------------------------------------------------

/// Newtype wrapping a tensor's shape as a list of dimension sizes.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TensorShape(pub Vec<usize>);

impl TensorShape {
    /// Total number of elements in the tensor.
    pub fn numel(&self) -> usize {
        self.0.iter().product()
    }

    /// Number of dimensions.
    pub fn ndim(&self) -> usize {
        self.0.len()
    }
}

// ---------------------------------------------------------------------------
// TensorDtype
// ---------------------------------------------------------------------------

/// Data type of a tensor's elements.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TensorDtype {
    /// 32-bit IEEE 754 floating point.
    F32,
    /// 16-bit IEEE 754 floating point.
    F16,
    /// 16-bit bfloat (brain float).
    BF16,
}

// ---------------------------------------------------------------------------
// TensorMeta
// ---------------------------------------------------------------------------

/// Metadata for a single tensor: shape and data type.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TensorMeta {
    /// Shape of the tensor.
    pub shape: TensorShape,
    /// Data type of the tensor elements.
    pub dtype: TensorDtype,
}

// ---------------------------------------------------------------------------
// WeightError
// ---------------------------------------------------------------------------

/// Errors that can occur during weight manifest operations.
#[derive(Debug)]
pub enum WeightError {
    /// A required tensor was not found.
    TensorNotFound(String),
    /// A tensor's shape does not match the expected shape.
    ShapeMismatch {
        name: String,
        expected: TensorShape,
        actual: TensorShape,
    },
    /// A tensor's data type does not match.
    DtypeMismatch {
        name: String,
        expected: TensorDtype,
        actual: TensorDtype,
    },
    /// The manifest or weight file was not found on disk.
    FileNotFound(String),
    /// JSON parsing error.
    ParseError(String),
    /// General I/O error.
    IoError(String),
}

impl std::fmt::Display for WeightError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            WeightError::TensorNotFound(name) => write!(f, "tensor not found: {name}"),
            WeightError::ShapeMismatch {
                name,
                expected,
                actual,
            } => write!(
                f,
                "shape mismatch for '{name}': expected {:?}, got {:?}",
                expected.0, actual.0
            ),
            WeightError::DtypeMismatch {
                name,
                expected,
                actual,
            } => write!(
                f,
                "dtype mismatch for '{name}': expected {expected:?}, got {actual:?}"
            ),
            WeightError::FileNotFound(path) => write!(f, "file not found: {path}"),
            WeightError::ParseError(msg) => write!(f, "parse error: {msg}"),
            WeightError::IoError(msg) => write!(f, "I/O error: {msg}"),
        }
    }
}

impl std::error::Error for WeightError {}

// ---------------------------------------------------------------------------
// WeightManifest
// ---------------------------------------------------------------------------

/// Declares expected tensor names, shapes, and data types for model validation.
///
/// Serialize to JSON to persist alongside model files; deserialize to validate
/// loaded weight sets.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WeightManifest {
    /// Map of tensor name to its expected metadata.
    pub tensors: HashMap<String, TensorMeta>,
}

impl WeightManifest {
    /// Create an empty manifest.
    pub fn new() -> Self {
        Self {
            tensors: HashMap::new(),
        }
    }

    /// Add an expected tensor entry (builder pattern).
    pub fn with_tensor(mut self, name: &str, shape: Vec<usize>, dtype: TensorDtype) -> Self {
        self.tensors.insert(
            name.to_string(),
            TensorMeta {
                shape: TensorShape(shape),
                dtype,
            },
        );
        self
    }

    /// Load a manifest from a JSON file.
    pub fn from_json(path: &str) -> Result<Self, WeightError> {
        let data = std::fs::read_to_string(path).map_err(|e| match e.kind() {
            std::io::ErrorKind::NotFound => WeightError::FileNotFound(path.to_string()),
            _ => WeightError::IoError(e.to_string()),
        })?;
        serde_json::from_str(&data).map_err(|e| WeightError::ParseError(e.to_string()))
    }

    /// Save the manifest as pretty-printed JSON.
    pub fn to_json(&self, path: &str) -> Result<(), WeightError> {
        let json = serde_json::to_string_pretty(self)
            .map_err(|e| WeightError::ParseError(e.to_string()))?;
        std::fs::write(path, json).map_err(|e| WeightError::IoError(e.to_string()))
    }

    /// Validate a single tensor against its expected entry.
    ///
    /// Checks that the tensor exists in the manifest and that the provided
    /// shape and dtype match the expected values.
    pub fn validate_tensor(
        &self,
        name: &str,
        shape: &TensorShape,
        dtype: TensorDtype,
    ) -> Result<(), WeightError> {
        let meta = self
            .tensors
            .get(name)
            .ok_or_else(|| WeightError::TensorNotFound(name.to_string()))?;

        if meta.shape != *shape {
            return Err(WeightError::ShapeMismatch {
                name: name.to_string(),
                expected: meta.shape.clone(),
                actual: shape.clone(),
            });
        }

        if meta.dtype != dtype {
            return Err(WeightError::DtypeMismatch {
                name: name.to_string(),
                expected: meta.dtype,
                actual: dtype,
            });
        }

        Ok(())
    }
}

impl Default for WeightManifest {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_manifest_json_roundtrip() {
        let manifest = WeightManifest::new()
            .with_tensor("encoder.weight", vec![256, 4], TensorDtype::F32)
            .with_tensor("encoder.bias", vec![256], TensorDtype::F16)
            .with_tensor("decoder.weight", vec![7, 256], TensorDtype::BF16);

        let json = serde_json::to_string_pretty(&manifest).unwrap();
        let loaded: WeightManifest = serde_json::from_str(&json).unwrap();

        assert_eq!(loaded.tensors.len(), 3);
        assert_eq!(
            loaded.tensors["encoder.weight"].shape,
            TensorShape(vec![256, 4])
        );
        assert_eq!(loaded.tensors["encoder.weight"].dtype, TensorDtype::F32);
        assert_eq!(loaded.tensors["encoder.bias"].shape, TensorShape(vec![256]));
        assert_eq!(loaded.tensors["encoder.bias"].dtype, TensorDtype::F16);
    }

    #[test]
    fn test_manifest_file_roundtrip() {
        let manifest = WeightManifest::new()
            .with_tensor("layer.0.weight", vec![128, 64], TensorDtype::F32)
            .with_tensor("layer.0.bias", vec![128], TensorDtype::F32);

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("manifest.json");
        let path_str = path.to_str().unwrap();

        manifest.to_json(path_str).unwrap();
        let loaded = WeightManifest::from_json(path_str).unwrap();

        assert_eq!(loaded.tensors.len(), 2);
        assert_eq!(
            loaded.tensors["layer.0.weight"].shape,
            TensorShape(vec![128, 64])
        );
    }

    #[test]
    fn test_validate_tensor_success() {
        let manifest = WeightManifest::new().with_tensor("weight", vec![256, 4], TensorDtype::F32);

        manifest
            .validate_tensor("weight", &TensorShape(vec![256, 4]), TensorDtype::F32)
            .unwrap();
    }

    #[test]
    fn test_validate_tensor_not_found() {
        let manifest = WeightManifest::new();

        let err = manifest
            .validate_tensor("missing", &TensorShape(vec![1]), TensorDtype::F32)
            .unwrap_err();
        assert!(matches!(err, WeightError::TensorNotFound(_)));
    }

    #[test]
    fn test_validate_tensor_shape_mismatch() {
        let manifest = WeightManifest::new().with_tensor("weight", vec![256, 4], TensorDtype::F32);

        let err = manifest
            .validate_tensor("weight", &TensorShape(vec![128, 4]), TensorDtype::F32)
            .unwrap_err();
        assert!(matches!(err, WeightError::ShapeMismatch { .. }));
    }

    #[test]
    fn test_validate_tensor_dtype_mismatch() {
        let manifest = WeightManifest::new().with_tensor("weight", vec![256, 4], TensorDtype::F32);

        let err = manifest
            .validate_tensor("weight", &TensorShape(vec![256, 4]), TensorDtype::F16)
            .unwrap_err();
        assert!(matches!(err, WeightError::DtypeMismatch { .. }));
    }

    #[test]
    fn test_tensor_shape_numel() {
        let shape = TensorShape(vec![3, 4, 5]);
        assert_eq!(shape.numel(), 60);
        assert_eq!(shape.ndim(), 3);
    }

    #[test]
    fn test_manifest_from_json_file_not_found() {
        let err = WeightManifest::from_json("/nonexistent/path.json").unwrap_err();
        assert!(matches!(err, WeightError::FileNotFound(_)));
    }

    #[test]
    fn test_weight_error_display() {
        let err = WeightError::TensorNotFound("foo".to_string());
        assert!(err.to_string().contains("foo"));

        let err = WeightError::ShapeMismatch {
            name: "bar".to_string(),
            expected: TensorShape(vec![2, 3]),
            actual: TensorShape(vec![4, 5]),
        };
        assert!(err.to_string().contains("bar"));
    }
}
