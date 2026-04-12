//! SafeTensors weight loading for neural network inference.
//!
//! Provides a [`WeightLoader`] trait for abstracting weight file formats,
//! with a [`SafeTensorsLoader`] implementation that reads `.safetensors` files
//! and converts tensors to `nalgebra::DMatrix<f32>`.

use nalgebra::DMatrix;
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// Errors that can occur during weight loading and validation.
#[derive(Debug)]
pub enum WeightError {
    /// I/O error (file not found, permission denied, etc.)
    Io(String),
    /// File format / deserialization error.
    Format(String),
    /// A required tensor was not found in the weight file.
    MissingTensor(String),
    /// A tensor's shape does not match the expected shape.
    ShapeMismatch {
        name: String,
        expected: Vec<usize>,
        got: Vec<usize>,
    },
}

impl std::fmt::Display for WeightError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            WeightError::Io(msg) => write!(f, "weight I/O error: {msg}"),
            WeightError::Format(msg) => write!(f, "weight format error: {msg}"),
            WeightError::MissingTensor(name) => write!(f, "missing tensor: {name}"),
            WeightError::ShapeMismatch {
                name,
                expected,
                got,
            } => {
                write!(
                    f,
                    "shape mismatch for tensor '{name}': expected {expected:?}, got {got:?}"
                )
            }
        }
    }
}

impl std::error::Error for WeightError {}

// ---------------------------------------------------------------------------
// WeightLoader trait
// ---------------------------------------------------------------------------

/// Abstraction for loading weight tensors from a file.
pub trait WeightLoader {
    /// Load all tensors from the file at `path` into a [`WeightSet`].
    fn load(&self, path: &str) -> Result<WeightSet, WeightError>;
}

// ---------------------------------------------------------------------------
// WeightSet
// ---------------------------------------------------------------------------

/// A collection of named weight tensors.
#[derive(Debug)]
pub struct WeightSet {
    /// Map of tensor name to nalgebra `DMatrix<f32>`.
    pub tensors: HashMap<String, DMatrix<f32>>,
}

impl WeightSet {
    /// Look up a tensor by name.
    pub fn get(&self, name: &str) -> Option<&DMatrix<f32>> {
        self.tensors.get(name)
    }

    /// Look up a tensor by name, returning an error if missing.
    pub fn get_or_err(&self, name: &str) -> Result<&DMatrix<f32>, WeightError> {
        self.tensors
            .get(name)
            .ok_or_else(|| WeightError::MissingTensor(name.to_string()))
    }

    /// Return sorted list of tensor names.
    pub fn tensor_names(&self) -> Vec<&str> {
        let mut names: Vec<&str> = self.tensors.keys().map(|s| s.as_str()).collect();
        names.sort();
        names
    }

    /// Validate that all tensors in the manifest are present and have the
    /// expected shapes. Extra tensors in the weight set are ignored.
    pub fn validate_shapes(&self, manifest: &WeightManifest) -> Result<(), WeightError> {
        for (name, expected_shape) in &manifest.expected {
            let mat = self
                .tensors
                .get(name)
                .ok_or_else(|| WeightError::MissingTensor(name.clone()))?;

            let got = vec![mat.nrows(), mat.ncols()];
            if got != *expected_shape {
                return Err(WeightError::ShapeMismatch {
                    name: name.clone(),
                    expected: expected_shape.clone(),
                    got,
                });
            }
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// WeightManifest
// ---------------------------------------------------------------------------

/// Declares expected tensor names and shapes for validation.
pub struct WeightManifest {
    /// Map of tensor name to expected `[rows, cols]` shape.
    pub expected: HashMap<String, Vec<usize>>,
}

impl WeightManifest {
    /// Create an empty manifest.
    pub fn new() -> Self {
        Self {
            expected: HashMap::new(),
        }
    }

    /// Add an expected tensor entry (builder pattern).
    pub fn expect(mut self, name: &str, shape: Vec<usize>) -> Self {
        self.expected.insert(name.to_string(), shape);
        self
    }
}

impl Default for WeightManifest {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// SafeTensorsLoader
// ---------------------------------------------------------------------------

/// Loads weight tensors from `.safetensors` files.
pub struct SafeTensorsLoader;

impl WeightLoader for SafeTensorsLoader {
    fn load(&self, path: &str) -> Result<WeightSet, WeightError> {
        let data = std::fs::read(path).map_err(|e| WeightError::Io(e.to_string()))?;
        let tensors = safetensors::SafeTensors::deserialize(&data)
            .map_err(|e| WeightError::Format(e.to_string()))?;

        let mut map = HashMap::new();
        for (name, view) in tensors.iter() {
            let shape = view.shape();
            let raw = view.data();
            let floats: Vec<f32> = raw
                .chunks(4)
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect();

            // safetensors stores row-major; nalgebra is column-major, so use
            // from_row_slice to handle the transpose.
            let mat = if shape.len() == 2 {
                DMatrix::from_row_slice(shape[0], shape[1], &floats)
            } else if shape.len() == 1 {
                DMatrix::from_row_slice(shape[0], 1, &floats)
            } else {
                // Flatten higher-dim tensors: first dim = rows, rest = cols.
                let rows = shape[0];
                let cols: usize = shape[1..].iter().product();
                DMatrix::from_row_slice(rows, cols, &floats)
            };
            map.insert(name.to_string(), mat);
        }

        Ok(WeightSet { tensors: map })
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: path to the test weights file relative to the workspace root.
    fn test_weights_path() -> String {
        let manifest_dir = env!("CARGO_MANIFEST_DIR");
        format!("{manifest_dir}/../../test-data/models/test_weights.safetensors")
    }

    #[test]
    fn test_load_safetensors_file() {
        let loader = SafeTensorsLoader;
        let ws = loader.load(&test_weights_path()).unwrap();

        assert_eq!(ws.tensors.len(), 3);

        let mut names = ws.tensor_names();
        names.sort();
        assert_eq!(
            names,
            vec![
                "decoder.head.weight",
                "encoder.layer.0.bias",
                "encoder.layer.0.weight",
            ]
        );
    }

    #[test]
    fn test_tensor_shapes() {
        let loader = SafeTensorsLoader;
        let ws = loader.load(&test_weights_path()).unwrap();

        let w = ws.get("encoder.layer.0.weight").unwrap();
        assert_eq!((w.nrows(), w.ncols()), (256, 4));

        let b = ws.get("encoder.layer.0.bias").unwrap();
        assert_eq!((b.nrows(), b.ncols()), (256, 1));

        let h = ws.get("decoder.head.weight").unwrap();
        assert_eq!((h.nrows(), h.ncols()), (7, 256));
    }

    #[test]
    fn test_validate_shapes_passes() {
        let loader = SafeTensorsLoader;
        let ws = loader.load(&test_weights_path()).unwrap();

        let manifest = WeightManifest::new()
            .expect("encoder.layer.0.weight", vec![256, 4])
            .expect("encoder.layer.0.bias", vec![256, 1])
            .expect("decoder.head.weight", vec![7, 256]);

        ws.validate_shapes(&manifest).unwrap();
    }

    #[test]
    fn test_validate_shapes_fails_on_mismatch() {
        let loader = SafeTensorsLoader;
        let ws = loader.load(&test_weights_path()).unwrap();

        let manifest = WeightManifest::new().expect("encoder.layer.0.weight", vec![128, 4]);

        let err = ws.validate_shapes(&manifest).unwrap_err();
        assert!(matches!(err, WeightError::ShapeMismatch { .. }));
    }

    #[test]
    fn test_missing_file_error() {
        let loader = SafeTensorsLoader;
        let err = loader.load("/nonexistent/path.safetensors").unwrap_err();
        assert!(matches!(err, WeightError::Io(_)));
    }

    #[test]
    fn test_get_or_err_missing_tensor() {
        let loader = SafeTensorsLoader;
        let ws = loader.load(&test_weights_path()).unwrap();

        let err = ws.get_or_err("nonexistent.tensor").unwrap_err();
        assert!(matches!(err, WeightError::MissingTensor(_)));
    }

    #[test]
    fn test_weight_set_tensor_names() {
        let loader = SafeTensorsLoader;
        let ws = loader.load(&test_weights_path()).unwrap();

        let names = ws.tensor_names();
        assert_eq!(names.len(), 3);
        // tensor_names() returns sorted
        assert_eq!(names[0], "decoder.head.weight");
        assert_eq!(names[1], "encoder.layer.0.bias");
        assert_eq!(names[2], "encoder.layer.0.weight");
    }
}
