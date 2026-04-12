## 1. Weight Metadata Types (thresh-core)

- [ ] 1.1 Add `crates/thresh-core/src/weights.rs` module with `TensorShape(Vec<usize>)` newtype, `TensorDtype` enum (`F32`, `F16`, `BF16`), and `TensorMeta { shape: TensorShape, dtype: TensorDtype }`.
- [ ] 1.2 Define `WeightManifest` struct: `HashMap<String, TensorMeta>` with serde Serialize/Deserialize for JSON round-tripping.
- [ ] 1.3 Implement `WeightManifest::from_json(path)` and `WeightManifest::to_json(path)` for file I/O.
- [ ] 1.4 Add validation methods: `WeightManifest::validate_tensor(name, shape, dtype) -> Result<(), WeightError>` that checks a tensor against its expected entry.
- [ ] 1.5 Define `WeightError` enum with variants: `TensorNotFound`, `ShapeMismatch { expected, actual }`, `DtypeMismatch`, `FileNotFound`, `ParseError`, `IoError`.
- [ ] 1.6 Re-export `weights` module from `thresh-core/src/lib.rs`. Add unit tests for manifest JSON round-trip and validation.

## 2. WeightLoader Trait (thresh-inference)

- [ ] 2.1 Define `WeightLoader` trait in `crates/thresh-inference/src/weights/mod.rs` with methods: `load_tensor(name) -> Result<DMatrix<f32>>`, `tensor_names() -> Vec<String>`, `tensor_shape(name) -> Result<(usize, usize)>`, `validate(manifest) -> Result<()>`.
- [ ] 2.2 Add `weights` module to thresh-inference `lib.rs` with re-exports.
- [ ] 2.3 Write trait documentation with usage examples and contract specifications (error behavior, thread safety).

## 3. SafeTensors Loader (thresh-inference)

- [ ] 3.1 Add `safetensors = "0.4"` and `memmap2` as optional dependencies in thresh-inference's Cargo.toml, gated behind `features = ["safetensors"]`.
- [ ] 3.2 Implement `SafeTensorsLoader` struct that opens a .safetensors file via memory-mapped I/O (`memmap2::Mmap`).
- [ ] 3.3 Implement `WeightLoader::load_tensor` for `SafeTensorsLoader`: read named tensor from mmap, validate dtype, convert raw bytes to `DMatrix<f32>` with correct shape. Handle f16/bf16 by upcasting to f32.
- [ ] 3.4 Implement `WeightLoader::tensor_names` by iterating the SafeTensors metadata header.
- [ ] 3.5 Implement `WeightLoader::tensor_shape` by reading shape from the SafeTensors metadata without loading tensor data.
- [ ] 3.6 Implement `WeightLoader::validate` by checking all manifest entries against the SafeTensors file's metadata.

## 4. Detector Integration (thresh-inference)

- [ ] 4.1 Add `load_weights(&mut self, path: &Path) -> Result<(), WeightError>` method to the `OnnxDetector` struct (or a shared detector trait) for weight override.
- [ ] 4.2 Implement atomic weight swap: load all tensors into a temporary buffer, validate all shapes, then replace internal weight matrices only if all validations pass.
- [ ] 4.3 Add a `WeightSet` struct that holds all loaded weight matrices as a `HashMap<String, DMatrix<f32>>`, providing typed access by layer name.

## 5. Tests

- [ ] 5.1 Unit test: `WeightManifest` JSON serialization round-trip (serialize, deserialize, compare).
- [ ] 5.2 Unit test: `WeightManifest::validate_tensor` rejects shape mismatches and unknown tensor names.
- [ ] 5.3 Create a small test .safetensors file (2-3 tensors, known shapes) using the safetensors Python library, commit to `test-data/`.
- [ ] 5.4 Unit test: `SafeTensorsLoader` loads known tensors from the test file with correct shapes and values.
- [ ] 5.5 Unit test: `SafeTensorsLoader::validate` succeeds with a matching manifest and fails with a mismatched manifest.
- [ ] 5.6 Unit test: `load_weights` on a detector succeeds with valid weights and fails atomically with invalid weights (previous weights preserved).
- [ ] 5.7 Integration test: load a complete weight set from SafeTensors, verify all tensors are accessible and have expected dimensions.
- [ ] 5.8 Add a CI job note ensuring SafeTensors tests only run when the `safetensors` feature is enabled.
