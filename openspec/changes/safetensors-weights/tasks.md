## 1. Weight Metadata Types (thresh-core)

- [x] 1.1 Add `crates/thresh-core/src/weights.rs` module with `TensorShape(Vec<usize>)` newtype, `TensorDtype` enum (`F32`, `F16`, `BF16`), and `TensorMeta { shape: TensorShape, dtype: TensorDtype }`.
- [x] 1.2 Define `WeightManifest` struct: `HashMap<String, TensorMeta>` with serde Serialize/Deserialize for JSON round-tripping.
- [x] 1.3 Implement `WeightManifest::from_json(path)` and `WeightManifest::to_json(path)` for file I/O.
- [x] 1.4 Add validation methods: `WeightManifest::validate_tensor(name, shape, dtype) -> Result<(), WeightError>` that checks a tensor against its expected entry.
- [x] 1.5 Define `WeightError` enum with variants: `TensorNotFound`, `ShapeMismatch { expected, actual }`, `DtypeMismatch`, `FileNotFound`, `ParseError`, `IoError`.
- [x] 1.6 Re-export `weights` module from `thresh-core/src/lib.rs`. Add unit tests for manifest JSON round-trip and validation.

## 2. WeightLoader Trait (thresh-inference)

- [x] 2.1 Define `WeightLoader` trait in `crates/thresh-inference/src/weights.rs` with `load(path) -> Result<WeightSet>` method.
- [x] 2.2 Add `weights` module to thresh-inference `lib.rs` with re-exports.
- [x] 2.3 Write trait documentation with usage examples and contract specifications (error behavior, thread safety).

## 3. SafeTensors Loader (thresh-inference)

- [x] 3.1 Add `safetensors = "0.4"` as a dependency in thresh-inference's Cargo.toml (non-optional, pure-Rust).
- [x] 3.2 Implement `SafeTensorsLoader` struct that reads .safetensors files via `std::fs::read`.
- [x] 3.3 Implement `WeightLoader::load` for `SafeTensorsLoader`: deserialize file, convert tensor views to `DMatrix<f32>` with correct row-major to column-major handling.
- [x] 3.4 Implement `WeightSet::tensor_names` returning sorted list of tensor names.
- [x] 3.5 Implement `WeightSet::get` and `get_or_err` for named tensor access.
- [x] 3.6 Implement `WeightSet::validate_shapes` checking all manifest entries against loaded tensor shapes.

## 4. Detector Integration (thresh-inference)

- [x] 4.1 ~~Add `load_weights(&mut self, path: &Path) -> Result<(), WeightError>` method to the `OnnxDetector` struct (or a shared detector trait) for weight override.~~ **Deferred** — requires `onnx` feature and ORT binary for testing.
- [x] 4.2 ~~Implement atomic weight swap: load all tensors into a temporary buffer, validate all shapes, then replace internal weight matrices only if all validations pass.~~ **Deferred** — requires `onnx` feature and ORT binary for testing.
- [x] 4.3 Add a `WeightSet` struct that holds all loaded weight matrices as a `HashMap<String, DMatrix<f32>>`, providing typed access by layer name.

## 5. Tests

- [x] 5.1 Unit test: `WeightManifest` JSON serialization round-trip (serialize, deserialize, compare).
- [x] 5.2 Unit test: `WeightManifest::validate_tensor` rejects shape mismatches and unknown tensor names.
- [x] 5.3 Create a small test .safetensors file (3 tensors, known shapes) using the safetensors Python library at `test-data/models/test_weights.safetensors`.
- [x] 5.4 Unit test: `SafeTensorsLoader` loads known tensors from the test file with correct shapes and names.
- [x] 5.5 Unit test: `WeightSet::validate_shapes` succeeds with a matching manifest and fails with a mismatched manifest.
- [x] 5.6 ~~Unit test: `load_weights` on a detector succeeds with valid weights and fails atomically with invalid weights (previous weights preserved).~~ **Deferred** — requires `onnx` feature and ORT binary for testing.
- [x] 5.7 Integration test: load a complete weight set from SafeTensors, verify all tensors are accessible and have expected dimensions.
- [x] 5.8 SafeTensors tests run in the default feature set since `safetensors` is a non-optional (always-on) dependency. No feature gate needed; CI runs these tests via `cargo test --workspace`.
