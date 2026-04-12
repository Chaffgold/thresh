## Context

thresh-inference currently depends on ONNX Runtime for all neural network inference. Model weights are embedded within the ONNX graph file, meaning any weight update requires re-exporting the entire model. The SafeTensors format (Hugging Face standard) stores weights independently in a safe, memory-mapped binary format. Adding SafeTensors support enables a separation between model architecture (defined in Rust code or ONNX graph) and model weights (loaded from .safetensors files at runtime). This is a prerequisite for the `rust-native-detector` change, which needs a weight loading mechanism for its pure-Rust inference path.

thresh-core provides nalgebra-based types for state vectors and covariance matrices, but has no types for representing weight tensor metadata. thresh-inference has the `onnx` feature gate pattern established, providing a model for gating the SafeTensors dependency.

## Goals / Non-Goals

**Goals:**
- Define a `WeightLoader` trait that abstracts weight loading and enables future format support
- Implement `SafeTensorsLoader` using the `safetensors` crate (v0.4) with memory-mapped I/O
- Map SafeTensors tensors to `nalgebra::DMatrix<f32>` with shape validation
- Provide a JSON manifest format for declaring expected tensor names, shapes, and dtypes
- Integrate weight loading into detector structs via `load_weights(path)`
- Add `WeightMetadata` types to thresh-core for tensor shape and dtype representation
- Feature-gate behind `safetensors` feature flag

**Non-Goals:**
- Supporting other weight formats (GGUF, pickle, NumPy)
- Training, fine-tuning, or gradient computation
- GPU memory placement or device-specific loading
- Automatic architecture inference from weight shapes
- Weight quantization or compression

## Decisions

### 1. `WeightLoader` trait in thresh-inference

**Decision:** Define a `WeightLoader` trait as the abstraction layer for loading named tensors from any serialization format:

```
pub trait WeightLoader: Send + Sync {
    fn load_tensor(&self, name: &str) -> Result<DMatrix<f32>, WeightError>;
    fn tensor_names(&self) -> Vec<String>;
    fn validate(&self, manifest: &WeightManifest) -> Result<(), WeightError>;
    fn tensor_shape(&self, name: &str) -> Result<(usize, usize), WeightError>;
}
```

**Rationale:** Trait-based abstraction allows future formats (GGUF, custom binary) to slot in without changing downstream code. The trait lives in thresh-inference because it is an inference concern, not a core type.

### 2. `SafeTensorsLoader` uses memory-mapped I/O

**Decision:** `SafeTensorsLoader` opens the .safetensors file using `safetensors::SafeTensors::deserialize` with a memory-mapped byte slice (via `memmap2`). Tensor data is accessed zero-copy from the mmap and converted to `DMatrix<f32>` only when `load_tensor` is called.

**Rationale:** Memory-mapped I/O avoids reading the entire file into heap memory upfront. For large weight files (hundreds of MB), this significantly reduces memory pressure and startup time. The `safetensors` crate natively supports operating on byte slices, making mmap integration straightforward. The `memmap2` crate is already a transitive dependency via `ort`.

### 3. JSON manifest for shape validation

**Decision:** Define a `WeightManifest` struct serialized as JSON that maps tensor names to expected `(rows, cols)` shapes and `f32`/`f16`/`bf16` dtype:

```json
{
  "tensors": {
    "decoder.layer.0.self_attn.q_proj.weight": {"shape": [256, 256], "dtype": "f32"},
    "decoder.layer.0.self_attn.q_proj.bias": {"shape": [256, 1], "dtype": "f32"}
  }
}
```

**Rationale:** Explicit manifests catch weight file mismatches (wrong model version, corrupted file, architecture change) at load time rather than producing silent incorrect inference results. JSON is human-readable and easy to generate from the Python training pipeline.

### 4. `DMatrix<f32>` as the target type for weight tensors

**Decision:** All loaded weight tensors are returned as `nalgebra::DMatrix<f32>`. Bias vectors are represented as single-column matrices `DMatrix<f32>` with shape `(n, 1)`.

**Rationale:** nalgebra's `DMatrix` is the dynamically-sized matrix type used throughout thresh. Using `f32` matches the native detector's precision. Representing biases as column matrices rather than `DVector` keeps the API uniform and simplifies downstream matmul code (`W * x + b` where both are DMatrix).

### 5. Feature gate: `safetensors`

**Decision:** The `safetensors` dependency and all SafeTensors-specific code are gated behind `features = ["safetensors"]` in thresh-inference's Cargo.toml. The `WeightLoader` trait itself is always available (it has no external dependencies).

**Rationale:** Users who only need ONNX inference should not pay the compile-time cost of the safetensors crate. The trait remains ungated so the native detector (gated behind `native-detector`) can depend on it without also requiring `safetensors` -- enabling alternative loader implementations.

### 6. Integration via `load_weights(path)` on detector structs

**Decision:** Add a `load_weights(&mut self, path: &Path) -> Result<(), WeightError>` method to detector structs. This method: (1) creates a `SafeTensorsLoader` from the path, (2) validates against the detector's manifest, (3) loads all tensors, (4) replaces internal weight matrices atomically.

**Rationale:** The method-level integration means weights can be hot-swapped on a live detector without reconstructing the entire pipeline. Atomic replacement (all-or-nothing) prevents partial state where some layers have new weights and others have old ones.

### 7. `WeightMetadata` types in thresh-core

**Decision:** Add a `weights` module to thresh-core with: `TensorShape(Vec<usize>)`, `TensorDtype` enum (`F32`, `F16`, `BF16`), `TensorMeta { shape, dtype }`, and `WeightManifest { tensors: HashMap<String, TensorMeta> }`.

**Rationale:** These types need to be shared between thresh-inference (which uses them) and potentially thresh-eval or thresh-synth (which might generate test weight files). Placing them in thresh-core, which is the shared foundation, avoids circular dependencies.

## Risks / Trade-offs

**[Risk] SafeTensors crate version stability.** The `safetensors` crate is at v0.4 and is actively developed by Hugging Face. A breaking change in v0.5 could require updates. Mitigation: pin to `safetensors = "0.4"` and encapsulate all usage behind the `WeightLoader` trait so updates are localized.

**[Risk] Memory-mapped files and concurrent access.** If a .safetensors file is modified or deleted while the loader holds an mmap, behavior is platform-dependent (segfault on some systems). Mitigation: document that weight files must not be modified while loaded; consider copying file contents to heap on platforms where mmap safety is a concern.

**[Trade-off] f32-only weight loading vs multi-precision.** Only f32 weights are supported initially. f16 and bf16 weights in SafeTensors files are upcast to f32 on load. This simplifies the API but doubles memory for models stored in f16. Acceptable for v0.3.0; native f16 support can be added later.

**[Trade-off] JSON manifest vs self-describing validation.** A separate manifest file is an extra artifact to maintain. Alternative: infer expected shapes from the model architecture at compile time. Chosen approach is more flexible (works for arbitrary architectures) at the cost of requiring manifest generation.

## Open Questions

- Should the manifest be embedded in the SafeTensors file's metadata header, or always a separate JSON file?
- Should `WeightLoader` support lazy loading (load tensors on demand) or eager loading (load all at construction)?
- Is `memmap2` the right mmap crate, or should we use `mmap-rs` for better safety guarantees?
- Should the `WeightLoader` trait support async loading for large files?
