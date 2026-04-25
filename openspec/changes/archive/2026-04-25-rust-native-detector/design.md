## Context

thresh-inference currently requires ONNX Runtime (via the `ort` crate) for all neural network inference. ORT is a large C++ library that complicates cross-compilation, increases binary size, and is unavailable on some embedded targets. The `safetensors-weights` change adds a Rust-native weight loading mechanism. This change builds on that foundation by implementing the inference computation itself in pure Rust, creating a complete ORT-free inference path.

The target architecture is a simplified DETR (DEtection TRansformer) decoder. The full DETR model has an encoder (processes image features) and a decoder (transforms object queries into detections). This implementation covers only the decoder and detection head, assuming pre-computed encoder features or operating directly on object queries. This is sufficient for the tracker's detection pipeline where the encoder can be replaced by sensor-specific feature extraction.

thresh already uses nalgebra extensively for matrix math in the filter and association crates. The native detector extends this to transformer operations (attention, FFN, layer norm).

## Goals / Non-Goals

**Goals:**
- Implement multi-head self-attention with configurable heads and dimensions
- Implement feed-forward network with GELU activation
- Implement layer normalization with learnable parameters
- Implement sinusoidal positional encoding
- Compose into a 6-layer transformer decoder with 256-dim embeddings and 8 heads
- Implement detection head (class logits + bounding box regression)
- Load all weights from SafeTensors via the `WeightLoader` trait
- Feature-gate behind `native-detector`
- Achieve correct numerical output matching a PyTorch reference implementation

**Non-Goals:**
- Training, backpropagation, or automatic differentiation
- The encoder half of DETR
- f16/bf16 precision
- Batch sizes > 1
- GPU acceleration
- Performance parity with ORT (correctness over speed)
- Non-maximum suppression (downstream concern)

## Decisions

### 1. Module structure under `thresh-inference/src/native/`

**Decision:** Organize the native detector as a submodule tree:

```
src/native/
    mod.rs           -- NativeDetector struct, feature gate
    attention.rs     -- MultiHeadAttention
    ffn.rs           -- FeedForwardNetwork
    layer_norm.rs    -- LayerNorm
    positional.rs    -- SinusoidalPositionalEncoding
    decoder.rs       -- TransformerDecoderLayer, TransformerDecoder
    detection_head.rs -- DetectionHead (class logits + bbox)
```

**Rationale:** Each component is independently testable. The module tree mirrors the logical architecture of the transformer. All gated behind `#[cfg(feature = "native-detector")]`.

### 2. nalgebra `DMatrix<f32>` for all tensor operations

**Decision:** Use `nalgebra::DMatrix<f32>` as the tensor type for all intermediate computations. No new tensor library dependency.

**Rationale:** nalgebra is already a core dependency across the workspace. While it lacks native batched matmul and GPU support, for batch-size-1 inference on CPU it is sufficient. Adding a tensor library (ndarray, burn, candle) would introduce a significant dependency for a feature that is intentionally lightweight. The performance trade-off is acceptable because the native detector targets edge/embedded use cases where inference speed is secondary to deployment simplicity.

### 3. Fixed architecture with configurable depth/width

**Decision:** The default architecture is 6 layers, 256-dim, 8 heads, 1024-dim FFN. These are configurable via `NativeDetectorConfig`:

```
pub struct NativeDetectorConfig {
    pub num_layers: usize,     // default: 6
    pub embed_dim: usize,      // default: 256
    pub num_heads: usize,      // default: 8
    pub ffn_dim: usize,        // default: 1024
    pub num_queries: usize,    // default: 100
    pub num_classes: usize,    // configurable per dataset
}
```

**Rationale:** Fixed defaults match the DETR-small configuration and are sufficient for the target use case. Configurability allows experimentation without code changes. The config determines the expected weight shapes, which are validated at load time.

### 4. Scaled dot-product attention implementation

**Decision:** Implement attention as:
1. Project input: `Q = X * W_q + b_q`, `K = X * W_k + b_k`, `V = X * W_v + b_v`
2. Reshape to heads: split the `embed_dim` dimension into `num_heads` chunks of `head_dim`
3. Attention: `A = softmax(Q * K^T / sqrt(head_dim)) * V`
4. Concatenate heads and project: `out = concat(heads) * W_o + b_o`

Softmax is computed in a numerically stable way (subtract max before exp).

**Rationale:** This is the standard transformer attention mechanism. The per-head computation avoids allocating a single large `(num_heads * seq_len, seq_len)` attention matrix. Stable softmax prevents overflow for large attention logits.

### 5. GELU activation

**Decision:** Implement GELU using the exact formula: `gelu(x) = 0.5 * x * (1 + erf(x / sqrt(2)))`. Use the `libm` crate for `erf` if `std::f32::erf` is not available, or use the tanh approximation: `gelu(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))`.

**Rationale:** GELU is the standard activation in transformers. The tanh approximation is faster and matches PyTorch's default `gelu` implementation, ensuring numerical compatibility with reference weights.

### 6. Layer normalization with learnable parameters

**Decision:** Implement layer norm as: `y = (x - mean) / sqrt(var + eps) * gamma + beta`, where `gamma` (scale) and `beta` (bias) are learnable parameters loaded from SafeTensors. `eps = 1e-5`.

**Rationale:** Standard transformer layer norm. The learnable parameters are essential for correctness -- using fixed scale=1, bias=0 would produce incorrect outputs.

### 7. Inference-only, no autograd

**Decision:** All operations are forward-only. No computation graph is recorded, no gradient storage, no backward pass.

**Rationale:** thresh is a tracking framework, not a training framework. Inference-only simplifies the implementation dramatically and eliminates the need for an autograd engine. Model training happens in PyTorch; only the trained weights are loaded into thresh.

### 8. Helper functions in thresh-core

**Decision:** Add a `tensor_ops` module to thresh-core with pure functions: `softmax(v: &DVector<f32>) -> DVector<f32>`, `gelu(x: f32) -> f32`, `layer_norm(x: &DVector<f32>, gamma: &DVector<f32>, beta: &DVector<f32>, eps: f32) -> DVector<f32>`.

**Rationale:** These are general-purpose numerical operations that may be useful outside the native detector (e.g., in fusion or evaluation). Placing them in thresh-core makes them available workspace-wide.

## Risks / Trade-offs

**[Risk] Numerical divergence from PyTorch reference.** Floating-point operation ordering, reduction algorithms, and activation function approximations may produce slightly different results than PyTorch. Mitigation: add a numerical equivalence test suite that compares Rust outputs against saved PyTorch reference outputs for fixed inputs, with a tolerance of 1e-5.

**[Risk] Performance significantly worse than ORT.** nalgebra's BLAS operations may be 5-10x slower than ORT's optimized kernels for the matrix sizes involved. Mitigation: this is acceptable for the target use case (edge deployment where ORT is unavailable). Document the performance trade-off. Consider optional BLAS backend (OpenBLAS, MKL) via nalgebra's feature flags for users who need speed.

**[Trade-off] No batching support.** Batch size is fixed at 1, meaning each inference call processes one set of queries. This simplifies the implementation but prevents throughput optimization via batching. Acceptable for real-time tracking where latency matters more than throughput.

**[Trade-off] Decoder-only architecture.** Omitting the encoder means the native detector cannot process raw images. It operates on pre-computed features or object queries. This limits standalone use but matches the tracker's architecture where sensor-specific feature extraction is a separate concern.

**[Risk] Large weight files for the simplified architecture.** Even the 6-layer, 256-dim decoder has ~5M parameters (~20MB in f32 SafeTensors). For embedded deployment this may be significant. Mitigation: document weight file sizes; f16 weight support can halve this in a future iteration.

## Open Questions

- Should the native detector expose intermediate layer outputs for debugging/visualization?
- Should we provide pre-trained weights for a reference dataset, or only the architecture?
- Is the tanh GELU approximation sufficient, or should we support both exact and approximate?
- Should the `NativeDetectorConfig` support cross-attention (decoder attending to encoder output) for future encoder integration?
- Should we benchmark against `candle` or `burn` as alternative pure-Rust inference backends before committing to raw nalgebra?
