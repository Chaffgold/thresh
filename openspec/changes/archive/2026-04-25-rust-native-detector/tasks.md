## 1. Module Structure (thresh-inference)

- [x] 1.1 Create `crates/thresh-inference/src/native_detector.rs` (single-file module, always compiled — no feature gate needed since it has no heavy deps). Added `pub mod native_detector;` to lib.rs.
- [x] 1.2 All components (attention, FFN, layer norm, decoder, detection head) implemented in `native_detector.rs`.
- [x] 1.3 Define `NativeDetectorConfig` struct with fields: `d_model`, `n_heads`, `d_ff`, `n_layers`, `n_queries`, `n_classes`, `confidence_threshold`, `nms_iou_threshold`. Defaults: 256, 8, 1024, 6, 100, 10, 0.5, 0.4.

## 2. Tensor Ops Helpers (thresh-inference native_detector)

- [x] 2.1 Implemented `softmax_rows(m: &DMatrix<f32>) -> DMatrix<f32>` using log-sum-exp for numerical stability (in native_detector.rs).
- [x] 2.2 ~~Implement `gelu(x: f32) -> f32` using the tanh approximation.~~ Not needed — using ReLU per design spec.
- [x] 2.3 Implemented `layer_norm(x: &DMatrix<f32>, gamma: &DVector<f32>, beta: &DVector<f32>, eps: f32) -> DMatrix<f32>`.
- [x] 2.4 Implemented `softmax_rows` that applies softmax along each row (for attention weights).
- [x] 2.5 Unit tests for relu, softmax_rows, and layer_norm.

## 3. Multi-Head Self-Attention (thresh-inference)

- [x] 3.1 Define `MultiHeadAttention` struct with weight matrices: `w_q`, `w_k`, `w_v`, `w_o` (all `DMatrix<f32>`), config: `n_heads`, `d_head`, `d_model`.
- [x] 3.2 Implement `MultiHeadAttention::forward(x: &DMatrix<f32>) -> DMatrix<f32>`: compute Q, K, V projections, split into heads, scaled dot-product attention, concatenate, output projection.
- [x] 3.3 Implement `MultiHeadAttention::from_weights(weights: &WeightSet, prefix: &str) -> Result<Self, WeightError>`.
- [x] 3.4 Unit test: verify attention output shape is `(seq_len, d_model)` for known input dimensions.

## 4. Feed-Forward Network (thresh-inference)

- [x] 4.1 Define `FeedForward` struct with `w1`, `b1`, `w2`, `b2` weight matrices/biases.
- [x] 4.2 Implement `FeedForward::forward(x: &DMatrix<f32>) -> DMatrix<f32>`: linear1 -> ReLU -> linear2.
- [x] 4.3 Implement `FeedForward::from_weights(weights, prefix)`.
- [x] 4.4 Unit test: verify FFN output shape matches input shape (d_model).

## 5. Layer Normalization (thresh-inference)

- [x] 5.1 Layer norm implemented as a free function `layer_norm(x, gamma, beta, eps)` with per-layer gamma/beta stored in `DecoderLayer`.
- [x] 5.2 Implemented: normalize each row to zero mean/unit variance, apply scale and bias.
- [x] 5.3 Gamma/beta loaded via `DecoderLayer::from_weights`.

## 6. Positional Encoding (thresh-inference)

- [x] 6.1 ~~Implement `SinusoidalPositionalEncoding::new(max_len, embed_dim) -> Self`.~~ Not needed — using learned query embeddings per design.
- [x] 6.2 ~~Implement `SinusoidalPositionalEncoding::forward`.~~ Not needed — using learned query embeddings per design.

## 7. Transformer Decoder (thresh-inference)

- [x] 7.1 Define `DecoderLayer` struct composing: self-attention, FFN, two layer norms (gamma/beta). Implement forward pass with pre-norm residual connections.
- [x] 7.2 Decoder layers stored as `Vec<DecoderLayer>` in `NativeDetector`, chained sequentially in `detect()`.
- [x] 7.3 Implement `DecoderLayer::from_weights(weights, prefix)` that loads weights using indexed prefixes (`layer.0.`, `layer.1.`, etc.).

## 8. Detection Head (thresh-inference)

- [x] 8.1 Define `DetectionHead` struct with `class_proj` (d_model -> n_classes) and `box_proj` (d_model -> 7 for [x,y,z,l,w,h,yaw]).
- [x] 8.2 Implement `DetectionHead::forward(x: &DMatrix<f32>) -> Vec<Detection3D>` — sigmoid for class confidence, direct box regression.
- [x] 8.3 Implement `DetectionHead::from_weights(weights, prefix)`.

## 9. NativeDetector Top-Level (thresh-inference)

- [x] 9.1 Define `NativeDetector` struct composing: `input_proj`, `query_embed`, `Vec<DecoderLayer>`, `DetectionHead`, `NativeDetectorConfig`.
- [x] 9.2 Implement `NativeDetector::from_safetensors(config, weights, input_dim) -> Result<Self, WeightError>`.
- [x] 9.3 Implement `DetectionPipeline` for `NativeDetector` — full forward pass from `SensorInput` to `Vec<Detection3D>` with confidence filtering and NMS.

## 10. Tests and Validation

- [x] 10.1 ~~Create a reference test: generate random input, compute expected output in PyTorch, save input/output as test fixtures.~~ **Deferred** — requires PyTorch fixture generation; planned for when a trained model is available.
- [x] 10.2 Unit tests: relu, softmax_rows, layer_norm, MHA shape, FFN shape, decoder layer shape, detection head outputs, end-to-end forward pass, pipeline trait, config defaults, empty input (11 tests total).
- [x] 10.3 Unit test: `NativeDetector::from_safetensors` fails gracefully with missing or corrupted weight files. Implemented `test_from_safetensors_fails_gracefully_missing_file`.
- [x] 10.4 Benchmark: measure inference latency for the default architecture on CPU. Implemented `benchmark_native_detector_inference_latency` (small model, 10 iterations, prints per-inference timing via `eprintln`).
