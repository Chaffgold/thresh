## 1. Module Structure (thresh-inference)

- [ ] 1.1 Create `crates/thresh-inference/src/native/mod.rs` with `#[cfg(feature = "native-detector")]` gate. Add `native-detector` feature to Cargo.toml with dependency on `safetensors` feature.
- [ ] 1.2 Create submodule files: `attention.rs`, `ffn.rs`, `layer_norm.rs`, `positional.rs`, `decoder.rs`, `detection_head.rs`.
- [ ] 1.3 Define `NativeDetectorConfig` struct with fields: `num_layers`, `embed_dim`, `num_heads`, `ffn_dim`, `num_queries`, `num_classes`. Add defaults (6, 256, 8, 1024, 100, configurable).

## 2. Tensor Ops Helpers (thresh-core)

- [ ] 2.1 Add `crates/thresh-core/src/tensor_ops.rs` module with `softmax(v: &DVector<f32>) -> DVector<f32>` using log-sum-exp for numerical stability.
- [ ] 2.2 Implement `gelu(x: f32) -> f32` using the tanh approximation: `0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))`.
- [ ] 2.3 Implement `layer_norm(x: &DVector<f32>, gamma: &DVector<f32>, beta: &DVector<f32>, eps: f32) -> DVector<f32>`.
- [ ] 2.4 Implement `softmax_2d(m: &DMatrix<f32>) -> DMatrix<f32>` that applies softmax along each row (for attention weights).
- [ ] 2.5 Re-export from `thresh-core/src/lib.rs`. Add unit tests for each function against known reference values.

## 3. Multi-Head Self-Attention (thresh-inference)

- [ ] 3.1 Define `MultiHeadAttention` struct with weight matrices: `w_q`, `w_k`, `w_v`, `w_o` (all `DMatrix<f32>`), bias vectors: `b_q`, `b_k`, `b_v`, `b_o`, and config: `num_heads`, `head_dim`.
- [ ] 3.2 Implement `MultiHeadAttention::forward(x: &DMatrix<f32>) -> DMatrix<f32>`: compute Q, K, V projections, split into heads, scaled dot-product attention, concatenate, output projection.
- [ ] 3.3 Implement `MultiHeadAttention::load_weights(loader: &dyn WeightLoader, prefix: &str) -> Result<Self>` that loads all 8 weight/bias tensors using the layer name prefix.
- [ ] 3.4 Unit test: verify attention output shape is `(seq_len, embed_dim)` for known input dimensions.

## 4. Feed-Forward Network (thresh-inference)

- [ ] 4.1 Define `FeedForwardNetwork` struct with `w1`, `b1`, `w2`, `b2` weight matrices/biases and `ffn_dim` config.
- [ ] 4.2 Implement `FeedForwardNetwork::forward(x: &DMatrix<f32>) -> DMatrix<f32>`: linear1 -> GELU -> linear2.
- [ ] 4.3 Implement `FeedForwardNetwork::load_weights(loader, prefix)`.
- [ ] 4.4 Unit test: verify FFN output shape matches input shape (embed_dim).

## 5. Layer Normalization (thresh-inference)

- [ ] 5.1 Define `LayerNorm` struct with `gamma` (scale) and `beta` (bias) vectors and `eps` constant.
- [ ] 5.2 Implement `LayerNorm::forward(x: &DMatrix<f32>) -> DMatrix<f32>`: normalize each row, apply scale and bias.
- [ ] 5.3 Implement `LayerNorm::load_weights(loader, prefix)`.

## 6. Positional Encoding (thresh-inference)

- [ ] 6.1 Implement `SinusoidalPositionalEncoding::new(max_len, embed_dim) -> Self` that precomputes the sinusoidal encoding matrix.
- [ ] 6.2 Implement `SinusoidalPositionalEncoding::forward(x: &DMatrix<f32>) -> DMatrix<f32>` that adds the positional encoding to the input.

## 7. Transformer Decoder (thresh-inference)

- [ ] 7.1 Define `TransformerDecoderLayer` struct composing: self-attention, FFN, two layer norms. Implement forward pass with residual connections (pre-norm: norm -> attention -> add, norm -> FFN -> add).
- [ ] 7.2 Define `TransformerDecoder` struct holding `Vec<TransformerDecoderLayer>`. Implement forward pass that chains layers sequentially.
- [ ] 7.3 Implement `TransformerDecoder::load_weights(loader, num_layers)` that loads weights for all layers using indexed prefixes (`decoder.layer.0.`, `decoder.layer.1.`, etc.).

## 8. Detection Head (thresh-inference)

- [ ] 8.1 Define `DetectionHead` struct with `class_proj` (linear: embed_dim -> num_classes) and `bbox_proj` (linear: embed_dim -> 4).
- [ ] 8.2 Implement `DetectionHead::forward(x: &DMatrix<f32>) -> (DMatrix<f32>, DMatrix<f32>)` returning (class_logits, bbox_coords).
- [ ] 8.3 Implement `DetectionHead::load_weights(loader)`.

## 9. NativeDetector Top-Level (thresh-inference)

- [ ] 9.1 Define `NativeDetector` struct composing: `SinusoidalPositionalEncoding`, `TransformerDecoder`, `DetectionHead`, `NativeDetectorConfig`.
- [ ] 9.2 Implement `NativeDetector::from_safetensors(path, config) -> Result<Self>` that loads all weights and constructs the full pipeline.
- [ ] 9.3 Implement `NativeDetector::detect(queries: &DMatrix<f32>) -> Result<Vec<Detection>>` that runs the full forward pass and converts outputs to thresh `Detection` types.

## 10. Tests and Validation

- [ ] 10.1 Create a reference test: generate random input, compute expected output in PyTorch, save input/output as test fixtures. Verify Rust output matches within 1e-5 tolerance.
- [ ] 10.2 Unit test: end-to-end forward pass with random weights produces correctly-shaped outputs.
- [ ] 10.3 Unit test: `NativeDetector::from_safetensors` fails gracefully with missing or corrupted weight files.
- [ ] 10.4 Benchmark: measure inference latency for the default architecture (6 layers, 256-dim, 100 queries) on CPU. Document results.
