# Rust-Native Detection Head

## What

Implement a lightweight detection head (simplified DETR decoder) directly in Rust so thresh can run inference without ONNX Runtime. The architecture is a simplified DETR decoder with 6 transformer layers, 256-dimensional embeddings, and 8 attention heads. Weights are loaded via the SafeTensors loader from the `safetensors-weights` change.

## Why

The current detection pipeline requires ONNX Runtime, which is a large C++ dependency that complicates builds, limits deployment targets, and is unavailable on some embedded/edge platforms. A pure-Rust detection head removes the ORT dependency for inference, enables embedded and edge deployment where ORT cannot be cross-compiled, and provides a reference implementation that pairs with the SafeTensors weight loader. It also allows the Rust compiler to optimize the entire inference path end-to-end, and makes the detection pipeline testable in CI without any native library dependencies.

## How

- Implement multi-head self-attention in pure Rust using nalgebra: Q/K/V projections, scaled dot-product attention, output projection
- Implement feed-forward network: two linear layers with GELU activation
- Implement layer normalization with learnable scale/bias parameters
- Implement sinusoidal positional encoding for object queries
- Compose these into a `TransformerDecoderLayer` and stack 6 layers into a `TransformerDecoder`
- Implement the detection head: linear projection from decoder output to class logits and bounding box coordinates
- Implement the full forward pass: positional encoding + decoder layers + detection head
- Load all weights from SafeTensors format using the `WeightLoader` trait
- Feature-gate behind `native-detector` to keep it optional

## Out of scope

- Training or backpropagation (inference-only, no autograd)
- The encoder half of DETR (assumes pre-computed encoder features or operates on raw queries)
- f16/bf16 precision (f32 only in this iteration)
- Batch sizes greater than 1
- GPU acceleration (CPU-only)
- Non-maximum suppression (handled downstream by thresh-tracker)

## Affected crates

- thresh-inference: native detector module (`native/`), transformer components, detection head, feature gate
- thresh-core: tensor operation helpers (softmax, GELU, layer norm utilities)
