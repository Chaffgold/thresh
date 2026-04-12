# SafeTensors Model Weight Loading

## What

Add support for loading neural network weights from SafeTensors format (.safetensors files) into thresh's detection pipeline, enabling weight-only distribution and hot-swapping without re-exporting ONNX. Introduce a `WeightLoader` trait for format abstraction and a `SafeTensorsLoader` implementation that maps named tensors to nalgebra matrices, with shape validation on load.

## Why

SafeTensors is the Hugging Face standard for weight serialization -- safe (no pickle/arbitrary code execution), fast (memory-mapped, zero-copy reads), and a tiny dependency. Currently thresh-inference requires a fully-exported ONNX model for any detection work. Supporting SafeTensors enables a Rust-native model architecture where the computation graph is defined in code and only the weights are loaded from disk. This decouples model architecture evolution from weight distribution: weights can be updated independently, hot-swapped at runtime, and distributed without including the full ONNX graph. It also lays the foundation for the `rust-native-detector` change, which needs a weight loading mechanism.

## How

- Add `safetensors = "0.4"` as an optional dependency to thresh-inference behind a `safetensors` feature gate
- Define a `WeightLoader` trait in thresh-inference with methods for loading named tensors, querying available tensor names, and validating shapes against an expected manifest
- Implement `SafeTensorsLoader` that wraps the `safetensors` crate, providing memory-mapped file access and zero-copy tensor reads
- Define a JSON manifest format that maps tensor names to expected shapes and data types, enabling validation before weights are used
- Add `WeightMetadata` types to thresh-core for representing tensor shape, dtype, and named weight sets
- Map SafeTensors tensors to `nalgebra::DMatrix<f32>` (the target type for all weight matrices)
- Integrate with the detection pipeline via a `load_weights(path)` method on detector structs, allowing weight override of ONNX-embedded weights or standalone use with the native detector

## Out of scope

- Training or fine-tuning weights (inference-only)
- Loading weights from other formats (GGUF, pickle, NumPy .npy) -- the trait enables future formats but only SafeTensors is implemented
- Automatic model architecture inference from weight shapes
- Weight quantization or compression
- GPU memory placement (CPU-only in this iteration)

## Affected crates

- thresh-inference: `WeightLoader` trait, `SafeTensorsLoader` implementation, integration with detector structs, feature gate
- thresh-core: `WeightMetadata` types (tensor shape, dtype, named weight set descriptor)
