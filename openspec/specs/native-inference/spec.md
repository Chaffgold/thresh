# native-inference Specification

## Purpose
TBD - created by archiving change rust-native-detector. Update Purpose after archive.
## Requirements
### Requirement: Multi-head self-attention

The system MUST implement multi-head scaled dot-product self-attention with configurable head count and embedding dimension, using nalgebra matrices for all linear algebra operations.

#### Scenario: Attention computation with 8 heads

**WHEN** the attention module receives input queries of dimension 256 with 8 attention heads

**THEN** it computes Q, K, V projections, splits into 8 heads of dimension 32 each, applies scaled dot-product attention with scaling factor `1/sqrt(32)`, concatenates heads, and applies the output projection

**SHALL** produce output of the same dimension (256) as the input queries

### Requirement: Complete forward pass

The system MUST implement a complete forward pass through the decoder stack that accepts object queries and produces detection outputs (class logits and bounding box coordinates).

#### Scenario: Detection inference on object queries

**WHEN** the native detector receives N object queries (e.g., 100 queries of dimension 256)

**THEN** it processes them through 6 decoder layers (self-attention + feed-forward + layer norm) with sinusoidal positional encoding, followed by the detection head

**SHALL** produce N detection outputs, each containing class logits and a 4-element bounding box coordinate vector

### Requirement: Weight loading from SafeTensors

The system MUST load all model parameters from a SafeTensors file using the `WeightLoader` trait, mapping named tensors to the correct layers in the architecture.

#### Scenario: Initializing detector from SafeTensors weights

**WHEN** a `NativeDetector` is constructed with a path to a SafeTensors weight file

**THEN** it loads all required weight tensors (attention projections, feed-forward weights, layer norm parameters, detection head weights), validates their shapes against the architecture specification

**SHALL** be ready for inference immediately after construction without additional initialization steps

### Requirement: Feature-gated compilation

The system MUST gate the native detector behind a `native-detector` feature flag so it does not increase compile time or binary size for users who do not need it.

#### Scenario: Building without native detector

**WHEN** thresh-inference is compiled without the `native-detector` feature flag

**THEN** the native detector module and its dependencies MUST be excluded from compilation

**SHALL** not affect the availability or behavior of the ONNX-based detector

