## Capability: SafeTensors Weight Loading

### Overview

The SafeTensors weight loading capability provides a trait-based interface for loading neural network weights from SafeTensors files into nalgebra matrices, with manifest-driven shape validation and memory-mapped I/O for efficient access.

## ADDED Requirements

### Requirement: WeightLoader trait abstraction

The system MUST provide a `WeightLoader` trait that abstracts weight loading across serialization formats, with methods for loading named tensors as `DMatrix<f32>`, querying available tensor names, and validating tensor shapes against an expected manifest.

#### Scenario: Loading a named weight tensor

**WHEN** a caller requests a tensor by name from a SafeTensors file via `WeightLoader::load_tensor(name)`

**THEN** the loader reads the tensor data using memory-mapped I/O without copying the entire file into memory

**SHALL** return a `DMatrix<f32>` with the correct shape matching the tensor's stored dimensions

### Requirement: Manifest-based shape validation

The system MUST validate loaded tensor shapes against a JSON manifest that specifies expected tensor names, shapes, and data types before weights are used in the detection pipeline.

#### Scenario: Shape mismatch detection on load

**WHEN** a SafeTensors file contains a tensor whose shape does not match the manifest's expected shape for that tensor name

**THEN** the loader MUST return an error identifying the tensor name, expected shape, and actual shape

**SHALL** not partially load weights when any tensor fails validation

### Requirement: Integration with detector structs

The system MUST provide a `load_weights(path)` method on detector structs that loads and validates a complete weight set from a SafeTensors file, replacing any previously loaded weights.

#### Scenario: Hot-swapping weights on a running detector

**WHEN** `load_weights(path)` is called with a new SafeTensors file path on an existing detector instance

**THEN** the detector validates all weight shapes against its architecture manifest and replaces its internal weight matrices

**SHALL** complete the weight swap atomically -- either all weights are replaced or none are, preserving the previous weights on failure
