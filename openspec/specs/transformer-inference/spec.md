# transformer-inference Specification

## Purpose
TBD - created by archiving change transformer-fusion-tracker. Update Purpose after archive.
## Requirements
### Requirement: ONNX model loading and session management
The system SHALL load ONNX model files using ONNX Runtime (via the `ort` Rust crate), creating inference sessions with configurable execution providers (CPU, CUDA, TensorRT). Each pipeline component (encoder, fusion module, detection head) SHALL be loaded as a separate ONNX session.

#### Scenario: Load modular pipeline components
- **WHEN** the user provides paths to separate ONNX files for a camera encoder, LiDAR encoder, BEV fusion module, and detection head
- **THEN** the system SHALL create independent inference sessions for each component and orchestrate them in the correct execution order

#### Scenario: Execution provider fallback
- **WHEN** TensorRT execution provider is requested but not available on the host
- **THEN** the system SHALL fall back to CUDA, then to CPU, logging a warning about the fallback

### Requirement: Dynamic shape support
The system SHALL support ONNX models with dynamic input dimensions (variable batch size, sequence length, number of points) by specifying dynamic axes and optimization profiles for TensorRT conversion.

#### Scenario: Variable point cloud size
- **WHEN** LiDAR scans contain varying numbers of points between frames (e.g., 20K to 120K points)
- **THEN** the inference session SHALL accept the variable-size input without recompilation or error

#### Scenario: Variable number of detections
- **WHEN** the detection head produces a variable number of output detections per frame
- **THEN** the system SHALL correctly parse the dynamic output tensor dimensions

### Requirement: Pipeline orchestration
The system SHALL orchestrate the modular inference pipeline: sensor-specific encoders → BEV projection (if applicable) → fusion → detection head, passing intermediate tensors between stages. The pipeline SHALL support both BEV-concatenation style (BEVFusion) and query-based style (TransFusion/CMT) architectures.

#### Scenario: BEVFusion-style pipeline execution
- **WHEN** configured with camera encoder, LiDAR encoder, BEV pooling, and convolutional fusion ONNX models
- **THEN** the system SHALL execute them in sequence, passing BEV feature maps between stages, and return 3D bounding box detections

#### Scenario: Query-based pipeline execution
- **WHEN** configured with a backbone encoder and transformer decoder ONNX model using cross-attention
- **THEN** the system SHALL pass feature maps as key-value inputs to the decoder and return detection results from the query outputs

### Requirement: Inference performance monitoring
The system SHALL measure and report per-component inference latency, total pipeline latency, and throughput (FPS). The system SHALL support INT8 and FP16 precision modes where the execution provider allows.

#### Scenario: Latency reporting
- **WHEN** the pipeline processes a frame
- **THEN** the system SHALL record wall-clock time for each ONNX session inference call and the total pipeline time, accessible via an API

#### Scenario: Mixed precision execution
- **WHEN** the user requests FP16 precision for encoder stages and INT8 for the detection head
- **THEN** the system SHALL configure each session with the requested precision and validate that outputs remain within acceptable numerical tolerance

