## Capability: Transformer-Based Detection

### Overview

The detection pipeline loads a pre-trained transformer model via ONNX Runtime and produces 3D bounding-box detections with class labels and confidence scores from sensor input tensors.

## ADDED Requirements

### Requirement: ONNX-based detection pipeline

The system MUST provide a detection pipeline that loads a pre-trained ONNX model and produces 3D bounding-box detections from sensor input tensors.

#### Scenario: Radar point cloud detection

**WHEN** a radar point cloud is passed through the detection pipeline

**THEN** the pipeline executes the ONNX model and decodes output tensors

**SHALL** produce detection bounding boxes with class labels and confidence scores
