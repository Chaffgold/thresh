# Transformer-Based Detection Pipeline

## What

Wire a pre-trained transformer detection model (DETR or RT-DETR architecture) into thresh-inference so radar and LiDAR point clouds produce detection bounding boxes via neural inference, which then feed into the classical Kalman tracker. This completes the "transformer" half of the hybrid transformer-plus-Bayesian architecture that thresh is designed around.

## Why

thresh-inference currently has ONNX Runtime plumbing (session management, tensor I/O, provider selection) but no actual model integration or detection pipeline. The project's stated goal is a hybrid architecture combining transformer-based detection with classical Bayesian state estimation. Without a working detection head, the inference crate is scaffolding with no function. Implementing the detection pipeline connects the neural frontend to the Kalman backend and makes the hybrid architecture real.

## How

- Define a `DetectionHead` trait in thresh-inference with `detect(&self, input: &SensorFrame) -> Vec<Detection>` interface
- Implement `OnnxDetrHead` that loads a DETR/RT-DETR ONNX model, runs pre-processing (feature extraction, normalization), inference, and post-processing (confidence thresholding, NMS)
- Add detection output types to thresh-core: `Detection { bbox, confidence, class_id, embedding }` with conversions to measurement vectors for the tracker
- Wire the detection output into thresh-tracker's existing detection input path via an adapter that converts bounding boxes to state-space measurements
- Ship a small pre-trained RT-DETR model (or test stub) for CI testing on CPU; gate the real model behind a feature flag

## Out of scope

- Model training or fine-tuning pipelines
- Custom transformer architectures beyond standard DETR/RT-DETR
- Real-time video input or camera frame capture
- GPU-only inference paths (must support CPU fallback for CI)
- End-to-end learned tracking (detection only, tracking remains classical)

## Affected crates

- thresh-inference: detection head trait, ONNX DETR implementation, pre/post-processing pipeline
- thresh-tracker: detection input adapter converting bounding box detections to filter measurements
- thresh-core: detection types, bounding box representation, detection-to-measurement conversion
