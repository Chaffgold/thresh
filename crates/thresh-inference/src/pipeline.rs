//! Inference pipeline configuration and orchestration.

use std::time::{Duration, Instant};

/// Precision configuration for a model stage.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Precision {
    Fp32,
    Fp16,
    Int8,
}

/// Execution provider preference.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExecutionProvider {
    Cpu,
    Cuda,
    TensorRt,
}

/// Configuration for a single pipeline stage (one ONNX model).
#[derive(Debug, Clone)]
pub struct StageConfig {
    /// Human-readable stage name.
    pub name: String,
    /// Path to the ONNX model file.
    pub model_path: String,
    /// Preferred precision.
    pub precision: Precision,
    /// Preferred execution provider.
    pub provider: ExecutionProvider,
    /// Dynamic input axes (name -> dim indices that are dynamic).
    pub dynamic_axes: Vec<(String, Vec<usize>)>,
}

/// A BEVFusion-style pipeline configuration.
#[derive(Debug, Clone)]
pub struct BevFusionPipeline {
    pub camera_encoder: StageConfig,
    pub lidar_encoder: StageConfig,
    pub bev_pool: StageConfig,
    pub fusion: StageConfig,
    pub detection_head: StageConfig,
}

impl BevFusionPipeline {
    /// Create a default BEVFusion pipeline config.
    pub fn default_config(model_dir: &str) -> Self {
        Self {
            camera_encoder: StageConfig {
                name: "camera_encoder".into(),
                model_path: format!("{model_dir}/camera_encoder.onnx"),
                precision: Precision::Fp16,
                provider: ExecutionProvider::TensorRt,
                dynamic_axes: vec![("images".into(), vec![0])],
            },
            lidar_encoder: StageConfig {
                name: "lidar_encoder".into(),
                model_path: format!("{model_dir}/lidar_encoder.onnx"),
                precision: Precision::Fp16,
                provider: ExecutionProvider::TensorRt,
                dynamic_axes: vec![("points".into(), vec![0])],
            },
            bev_pool: StageConfig {
                name: "bev_pool".into(),
                model_path: format!("{model_dir}/bev_pool.onnx"),
                precision: Precision::Fp16,
                provider: ExecutionProvider::TensorRt,
                dynamic_axes: vec![],
            },
            fusion: StageConfig {
                name: "fusion".into(),
                model_path: format!("{model_dir}/fusion.onnx"),
                precision: Precision::Fp16,
                provider: ExecutionProvider::TensorRt,
                dynamic_axes: vec![],
            },
            detection_head: StageConfig {
                name: "detection_head".into(),
                model_path: format!("{model_dir}/detection_head.onnx"),
                precision: Precision::Fp32,
                provider: ExecutionProvider::Cuda,
                dynamic_axes: vec![],
            },
        }
    }

    /// Get all stages in execution order.
    pub fn stages(&self) -> Vec<&StageConfig> {
        vec![
            &self.camera_encoder,
            &self.lidar_encoder,
            &self.bev_pool,
            &self.fusion,
            &self.detection_head,
        ]
    }
}

/// A query-based (TransFusion/CMT) pipeline configuration.
#[derive(Debug, Clone)]
pub struct QueryPipeline {
    pub backbone: StageConfig,
    pub transformer_decoder: StageConfig,
    pub detection_head: StageConfig,
}

impl QueryPipeline {
    pub fn default_config(model_dir: &str) -> Self {
        Self {
            backbone: StageConfig {
                name: "backbone".into(),
                model_path: format!("{model_dir}/backbone.onnx"),
                precision: Precision::Fp16,
                provider: ExecutionProvider::TensorRt,
                dynamic_axes: vec![("input".into(), vec![0])],
            },
            transformer_decoder: StageConfig {
                name: "transformer_decoder".into(),
                model_path: format!("{model_dir}/decoder.onnx"),
                precision: Precision::Fp16,
                provider: ExecutionProvider::TensorRt,
                dynamic_axes: vec![],
            },
            detection_head: StageConfig {
                name: "detection_head".into(),
                model_path: format!("{model_dir}/det_head.onnx"),
                precision: Precision::Fp32,
                provider: ExecutionProvider::Cuda,
                dynamic_axes: vec![],
            },
        }
    }
}

/// Latency measurement for a pipeline run.
#[derive(Debug, Clone)]
pub struct PipelineLatency {
    /// Per-stage latencies.
    pub stages: Vec<(String, Duration)>,
    /// Total pipeline latency.
    pub total: Duration,
}

impl PipelineLatency {
    /// Create a new latency tracker.
    pub fn new() -> Self {
        Self {
            stages: Vec::new(),
            total: Duration::ZERO,
        }
    }

    /// Time a stage execution.
    pub fn time_stage<F, R>(&mut self, name: &str, f: F) -> R
    where
        F: FnOnce() -> R,
    {
        let start = Instant::now();
        let result = f();
        let elapsed = start.elapsed();
        self.stages.push((name.to_string(), elapsed));
        self.total += elapsed;
        result
    }
}

impl Default for PipelineLatency {
    fn default() -> Self {
        Self::new()
    }
}

/// Parse raw detection output tensors into BoundingBox3D structs.
pub fn parse_detections(
    boxes: &[f64],
    scores: &[f64],
    classes: &[u32],
    n_detections: usize,
) -> Vec<thresh_core::detection::BoundingBox3D> {
    let mut dets = Vec::with_capacity(n_detections);
    for i in 0..n_detections {
        let base = i * 7; // [x, y, z, l, w, h, yaw]
        if base + 6 >= boxes.len() {
            break;
        }
        dets.push(thresh_core::detection::BoundingBox3D {
            x: boxes[base],
            y: boxes[base + 1],
            z: boxes[base + 2],
            length: boxes[base + 3],
            width: boxes[base + 4],
            height: boxes[base + 5],
            yaw: boxes[base + 6],
            score: scores.get(i).copied().unwrap_or(0.0),
            class_id: classes.get(i).copied().unwrap_or(0),
            velocity: None,
        });
    }
    dets
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bevfusion_pipeline_has_5_stages() {
        let pipeline = BevFusionPipeline::default_config("/models");
        assert_eq!(pipeline.stages().len(), 5);
    }

    #[test]
    fn parse_detection_output() {
        let boxes = vec![
            1.0, 2.0, 3.0, 5.0, 2.0, 1.5, 0.1, 10.0, 20.0, 30.0, 4.0, 2.0, 1.0, 0.5,
        ];
        let scores = vec![0.9, 0.7];
        let classes = vec![0, 1];

        let dets = parse_detections(&boxes, &scores, &classes, 2);
        assert_eq!(dets.len(), 2);
        assert_eq!(dets[0].x, 1.0);
        assert_eq!(dets[0].score, 0.9);
        assert_eq!(dets[1].class_id, 1);
    }

    #[test]
    fn latency_tracking() {
        let mut lat = PipelineLatency::new();
        lat.time_stage("test", || std::thread::sleep(Duration::from_millis(1)));
        assert_eq!(lat.stages.len(), 1);
        assert!(lat.total >= Duration::from_millis(1));
    }
}
