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

    // --- Task 6.10: Pipeline configuration tests (no onnx feature required) ---

    #[test]
    fn bevfusion_stage_order() {
        let pipeline = BevFusionPipeline::default_config("/models");
        let names: Vec<&str> = pipeline.stages().iter().map(|s| s.name.as_str()).collect();
        assert_eq!(
            names,
            vec![
                "camera_encoder",
                "lidar_encoder",
                "bev_pool",
                "fusion",
                "detection_head"
            ]
        );
    }

    #[test]
    fn bevfusion_model_paths_use_dir_prefix() {
        let pipeline = BevFusionPipeline::default_config("/opt/models");
        for stage in pipeline.stages() {
            assert!(
                stage.model_path.starts_with("/opt/models/"),
                "stage '{}' model_path should start with the model dir",
                stage.name
            );
            assert!(
                stage.model_path.ends_with(".onnx"),
                "stage '{}' model_path should end with .onnx",
                stage.name
            );
        }
    }

    #[test]
    fn bevfusion_detection_head_uses_fp32_cuda() {
        let pipeline = BevFusionPipeline::default_config("/m");
        assert_eq!(pipeline.detection_head.precision, Precision::Fp32);
        assert_eq!(pipeline.detection_head.provider, ExecutionProvider::Cuda);
    }

    #[test]
    fn query_pipeline_has_3_stages() {
        let pipeline = QueryPipeline::default_config("/models");
        assert_eq!(pipeline.backbone.name, "backbone");
        assert_eq!(pipeline.transformer_decoder.name, "transformer_decoder");
        assert_eq!(pipeline.detection_head.name, "detection_head");
    }

    #[test]
    fn query_pipeline_model_paths() {
        let pipeline = QueryPipeline::default_config("/data/models");
        assert_eq!(pipeline.backbone.model_path, "/data/models/backbone.onnx");
        assert_eq!(
            pipeline.transformer_decoder.model_path,
            "/data/models/decoder.onnx"
        );
        assert_eq!(
            pipeline.detection_head.model_path,
            "/data/models/det_head.onnx"
        );
    }

    #[test]
    fn query_pipeline_detection_head_uses_fp32_cuda() {
        let pipeline = QueryPipeline::default_config("/m");
        assert_eq!(pipeline.detection_head.precision, Precision::Fp32);
        assert_eq!(pipeline.detection_head.provider, ExecutionProvider::Cuda);
    }

    #[test]
    fn parse_detections_handles_zero() {
        let dets = parse_detections(&[], &[], &[], 0);
        assert!(dets.is_empty());
    }

    #[test]
    fn parse_detections_truncates_when_boxes_too_short() {
        // Request 3 detections but only provide enough box data for 1
        let boxes = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
        let scores = vec![0.95, 0.8, 0.5];
        let classes = vec![0, 1, 2];

        let dets = parse_detections(&boxes, &scores, &classes, 3);
        assert_eq!(dets.len(), 1);
        assert_eq!(dets[0].score, 0.95);
        assert_eq!(dets[0].class_id, 0);
    }

    #[test]
    fn parse_detections_missing_scores_defaults_to_zero() {
        let boxes = vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0];
        let scores: Vec<f64> = vec![];
        let classes: Vec<u32> = vec![];

        let dets = parse_detections(&boxes, &scores, &classes, 1);
        assert_eq!(dets.len(), 1);
        assert_eq!(dets[0].score, 0.0);
        assert_eq!(dets[0].class_id, 0);
    }

    #[test]
    fn pipeline_latency_default() {
        let lat = PipelineLatency::default();
        assert!(lat.stages.is_empty());
        assert_eq!(lat.total, Duration::ZERO);
    }

    #[test]
    fn pipeline_latency_multiple_stages() {
        let mut lat = PipelineLatency::new();
        let _r1 = lat.time_stage("stage_a", || 42);
        let _r2 = lat.time_stage("stage_b", || "hello");
        assert_eq!(lat.stages.len(), 2);
        assert_eq!(lat.stages[0].0, "stage_a");
        assert_eq!(lat.stages[1].0, "stage_b");
        assert!(lat.total >= Duration::ZERO);
    }

    #[test]
    fn pipeline_latency_returns_value() {
        let mut lat = PipelineLatency::new();
        let result = lat.time_stage("compute", || 2 + 2);
        assert_eq!(result, 4);
    }

    // --- Task 6.11: Dynamic shape / axes configuration tests ---

    #[test]
    fn stage_config_stores_dynamic_axes() {
        let config = StageConfig {
            name: "test_stage".into(),
            model_path: "/tmp/test.onnx".into(),
            precision: Precision::Fp16,
            provider: ExecutionProvider::TensorRt,
            dynamic_axes: vec![("input".into(), vec![0, 2, 3]), ("output".into(), vec![0])],
        };
        assert_eq!(config.dynamic_axes.len(), 2);
        assert_eq!(config.dynamic_axes[0].0, "input");
        assert_eq!(config.dynamic_axes[0].1, vec![0, 2, 3]);
        assert_eq!(config.dynamic_axes[1].0, "output");
        assert_eq!(config.dynamic_axes[1].1, vec![0]);
    }

    #[test]
    fn stage_config_empty_dynamic_axes() {
        let config = StageConfig {
            name: "static_stage".into(),
            model_path: "/tmp/static.onnx".into(),
            precision: Precision::Int8,
            provider: ExecutionProvider::Cpu,
            dynamic_axes: vec![],
        };
        assert!(config.dynamic_axes.is_empty());
    }

    #[test]
    fn bevfusion_camera_encoder_has_batch_dynamic_axis() {
        let pipeline = BevFusionPipeline::default_config("/m");
        let cam = &pipeline.camera_encoder;
        assert_eq!(cam.dynamic_axes.len(), 1);
        let (name, dims) = &cam.dynamic_axes[0];
        assert_eq!(name, "images");
        // Dim 0 is the batch dimension
        assert!(dims.contains(&0));
    }

    #[test]
    fn bevfusion_lidar_encoder_has_batch_dynamic_axis() {
        let pipeline = BevFusionPipeline::default_config("/m");
        let lidar = &pipeline.lidar_encoder;
        assert_eq!(lidar.dynamic_axes.len(), 1);
        let (name, dims) = &lidar.dynamic_axes[0];
        assert_eq!(name, "points");
        assert!(dims.contains(&0));
    }

    #[test]
    fn bevfusion_static_stages_have_no_dynamic_axes() {
        let pipeline = BevFusionPipeline::default_config("/m");
        assert!(pipeline.bev_pool.dynamic_axes.is_empty());
        assert!(pipeline.fusion.dynamic_axes.is_empty());
        assert!(pipeline.detection_head.dynamic_axes.is_empty());
    }

    #[test]
    fn query_pipeline_backbone_has_batch_dynamic_axis() {
        let pipeline = QueryPipeline::default_config("/m");
        let backbone = &pipeline.backbone;
        assert_eq!(backbone.dynamic_axes.len(), 1);
        let (name, dims) = &backbone.dynamic_axes[0];
        assert_eq!(name, "input");
        assert!(dims.contains(&0));
    }

    #[test]
    fn query_pipeline_static_stages_have_no_dynamic_axes() {
        let pipeline = QueryPipeline::default_config("/m");
        assert!(pipeline.transformer_decoder.dynamic_axes.is_empty());
        assert!(pipeline.detection_head.dynamic_axes.is_empty());
    }

    #[test]
    fn stage_config_clone_preserves_dynamic_axes() {
        let config = StageConfig {
            name: "test".into(),
            model_path: "/test.onnx".into(),
            precision: Precision::Fp16,
            provider: ExecutionProvider::Cuda,
            dynamic_axes: vec![("x".into(), vec![0, 1]), ("y".into(), vec![0])],
        };
        let cloned = config.clone();
        assert_eq!(cloned.dynamic_axes.len(), 2);
        assert_eq!(cloned.dynamic_axes[0].0, "x");
        assert_eq!(cloned.dynamic_axes[0].1, config.dynamic_axes[0].1);
        assert_eq!(cloned.dynamic_axes[1].0, "y");
    }

    #[test]
    fn precision_and_provider_are_copy() {
        let p = Precision::Fp16;
        let p2 = p;
        assert_eq!(p, p2);

        let ep = ExecutionProvider::TensorRt;
        let ep2 = ep;
        assert_eq!(ep, ep2);
    }

    // --- Task 6.10 ONNX-gated tests ---

    #[cfg(feature = "onnx")]
    mod onnx_tests {
        use super::*;

        #[test]
        fn session_builder_errors_on_missing_model() {
            // Verify that attempting to create a session from a nonexistent
            // model file returns an error rather than panicking.
            let result = ort::Session::builder()
                .and_then(|builder| builder.commit_from_file("/nonexistent/model.onnx"));
            assert!(
                result.is_err(),
                "Loading a nonexistent model should return an error"
            );
        }

        #[test]
        fn session_builder_errors_on_invalid_data() {
            // Verify that loading garbage bytes as an ONNX model is an error.
            let garbage = vec![0u8; 64];
            let result =
                ort::Session::builder().and_then(|builder| builder.commit_from_memory(&garbage));
            assert!(
                result.is_err(),
                "Loading invalid model data should return an error"
            );
        }

        // Task 6.11 ONNX-gated: dynamic batch size configuration
        #[test]
        fn dynamic_axes_config_for_batch_dimension() {
            // Verify that the dynamic_axes field correctly represents
            // batch-dynamic inputs across pipeline stages.
            let pipeline = BevFusionPipeline::default_config("/models");

            // Camera encoder: "images" input has dynamic batch (dim 0)
            let cam_axes = &pipeline.camera_encoder.dynamic_axes;
            assert_eq!(cam_axes.len(), 1);
            assert_eq!(cam_axes[0].0, "images");
            assert_eq!(
                cam_axes[0].1,
                vec![0],
                "Batch dimension (0) should be dynamic for camera images"
            );

            // Static stages should have no dynamic axes
            assert!(
                pipeline.bev_pool.dynamic_axes.is_empty(),
                "bev_pool should have fixed shapes"
            );
        }
    }
}
