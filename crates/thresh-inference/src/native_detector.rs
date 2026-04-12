//! Pure-Rust DETR-style detection decoder using nalgebra.
//!
//! Implements a simplified transformer decoder and detection head that runs
//! entirely on CPU with no ONNX Runtime dependency. All matrix math uses
//! `nalgebra::DMatrix<f32>` / `DVector<f32>`.
//!
//! The model architecture:
//! 1. Project input features to `d_model` dimensions
//! 2. Add learned object query embeddings
//! 3. Pass through N transformer decoder layers (self-attention + FFN)
//! 4. Detection head produces class confidences + 3D bounding boxes
//! 5. Filter by confidence threshold and apply NMS

use nalgebra::{DMatrix, DVector};
use thresh_core::detection::Detection3D;

use crate::detection::{DetectionPipeline, SensorInput, filter_by_confidence, nms_3d};
use crate::weights::{WeightError, WeightSet};

// ---------------------------------------------------------------------------
// Pseudo-random matrix/vector initialization (no rand dependency)
// ---------------------------------------------------------------------------

/// Simple pseudo-random f32 generator using a linear congruential approach.
/// Not cryptographically secure — only for test weight initialization.
fn pseudo_random_matrix(rows: usize, cols: usize, seed: u64) -> DMatrix<f32> {
    let mut state = seed.wrapping_add(1);
    DMatrix::from_fn(rows, cols, |i, j| {
        // Mix row, col, and state for variety
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        let mixed =
            state ^ ((i as u64).wrapping_mul(2654435761)) ^ ((j as u64).wrapping_mul(40503));
        // Map to [-1, 1]
        ((mixed >> 33) as f32) / (u32::MAX as f32) * 2.0 - 1.0
    })
}

/// Pseudo-random vector initialization.
fn pseudo_random_vector(len: usize, seed: u64) -> DVector<f32> {
    let mat = pseudo_random_matrix(len, 1, seed);
    DVector::from_column_slice(mat.as_slice())
}

// ---------------------------------------------------------------------------
// Tensor operation helpers
// ---------------------------------------------------------------------------

/// Matrix multiplication wrapper for clarity.
fn matmul(a: &DMatrix<f32>, b: &DMatrix<f32>) -> DMatrix<f32> {
    a * b
}

/// ReLU activation applied element-wise.
fn relu(m: &DMatrix<f32>) -> DMatrix<f32> {
    m.map(|x| x.max(0.0))
}

/// Softmax along each row (last dimension).
///
/// Uses the log-sum-exp trick for numerical stability: subtract the row max
/// before exponentiating.
fn softmax_rows(m: &DMatrix<f32>) -> DMatrix<f32> {
    let mut result = m.clone();
    for i in 0..result.nrows() {
        let row = m.row(i);
        let max_val = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0f32;
        for j in 0..result.ncols() {
            let val = (row[j] - max_val).exp();
            result[(i, j)] = val;
            sum += val;
        }
        for j in 0..result.ncols() {
            result[(i, j)] /= sum;
        }
    }
    result
}

/// Layer normalization: normalize each row to zero mean, unit variance,
/// then scale + shift by learned gamma/beta.
fn layer_norm(
    x: &DMatrix<f32>,
    gamma: &DVector<f32>,
    beta: &DVector<f32>,
    eps: f32,
) -> DMatrix<f32> {
    let mut result = DMatrix::zeros(x.nrows(), x.ncols());
    let d = x.ncols() as f32;
    for i in 0..x.nrows() {
        // Compute mean
        let mut mean = 0.0f32;
        for j in 0..x.ncols() {
            mean += x[(i, j)];
        }
        mean /= d;

        // Compute variance
        let mut var = 0.0f32;
        for j in 0..x.ncols() {
            let diff = x[(i, j)] - mean;
            var += diff * diff;
        }
        var /= d;

        // Normalize, scale, shift
        let inv_std = 1.0 / (var + eps).sqrt();
        for j in 0..x.ncols() {
            result[(i, j)] = (x[(i, j)] - mean) * inv_std * gamma[j] + beta[j];
        }
    }
    result
}

/// Sigmoid activation applied element-wise.
fn sigmoid(m: &DMatrix<f32>) -> DMatrix<f32> {
    m.map(|x| 1.0 / (1.0 + (-x).exp()))
}

/// Add a bias vector (row-wise) to each row of a matrix.
fn add_bias_rows(m: &DMatrix<f32>, bias: &DVector<f32>) -> DMatrix<f32> {
    let mut result = m.clone();
    for i in 0..result.nrows() {
        for j in 0..result.ncols() {
            result[(i, j)] += bias[j];
        }
    }
    result
}

// ---------------------------------------------------------------------------
// Multi-Head Self-Attention
// ---------------------------------------------------------------------------

/// Multi-head self-attention layer.
pub struct MultiHeadAttention {
    n_heads: usize,
    #[allow(dead_code)]
    d_model: usize,
    d_head: usize,
    w_q: DMatrix<f32>,
    w_k: DMatrix<f32>,
    w_v: DMatrix<f32>,
    w_o: DMatrix<f32>,
}

impl MultiHeadAttention {
    /// Create with random weights (for testing).
    pub fn new(d_model: usize, n_heads: usize) -> Self {
        assert!(
            d_model.is_multiple_of(n_heads),
            "d_model must be divisible by n_heads"
        );
        let d_head = d_model / n_heads;
        let scale = (1.0 / d_model as f32).sqrt();
        Self {
            n_heads,
            d_model,
            d_head,
            w_q: pseudo_random_matrix(d_model, d_model, 100) * scale,
            w_k: pseudo_random_matrix(d_model, d_model, 200) * scale,
            w_v: pseudo_random_matrix(d_model, d_model, 300) * scale,
            w_o: pseudo_random_matrix(d_model, d_model, 400) * scale,
        }
    }

    /// Load from a WeightSet with a given prefix.
    ///
    /// Expected keys: `{prefix}.w_q`, `{prefix}.w_k`, `{prefix}.w_v`, `{prefix}.w_o`
    pub fn from_weights(weights: &WeightSet, prefix: &str) -> Result<Self, WeightError> {
        let w_q = weights.get_or_err(&format!("{prefix}.w_q"))?.clone();
        let w_k = weights.get_or_err(&format!("{prefix}.w_k"))?.clone();
        let w_v = weights.get_or_err(&format!("{prefix}.w_v"))?.clone();
        let w_o = weights.get_or_err(&format!("{prefix}.w_o"))?.clone();

        let d_model = w_q.nrows();
        let n_heads = if d_model >= 8 { 8 } else { 1 };
        let d_head = d_model / n_heads;

        Ok(Self {
            n_heads,
            d_model,
            d_head,
            w_q,
            w_k,
            w_v,
            w_o,
        })
    }

    /// Forward pass: x is (seq_len, d_model) -> output is (seq_len, d_model).
    pub fn forward(&self, x: &DMatrix<f32>) -> DMatrix<f32> {
        let seq_len = x.nrows();

        // Project to Q, K, V: each is (seq_len, d_model)
        let q = matmul(x, &self.w_q);
        let k = matmul(x, &self.w_k);
        let v = matmul(x, &self.w_v);

        // Process each head independently and collect outputs
        let scale = 1.0 / (self.d_head as f32).sqrt();
        let mut head_outputs = DMatrix::zeros(seq_len, self.n_heads * self.d_head);

        for h in 0..self.n_heads {
            let col_start = h * self.d_head;

            // Extract head slices: (seq_len, d_head)
            let q_h = q.columns(col_start, self.d_head);
            let k_h = k.columns(col_start, self.d_head);
            let v_h = v.columns(col_start, self.d_head);

            // Attention scores: (seq_len, seq_len)
            let scores = q_h * k_h.transpose() * scale;

            // Softmax over keys
            let attn = softmax_rows(&scores);

            // Weighted values: (seq_len, d_head)
            let out_h = attn * v_h;

            // Copy into combined output
            for i in 0..seq_len {
                for j in 0..self.d_head {
                    head_outputs[(i, col_start + j)] = out_h[(i, j)];
                }
            }
        }

        // Output projection
        matmul(&head_outputs, &self.w_o)
    }
}

// ---------------------------------------------------------------------------
// Feed-Forward Network
// ---------------------------------------------------------------------------

/// Two-layer feed-forward network with ReLU activation.
pub struct FeedForward {
    w1: DMatrix<f32>,
    b1: DVector<f32>,
    w2: DMatrix<f32>,
    b2: DVector<f32>,
}

impl FeedForward {
    /// Create with random weights (for testing).
    pub fn new(d_model: usize, d_ff: usize) -> Self {
        let scale = (1.0 / d_model as f32).sqrt();
        Self {
            w1: pseudo_random_matrix(d_model, d_ff, 500) * scale,
            b1: pseudo_random_vector(d_ff, 600) * scale,
            w2: pseudo_random_matrix(d_ff, d_model, 700) * scale,
            b2: pseudo_random_vector(d_model, 800) * scale,
        }
    }

    /// Load from a WeightSet with a given prefix.
    ///
    /// Expected keys: `{prefix}.w1`, `{prefix}.b1`, `{prefix}.w2`, `{prefix}.b2`
    pub fn from_weights(weights: &WeightSet, prefix: &str) -> Result<Self, WeightError> {
        let w1 = weights.get_or_err(&format!("{prefix}.w1"))?.clone();
        let b1_mat = weights.get_or_err(&format!("{prefix}.b1"))?;
        let w2 = weights.get_or_err(&format!("{prefix}.w2"))?.clone();
        let b2_mat = weights.get_or_err(&format!("{prefix}.b2"))?;

        let b1 = DVector::from_column_slice(b1_mat.as_slice());
        let b2 = DVector::from_column_slice(b2_mat.as_slice());

        Ok(Self { w1, b1, w2, b2 })
    }

    /// Forward pass: FFN(x) = ReLU(x * W1 + b1) * W2 + b2
    ///
    /// x is (seq_len, d_model) -> output is (seq_len, d_model)
    pub fn forward(&self, x: &DMatrix<f32>) -> DMatrix<f32> {
        let hidden = relu(&add_bias_rows(&matmul(x, &self.w1), &self.b1));
        add_bias_rows(&matmul(&hidden, &self.w2), &self.b2)
    }
}

// ---------------------------------------------------------------------------
// Transformer Decoder Layer
// ---------------------------------------------------------------------------

/// Single transformer decoder layer with pre-norm architecture.
pub struct DecoderLayer {
    self_attn: MultiHeadAttention,
    ffn: FeedForward,
    norm1_gamma: DVector<f32>,
    norm1_beta: DVector<f32>,
    norm2_gamma: DVector<f32>,
    norm2_beta: DVector<f32>,
}

impl DecoderLayer {
    /// Create with random weights (for testing).
    pub fn new(d_model: usize, n_heads: usize, d_ff: usize) -> Self {
        Self {
            self_attn: MultiHeadAttention::new(d_model, n_heads),
            ffn: FeedForward::new(d_model, d_ff),
            norm1_gamma: DVector::from_element(d_model, 1.0),
            norm1_beta: DVector::zeros(d_model),
            norm2_gamma: DVector::from_element(d_model, 1.0),
            norm2_beta: DVector::zeros(d_model),
        }
    }

    /// Load from a WeightSet with a given prefix.
    pub fn from_weights(weights: &WeightSet, prefix: &str) -> Result<Self, WeightError> {
        let self_attn = MultiHeadAttention::from_weights(weights, &format!("{prefix}.self_attn"))?;
        let ffn = FeedForward::from_weights(weights, &format!("{prefix}.ffn"))?;

        let norm1_gamma_mat = weights.get_or_err(&format!("{prefix}.norm1.gamma"))?;
        let norm1_beta_mat = weights.get_or_err(&format!("{prefix}.norm1.beta"))?;
        let norm2_gamma_mat = weights.get_or_err(&format!("{prefix}.norm2.gamma"))?;
        let norm2_beta_mat = weights.get_or_err(&format!("{prefix}.norm2.beta"))?;

        Ok(Self {
            self_attn,
            ffn,
            norm1_gamma: DVector::from_column_slice(norm1_gamma_mat.as_slice()),
            norm1_beta: DVector::from_column_slice(norm1_beta_mat.as_slice()),
            norm2_gamma: DVector::from_column_slice(norm2_gamma_mat.as_slice()),
            norm2_beta: DVector::from_column_slice(norm2_beta_mat.as_slice()),
        })
    }

    /// Forward pass with pre-norm residual connections:
    /// x = x + self_attn(layer_norm(x))
    /// x = x + ffn(layer_norm(x))
    pub fn forward(&self, x: &DMatrix<f32>) -> DMatrix<f32> {
        // Self-attention block with residual
        let normed1 = layer_norm(x, &self.norm1_gamma, &self.norm1_beta, 1e-5);
        let attn_out = self.self_attn.forward(&normed1);
        let x = x + attn_out;

        // FFN block with residual
        let normed2 = layer_norm(&x, &self.norm2_gamma, &self.norm2_beta, 1e-5);
        let ffn_out = self.ffn.forward(&normed2);
        x + ffn_out
    }
}

// ---------------------------------------------------------------------------
// Detection Head
// ---------------------------------------------------------------------------

/// Detection head that produces class confidences and 3D bounding boxes.
pub struct DetectionHead {
    /// Projection to class logits: (d_model, n_classes)
    class_proj: DMatrix<f32>,
    /// Projection to box parameters: (d_model, 7) for [x, y, z, l, w, h, yaw]
    box_proj: DMatrix<f32>,
}

impl DetectionHead {
    /// Create with random weights (for testing).
    pub fn new(d_model: usize, n_classes: usize) -> Self {
        let scale = (1.0 / d_model as f32).sqrt();
        Self {
            class_proj: pseudo_random_matrix(d_model, n_classes, 900) * scale,
            box_proj: pseudo_random_matrix(d_model, 7, 1000) * scale,
        }
    }

    /// Load from a WeightSet with a given prefix.
    pub fn from_weights(weights: &WeightSet, prefix: &str) -> Result<Self, WeightError> {
        let class_proj = weights.get_or_err(&format!("{prefix}.class_proj"))?.clone();
        let box_proj = weights.get_or_err(&format!("{prefix}.box_proj"))?.clone();
        Ok(Self {
            class_proj,
            box_proj,
        })
    }

    /// Forward pass: for each query row, produce a Detection3D.
    ///
    /// x is (n_queries, d_model). Returns one detection per query.
    pub fn forward(&self, x: &DMatrix<f32>) -> Vec<Detection3D> {
        let class_logits = sigmoid(&matmul(x, &self.class_proj)); // (n_queries, n_classes)
        let box_params = matmul(x, &self.box_proj); // (n_queries, 7)

        let mut detections = Vec::with_capacity(x.nrows());
        for i in 0..x.nrows() {
            // Find best class and its confidence
            let mut best_class = 0u32;
            let mut best_conf = class_logits[(i, 0)];
            for c in 1..class_logits.ncols() {
                if class_logits[(i, c)] > best_conf {
                    best_conf = class_logits[(i, c)];
                    best_class = c as u32;
                }
            }

            detections.push(Detection3D {
                position: [
                    box_params[(i, 0)] as f64,
                    box_params[(i, 1)] as f64,
                    box_params[(i, 2)] as f64,
                ],
                dimensions: [
                    box_params[(i, 3)].abs() as f64,
                    box_params[(i, 4)].abs() as f64,
                    box_params[(i, 5)].abs() as f64,
                ],
                yaw: box_params[(i, 6)] as f64,
                class_id: best_class,
                confidence: best_conf as f64,
            });
        }
        detections
    }
}

// ---------------------------------------------------------------------------
// NativeDetectorConfig
// ---------------------------------------------------------------------------

/// Configuration for the native DETR-style detector.
#[derive(Debug, Clone)]
pub struct NativeDetectorConfig {
    /// Model embedding dimension.
    pub d_model: usize,
    /// Number of attention heads.
    pub n_heads: usize,
    /// Feed-forward hidden dimension.
    pub d_ff: usize,
    /// Number of decoder layers.
    pub n_layers: usize,
    /// Number of object queries (max detections per frame).
    pub n_queries: usize,
    /// Number of object classes.
    pub n_classes: usize,
    /// Minimum confidence to keep a detection.
    pub confidence_threshold: f64,
    /// IoU threshold for non-maximum suppression.
    pub nms_iou_threshold: f64,
}

impl Default for NativeDetectorConfig {
    fn default() -> Self {
        Self {
            d_model: 256,
            n_heads: 8,
            d_ff: 1024,
            n_layers: 6,
            n_queries: 100,
            n_classes: 10,
            confidence_threshold: 0.5,
            nms_iou_threshold: 0.4,
        }
    }
}

// ---------------------------------------------------------------------------
// NativeDetector
// ---------------------------------------------------------------------------

/// Pure-Rust DETR-style transformer detection decoder.
///
/// Implements a simplified DETR decoder that operates on pre-computed features
/// or raw sensor point positions. All computation uses `nalgebra::DMatrix<f32>`
/// on CPU.
///
/// The forward pass:
/// 1. Project input points to `d_model` dimensions
/// 2. Add learned object query embeddings (broadcast to input)
/// 3. Pass through N decoder layers
/// 4. Detection head produces class confidences + 3D bounding boxes
/// 5. Filter by confidence, apply NMS
pub struct NativeDetector {
    config: NativeDetectorConfig,
    input_proj: DMatrix<f32>,
    query_embed: DMatrix<f32>,
    layers: Vec<DecoderLayer>,
    head: DetectionHead,
}

impl NativeDetector {
    /// Create with random weights (for testing).
    ///
    /// `input_dim` is the dimensionality of each input point (e.g., 3 for xyz).
    pub fn new_random(config: NativeDetectorConfig, input_dim: usize) -> Self {
        let scale = (1.0 / input_dim as f32).sqrt();
        let layers = (0..config.n_layers)
            .map(|_| DecoderLayer::new(config.d_model, config.n_heads, config.d_ff))
            .collect();

        Self {
            input_proj: pseudo_random_matrix(input_dim, config.d_model, 1100) * scale,
            query_embed: pseudo_random_matrix(config.n_queries, config.d_model, 1200) * scale,
            layers,
            head: DetectionHead::new(config.d_model, config.n_classes),
            config,
        }
    }

    /// Load from SafeTensors weights via a [`WeightSet`].
    ///
    /// Expected weight layout:
    /// - `input_proj` — (input_dim, d_model)
    /// - `query_embed` — (n_queries, d_model)
    /// - `layer.{i}.self_attn.w_q`, `.w_k`, `.w_v`, `.w_o` — (d_model, d_model)
    /// - `layer.{i}.ffn.w1`, `.b1`, `.w2`, `.b2`
    /// - `layer.{i}.norm1.gamma`, `.norm1.beta`, `.norm2.gamma`, `.norm2.beta`
    /// - `head.class_proj` — (d_model, n_classes)
    /// - `head.box_proj` — (d_model, 7)
    pub fn from_safetensors(
        config: NativeDetectorConfig,
        weights: &WeightSet,
        _input_dim: usize,
    ) -> Result<Self, WeightError> {
        let input_proj = weights.get_or_err("input_proj")?.clone();
        let query_embed = weights.get_or_err("query_embed")?.clone();

        let mut layers = Vec::with_capacity(config.n_layers);
        for i in 0..config.n_layers {
            let prefix = format!("layer.{i}");
            layers.push(DecoderLayer::from_weights(weights, &prefix)?);
        }

        let head = DetectionHead::from_weights(weights, "head")?;

        Ok(Self {
            config,
            input_proj,
            query_embed,
            layers,
            head,
        })
    }

    /// Return the configuration.
    pub fn config(&self) -> &NativeDetectorConfig {
        &self.config
    }
}

impl DetectionPipeline for NativeDetector {
    fn detect(&self, input: &SensorInput) -> Vec<Detection3D> {
        if input.points.is_empty() {
            return Vec::new();
        }

        // 1. Convert input points to a matrix (n_points, 3)
        let n_points = input.points.len();
        let mut points_mat = DMatrix::zeros(n_points, 3);
        for (i, p) in input.points.iter().enumerate() {
            points_mat[(i, 0)] = p[0] as f32;
            points_mat[(i, 1)] = p[1] as f32;
            points_mat[(i, 2)] = p[2] as f32;
        }

        // 2. Project to d_model: features = points * input_proj  -> (n_points, d_model)
        let features = matmul(&points_mat, &self.input_proj);

        // 3. Build decoder input: use query embeddings.
        //    If we have fewer points than queries, pad with query embeddings.
        //    If more, take the first n_queries projected features + add query embed.
        let n_queries = self.config.n_queries;
        let d_model = self.config.d_model;

        let mut x = DMatrix::zeros(n_queries, d_model);
        let n_use = n_points.min(n_queries);

        // Add projected features where available
        for i in 0..n_use {
            for j in 0..d_model {
                x[(i, j)] = features[(i, j)] + self.query_embed[(i, j)];
            }
        }
        // Fill remaining with just query embeddings
        for i in n_use..n_queries {
            for j in 0..d_model {
                x[(i, j)] = self.query_embed[(i, j)];
            }
        }

        // 4. Pass through decoder layers
        for layer in &self.layers {
            x = layer.forward(&x);
        }

        // 5. Detection head
        let mut detections = self.head.forward(&x);

        // 6. Post-processing: confidence filter + NMS
        filter_by_confidence(&mut detections, self.config.confidence_threshold);
        nms_3d(&mut detections, self.config.nms_iou_threshold);

        detections
    }

    fn name(&self) -> &str {
        "NativeDetector"
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_relu() {
        let m = DMatrix::from_row_slice(2, 3, &[-1.0, 0.0, 1.0, -0.5, 2.0, -3.0]);
        let r = relu(&m);
        assert!((r[(0, 0)] - 0.0).abs() < 1e-7);
        assert!((r[(0, 1)] - 0.0).abs() < 1e-7);
        assert!((r[(0, 2)] - 1.0).abs() < 1e-7);
        assert!((r[(1, 0)] - 0.0).abs() < 1e-7);
        assert!((r[(1, 1)] - 2.0).abs() < 1e-7);
        assert!((r[(1, 2)] - 0.0).abs() < 1e-7);
    }

    #[test]
    fn test_softmax_rows() {
        let m = DMatrix::from_row_slice(2, 3, &[1.0, 2.0, 3.0, 1.0, 1.0, 1.0]);
        let s = softmax_rows(&m);

        // Each row should sum to 1.0
        for i in 0..s.nrows() {
            let row_sum: f32 = (0..s.ncols()).map(|j| s[(i, j)]).sum();
            assert!(
                (row_sum - 1.0).abs() < 1e-6,
                "Row {i} sum = {row_sum}, expected 1.0"
            );
        }

        // Row 1 (uniform input) should have equal values
        let expected_uniform = 1.0 / 3.0;
        for j in 0..3 {
            assert!(
                (s[(1, j)] - expected_uniform).abs() < 1e-6,
                "Uniform row element = {}, expected {}",
                s[(1, j)],
                expected_uniform
            );
        }

        // Row 0: values should be monotonically increasing
        assert!(s[(0, 0)] < s[(0, 1)]);
        assert!(s[(0, 1)] < s[(0, 2)]);
    }

    #[test]
    fn test_layer_norm() {
        let x = DMatrix::from_row_slice(2, 4, &[1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0]);
        let gamma = DVector::from_element(4, 1.0);
        let beta = DVector::zeros(4);

        let normed = layer_norm(&x, &gamma, &beta, 1e-5);

        // Each row should have approximately zero mean and unit variance
        for i in 0..normed.nrows() {
            let mut mean = 0.0f32;
            for j in 0..normed.ncols() {
                mean += normed[(i, j)];
            }
            mean /= normed.ncols() as f32;
            assert!(mean.abs() < 1e-5, "Row {i} mean = {mean}, expected ~0.0");

            let mut var = 0.0f32;
            for j in 0..normed.ncols() {
                let diff = normed[(i, j)] - mean;
                var += diff * diff;
            }
            var /= normed.ncols() as f32;
            assert!(
                (var - 1.0).abs() < 0.01,
                "Row {i} variance = {var}, expected ~1.0"
            );
        }
    }

    #[test]
    fn test_multihead_attention_shape() {
        let d_model = 256;
        let n_heads = 8;
        let seq_len = 10;

        let mha = MultiHeadAttention::new(d_model, n_heads);
        let x = pseudo_random_matrix(seq_len, d_model, 42);
        let out = mha.forward(&x);

        assert_eq!(out.nrows(), seq_len);
        assert_eq!(out.ncols(), d_model);
    }

    #[test]
    fn test_feedforward_shape() {
        let d_model = 256;
        let d_ff = 1024;
        let seq_len = 10;

        let ffn = FeedForward::new(d_model, d_ff);
        let x = pseudo_random_matrix(seq_len, d_model, 42);
        let out = ffn.forward(&x);

        assert_eq!(out.nrows(), seq_len);
        assert_eq!(out.ncols(), d_model);
    }

    #[test]
    fn test_decoder_layer_shape() {
        let d_model = 256;
        let n_heads = 8;
        let d_ff = 1024;
        let seq_len = 10;

        let layer = DecoderLayer::new(d_model, n_heads, d_ff);
        let x = pseudo_random_matrix(seq_len, d_model, 42);
        let out = layer.forward(&x);

        assert_eq!(out.nrows(), seq_len);
        assert_eq!(out.ncols(), d_model);
    }

    #[test]
    fn test_detection_head_outputs_detections() {
        let d_model = 256;
        let n_classes = 10;
        let n_queries = 20;

        let head = DetectionHead::new(d_model, n_classes);
        let x = pseudo_random_matrix(n_queries, d_model, 42);
        let dets = head.forward(&x);

        assert_eq!(dets.len(), n_queries);
        // All confidences should be in [0, 1] (sigmoid output)
        for d in &dets {
            assert!(d.confidence >= 0.0 && d.confidence <= 1.0);
        }
    }

    #[test]
    fn test_native_detector_end_to_end() {
        let config = NativeDetectorConfig {
            d_model: 64,
            n_heads: 4,
            d_ff: 128,
            n_layers: 2,
            n_queries: 10,
            n_classes: 5,
            confidence_threshold: 0.0, // keep all for testing
            nms_iou_threshold: 1.0,    // disable NMS for testing
        };

        let detector = NativeDetector::new_random(config, 3);

        let input = SensorInput {
            points: vec![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
            intensities: None,
            timestamp: 0.0,
        };

        let detections = detector.detect(&input);

        // With confidence_threshold=0.0 and nms_iou_threshold=1.0, we should
        // get at most n_queries detections (some may be suppressed by NMS).
        assert!(
            !detections.is_empty(),
            "Should produce at least one detection"
        );
        assert!(
            detections.len() <= 10,
            "Should produce at most n_queries detections"
        );

        // Each detection should have valid structure
        for d in &detections {
            assert!(d.confidence >= 0.0 && d.confidence <= 1.0);
            assert!(d.class_id < 5);
        }
    }

    #[test]
    fn test_native_detector_implements_pipeline_trait() {
        let config = NativeDetectorConfig {
            d_model: 32,
            n_heads: 4,
            d_ff: 64,
            n_layers: 1,
            n_queries: 5,
            n_classes: 3,
            confidence_threshold: 0.0,
            nms_iou_threshold: 1.0,
        };

        let detector = NativeDetector::new_random(config, 3);

        // Verify it can be used as a trait object
        let pipeline: &dyn DetectionPipeline = &detector;
        assert_eq!(pipeline.name(), "NativeDetector");

        let input = SensorInput {
            points: vec![[0.0, 0.0, 0.0]],
            intensities: None,
            timestamp: 0.0,
        };
        let _ = pipeline.detect(&input);
    }

    #[test]
    fn test_native_detector_config_default() {
        let config = NativeDetectorConfig::default();
        assert_eq!(config.d_model, 256);
        assert_eq!(config.n_heads, 8);
        assert_eq!(config.d_ff, 1024);
        assert_eq!(config.n_layers, 6);
        assert_eq!(config.n_queries, 100);
        assert_eq!(config.n_classes, 10);
        assert!((config.confidence_threshold - 0.5).abs() < f64::EPSILON);
        assert!((config.nms_iou_threshold - 0.4).abs() < f64::EPSILON);
    }

    #[test]
    fn test_native_detector_empty_input() {
        let config = NativeDetectorConfig {
            d_model: 32,
            n_heads: 4,
            d_ff: 64,
            n_layers: 1,
            n_queries: 5,
            n_classes: 3,
            confidence_threshold: 0.0,
            nms_iou_threshold: 1.0,
        };

        let detector = NativeDetector::new_random(config, 3);
        let input = SensorInput {
            points: vec![],
            intensities: None,
            timestamp: 0.0,
        };

        let detections = detector.detect(&input);
        assert!(
            detections.is_empty(),
            "Empty input should produce no detections"
        );
    }
}
