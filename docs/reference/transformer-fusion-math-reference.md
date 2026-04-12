# Transformer-Based Multi-Sensor Fusion for Heterogeneous Target Tracking

### A Mathematical Reference for ML Engineers Bridging into Bayesian Tracking

> **Covers:** Kalman / EKF / UKF derivations, Transformer attention mechanics,
> Hungarian matching, Focal, GIoU & Hungarian loss functions, ONNX deployment
>
> **Audience:** ML engineers with strong deep-learning background seeking to
> understand classical tracking theory and how transformers can replace,
> augment, or extend it.

## Key Equations Summary

### Kalman Filter
- Predict: x_hat_{t|t-1} = F x_hat_{t-1|t-1}; P_{t|t-1} = F P_{t-1|t-1} F^T + Q
- Innovation: z_tilde = z - H x_hat_{t|t-1}; S = H P_{t|t-1} H^T + R
- Gain: K = P_{t|t-1} H^T S^{-1}
- Update: x_hat_{t|t} = x_hat_{t|t-1} + K z_tilde
- Joseph form: P_{t|t} = (I - KH) P_{t|t-1} (I - KH)^T + K R K^T

### Transformer Attention
- Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V
- Proven: causally-masked transformer can exactly represent a Kalman Filter (de Bezenac et al., 2023)

### Hungarian Assignment
- sigma* = argmin_sigma sum_i C_{i,sigma(i)}, solved in O(n^3)
- DETR cost: C = -lambda_cls * p_hat + lambda_L1 * |b - b_hat|_1 + lambda_GIoU * L_GIoU

### Loss Functions
- Focal: L = -alpha_t (1-p_t)^gamma log(p_t), gamma=2, alpha=0.25
- GIoU: L = 1 - (IoU - |C \ (A union B)| / |C|), range [0, 2]
- DETR total: focal + lambda_L1 * L1 + lambda_GIoU * L_GIoU after Hungarian matching

### Architecture Comparison
| Criterion | Transformer Fusion | Classical KF/UKF/IMM |
|---|---|---|
| Data association | Implicit (attention) | Explicit (Hungarian, JPDA, MHT) |
| Uncertainty | Uncalibrated confidence | Calibrated covariance P |
| Latency | 8-25 FPS (GPU) | >200 FPS (CPU) |
| Certifiability | Difficult | Straightforward |
| nuScenes AMOTA | >77% | ~60% |

### Recommended Hybrid Architecture
Transformer multi-sensor fusion for detection -> Classical IMM-UKF for state estimation
Best of both: learned cross-modal attention + principled uncertainty quantification.

### ONNX Deployment
Modular pipeline: Cam Encoder -> LiDAR Voxeliser -> BEV Fusion -> Detection Head -> Tracker (classical)
No published defense system uses fully end-to-end single ONNX tracker.

### Key References
- Vaswani et al. (2017) — Attention Is All You Need
- Bai et al. (2022) — TransFusion
- Liu et al. (2023) — BEVFusion
- Carion et al. (2020) — DETR
- de Bezenac et al. (2023) — Transformer as KF
- Lin et al. (2017) — Focal Loss
- Yin et al. (2021) — CenterPoint
