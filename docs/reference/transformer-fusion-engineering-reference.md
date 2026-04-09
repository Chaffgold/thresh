# Transformer-based multi-sensor fusion for multi-object tracking

**Transformer architectures have fundamentally reshaped how autonomous systems fuse data from cameras, LiDAR, and radar to track multiple objects simultaneously.** The cross-attention mechanism—where object queries selectively retrieve information from heterogeneous sensor feature maps—replaces hand-crafted geometric projection with learned, soft alignment that degrades gracefully under calibration error or sensor failure. This guide provides a self-contained engineering reference: from the Kalman filter equations that still underpin most tracking pipelines, through the data association mathematics that link detections to tracks, to the transformer fusion architectures (BEVFusion, TransFusion, CMT, FUTR3D) that currently dominate the nuScenes and Waymo benchmarks.

-----

## 1. State estimation: Kalman filter family from first principles

### 1.1 Linear Kalman filter

**State-space model.** A discrete-time linear system with state **x**_k in R^n and measurement **z**_k in R^m:

> **x**_k = **F**_k **x**_{k-1} + **G**_k **w**_k
> **z**_k = **H**_k **x**_k + **v**_k

where **F**_k is the state transition matrix, **H**_k the observation matrix, **w**_k ~ N(**0**, **Q**_k) process noise, and **v**_k ~ N(**0**, **R**_k) measurement noise.

**Predict step:**
> **x_hat**_{k|k-1} = **F**_k **x_hat**_{k-1|k-1}
> **P**_{k|k-1} = **F**_k **P**_{k-1|k-1} **F**_k^T + **Q**_k

**Update step:**
> **y_tilde**_k = **z**_k - **H**_k **x_hat**_{k|k-1} (innovation)
> **S**_k = **H**_k **P**_{k|k-1} **H**_k^T + **R**_k (innovation covariance)
> **K**_k = **P**_{k|k-1} **H**_k^T **S**_k^{-1} (Kalman gain)
> **x_hat**_{k|k} = **x_hat**_{k|k-1} + **K**_k **y_tilde**_k
> **P**_{k|k} = (**I** - **K**_k **H**_k) **P**_{k|k-1}

Joseph form for numerical stability:
> **P**_{k|k} = (**I** - **K**_k **H**_k) **P**_{k|k-1} (**I** - **K**_k **H**_k)^T + **K**_k **R**_k **K**_k^T

### 1.2 Extended Kalman filter

For nonlinear dynamics f() and observation h(), linearize via Jacobians:
> **F**_k = df/dx |_{x_hat_{k-1|k-1}}
> **H**_k = dh/dx |_{x_hat_{k|k-1}}

**Common nonlinear motion models:**

**Constant Turn Rate and Velocity (CTRV).** State **x** = [x, y, theta, v, omega]^T. For omega != 0:
> x_{k+1} = x_k + (v/omega)[sin(theta + omega*dt) - sin(theta)]
> y_{k+1} = y_k + (v/omega)[-cos(theta + omega*dt) + cos(theta)]
> theta_{k+1} = theta_k + omega*dt; v_{k+1} = v_k; omega_{k+1} = omega_k

### 1.3 Unscented Kalman filter

**Scaled sigma point selection** for state dimension n, mean x_hat, covariance P, parameters alpha, beta, kappa:
> lambda = alpha^2(n + kappa) - n

Generate 2n + 1 sigma points. Typical values: alpha in [1e-3, 1], beta = 2, kappa = 3 - n or 0.

The UKF captures mean and covariance accurately to **second order** for any nonlinearity, versus first order for the EKF. It is the default filter for CTRV-model trackers in autonomous driving stacks (Autoware, AB3DMOT).

-----

## 2. Multi-sensor fusion mathematics

### 2.1 Centralized measurement-level fusion

Stack measurements from all sensors:
> **z**_stacked = [z_1; z_2; ...; z_N]
> **H**_stacked = [H_1; H_2; ...; H_N]
> **R**_stacked = blkdiag(R_1, R_2, ..., R_N)

Heterogeneous sensor rates (LiDAR at 10 Hz, cameras at 20-30 Hz, radar at 13-20 Hz) require asynchronous update scheduling.

### 2.2 Information filter form

Parameterized by **Y** = P^{-1} (information matrix) and **y_hat** = P^{-1} x_hat (information state). Update is purely additive:
> Y_{k|k} = Y_{k|k-1} + H^T R^{-1} H
> y_hat_{k|k} = y_hat_{k|k-1} + H^T R^{-1} z_k

Natural choice for decentralized multi-sensor architectures.

### 2.3 Covariance intersection for unknown correlations

Fuses two estimates conservatively when cross-covariances are unknown:
> P_fused^{-1} = omega * P_A^{-1} + (1 - omega) * P_B^{-1}
> x_hat_fused = P_fused [omega * P_A^{-1} x_hat_A + (1 - omega) * P_B^{-1} x_hat_B]

-----

## 3. Data association: linking detections to tracks

### 3.1 Hungarian algorithm (linear assignment)

O(n^3) optimal one-to-one assignment. Cost typically Mahalanobis distance, IoU-based, or appearance-based.

### 3.2 Joint Probabilistic Data Association (JPDA)

Soft association weighting each measurement's contribution to each track by posterior association probability.

### 3.3 Multi-Hypothesis Tracking (MHT)

Defers hard decisions via tree of association hypotheses over sliding window. Murty's k-best algorithm finds top-k assignments in O(k * n^3).

-----

## 4. How transformers replace and extend classical fusion

### 4.1 Cross-attention as soft sensor fusion

> Attention(q, K, V) = softmax(q K^T / sqrt(d_k)) V

TransFusion: queries attend to LiDAR BEV features (layer 1), then camera features (layer 2) with Spatially Modulated Cross-Attention (SMCA).

### 4.2 Deformable attention for efficiency

Reduces O(N_queries * N_features) to O(N_queries * K) where K = 4-8 learned sampling points.

### 4.3 Positional encodings bridge coordinate systems

CMT coordinate encoding, PETR Position Embedding Transformation, sinusoidal 3D encoding.

-----

## 5. Dominant architectures

| Architecture | Fusion level | Key mechanism | nuScenes NDS | Latency |
|---|---|---|---|---|
| BEVFusion | BEV mid-fusion | Concatenation + conv | 72.9 | ~40 ms (Orin INT8) |
| TransFusion | Query-level | SMCA cross-attention | 71.7 | ~119 ms (3090) |
| CMT | Token-level | Coordinate-encoded cross-attention | 74.1 | -- |
| FUTR3D | Query-level | Modality-agnostic sampling | -- | -- |
| UniTR | Backbone-level | Shared weights + inter-modal blocks | 73.1 | 88.7 ms (A100) |

-----

## 6. Track queries: end-to-end learned data association

TrackFormer, MOTR, MeMOTR (+13.0% AssA on DanceTrack), MATR (71.3 HOTA on DanceTrack).

-----

## 7. Loss functions and training recipes

Hungarian matching loss with focal loss, L1 regression, GIoU loss. Multi-stage training (pretrain image backbone -> train LiDAR-only -> train fusion).

-----

## 8. Benchmarks and engineering realities

Current best: 77.9% AMOTA (NEMOT, LiDAR+Cam). Tracking-by-detection still leads end-to-end by ~10 AMOTA points. BEVFusion achieves 25 FPS on Jetson Orin (INT8).

**Sensor degradation strategies:** Decoupled streams, masked-modal training, soft attention down-weighting, augmentation.

-----

## Conclusion

Converging architecture: modality-specific encoders -> transformer fusion -> set-prediction detection -> query-propagation tracking. Most reliable recipe today: BEVFusion or TransFusion for detection, UKF (CTRV model) for state propagation, Hungarian assignment on fused motion-appearance cost matrix.
