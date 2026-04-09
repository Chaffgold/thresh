## 1. Project Scaffolding

- [ ] 1.1 Initialize Cargo workspace with `Cargo.toml` at root, defining all workspace members
- [ ] 1.2 Create `thresh-core` crate with common types: `StateVector`, `Measurement` enum, `Covariance`, `Timestamp`, coordinate frame types
- [ ] 1.3 Create `thresh-filter` crate with `nalgebra` dependency and `MotionModel` trait definition
- [ ] 1.4 Create `thresh-association` crate skeleton
- [ ] 1.5 Create `thresh-fusion` crate skeleton
- [ ] 1.6 Create `thresh-inference` crate with `ort` dependency
- [ ] 1.7 Create `thresh-tracker` crate skeleton
- [ ] 1.8 Create `thresh-bridge` crate with `pyo3` dependency, gated behind `stonesoup` feature
- [ ] 1.9 Create `thresh-synth` crate skeleton
- [ ] 1.10 Create `thresh-eval` crate skeleton
- [ ] 1.11 Create top-level `thresh` binary crate re-exporting workspace crates
- [ ] 1.12 Configure workspace-level `.gitignore` for target/, generated ONNX files, Python venvs

## 2. Core Types (thresh-core)

- [ ] 2.1 Define `StateVector` type wrapping `nalgebra::DVector<f64>` with dimension metadata
- [ ] 2.2 Define `CovarianceMatrix` type wrapping `nalgebra::DMatrix<f64>` with symmetry enforcement
- [ ] 2.3 Define `Measurement` enum with `Radar`, `EoIr`, `AdsB` variants, each with observation matrix H and noise R
- [ ] 2.4 Define `TrackId` (u64, globally unique), `TrackState` enum (Tentative, Confirmed, Coasting, Deleted)
- [ ] 2.5 Define `BoundingBox3D` struct (x, y, z, l, w, h, yaw) for detection outputs
- [ ] 2.6 Define `SensorRegistration` struct with position, orientation, coordinate transform methods
- [ ] 2.7 Implement polar-to-Cartesian and Cartesian-to-polar coordinate transforms
- [ ] 2.8 Write unit tests for all core type constructors and coordinate transforms

## 3. State Estimation (thresh-filter)

- [ ] 3.1 Define `MotionModel` trait with `predict`, `jacobian`, `process_noise` methods
- [ ] 3.2 Define `LinearModel` sub-trait extending `MotionModel` with constant `F` matrix access
- [ ] 3.3 Implement Constant Velocity (CV) motion model — state [x, vx, y, vy, z, vz]
- [ ] 3.4 Implement Constant Acceleration (CA) motion model — state [x, vx, ax, y, vy, ay, z, vz, az]
- [ ] 3.5 Implement CTRV motion model — state [x, y, theta, v, omega] with omega→0 degenerate handling
- [ ] 3.6 Implement Coordinated Turn model — state [x, vx, y, vy, omega] with quasi-linear transition
- [ ] 3.7 Implement Linear Kalman Filter: predict step, update step with Joseph-form covariance
- [ ] 3.8 Implement Extended Kalman Filter: Jacobian-based linearization, nonlinear predict/update
- [ ] 3.9 Implement Unscented Kalman Filter: sigma point generation (Van der Merwe), weighted predict/update
- [ ] 3.10 Add configurable UKF parameters (alpha, beta, kappa) with sensible defaults
- [ ] 3.11 Write tests: KF convergence on linear system, verify covariance stays PSD over 1000 steps
- [ ] 3.12 Write tests: EKF with CTRV model tracks a simulated turning target
- [ ] 3.13 Write tests: UKF sigma point weights sum to 1, second-order accuracy on polar-to-Cartesian
- [ ] 3.14 Write tests: CTRV model degenerates gracefully when omega < epsilon

## 4. Data Association (thresh-association)

- [ ] 4.1 Implement Hungarian algorithm (Jonker-Volgenant) for optimal linear assignment on cost matrices
- [ ] 4.2 Handle rectangular cost matrices with unassigned track/detection lists
- [ ] 4.3 Implement Mahalanobis distance computation: d^2 = (z - Hx)^T S^{-1} (z - Hx)
- [ ] 4.4 Implement chi-squared gating with configurable significance level and auto-adjustment for measurement dimension
- [ ] 4.5 Implement 2D IoU computation for bounding boxes
- [ ] 4.6 Implement 3D IoU computation for rotated bounding boxes (x, y, z, l, w, h, yaw)
- [ ] 4.7 Implement fused cost matrix: C = alpha * d_motion + (1-alpha) * d_appearance with configurable alpha
- [ ] 4.8 Implement cascaded association: high-confidence first pass, low-confidence second pass
- [ ] 4.9 Write tests: Hungarian on known 5x5 matrix matches expected optimal assignment
- [ ] 4.10 Write tests: Mahalanobis gating correctly rejects out-of-gate detections
- [ ] 4.11 Write tests: 3D IoU returns 1.0 for identical boxes, 0.0 for non-overlapping

## 5. Sensor Fusion (thresh-fusion)

- [ ] 5.1 Implement centralized measurement stacking: z_stacked, H_stacked, R_stacked (block-diagonal)
- [ ] 5.2 Implement single-update centralized fusion with stacked measurements
- [ ] 5.3 Implement asynchronous sensor updates — apply individual sensor measurements as they arrive
- [ ] 5.4 Implement Information Filter: Y = P^{-1}, y_hat = P^{-1} x_hat, additive update
- [ ] 5.5 Implement information-to-covariance conversion (Y^{-1} → P, x_hat recovery)
- [ ] 5.6 Implement Covariance Intersection (CI) with 1D line search for optimal omega
- [ ] 5.7 Implement sensor registration: store sensor params, transform measurements to common frame
- [ ] 5.8 Implement radar polar-to-Cartesian measurement conversion using sensor registration
- [ ] 5.9 Write tests: centralized fusion of 2 sensors matches sequential independent updates
- [ ] 5.10 Write tests: information filter produces equivalent result to standard KF update
- [ ] 5.11 Write tests: CI fused covariance bounds both input covariances

## 6. Transformer Inference (thresh-inference)

- [ ] 6.1 Set up `ort` crate with ONNX Runtime session creation and execution provider selection (CPU/CUDA/TensorRT)
- [ ] 6.2 Implement execution provider fallback chain: TensorRT → CUDA → CPU with logging
- [ ] 6.3 Implement ONNX model loader that creates sessions with dynamic axes support
- [ ] 6.4 Implement pipeline orchestrator: define stage ordering, pass intermediate tensors between sessions
- [ ] 6.5 Implement BEVFusion-style pipeline config: camera encoder → LiDAR encoder → BEV pool → fusion → detection head
- [ ] 6.6 Implement query-based pipeline config: backbone → transformer decoder with cross-attention → detections
- [ ] 6.7 Implement per-component latency measurement and total pipeline timing
- [ ] 6.8 Implement FP16/INT8 precision configuration per session
- [ ] 6.9 Parse detection outputs (boxes, scores, classes, velocities) from ONNX output tensors into `BoundingBox3D`
- [ ] 6.10 Write tests: load a simple test ONNX model, verify session creation and inference runs
- [ ] 6.11 Write tests: dynamic shape input with different batch sizes produces correct output shapes

## 7. Track Management (thresh-tracker)

- [ ] 7.1 Implement Track struct: id, state, covariance, lifecycle state, class, history
- [ ] 7.2 Implement track lifecycle state machine: Tentative → Confirmed → Coasting → Deleted
- [ ] 7.3 Implement M-of-N confirmation policy (configurable M and N)
- [ ] 7.4 Implement max-coast-age deletion policy
- [ ] 7.5 Implement track birth from unassigned detections with configurable initial covariance
- [ ] 7.6 Implement multi-sensor corroborated track initialization (spatial gating across sensors)
- [ ] 7.7 Implement class-specific track heads: mapping target class → motion model + noise params + policies
- [ ] 7.8 Implement track class reclassification with motion model switching and state adaptation
- [ ] 7.9 Implement globally unique TrackId allocation (monotonic counter, never reused)
- [ ] 7.10 Implement the main tracker loop: predict all tracks → get detections → associate → update → manage lifecycle
- [ ] 7.11 Write tests: M-of-N confirmation with 3-of-5 policy
- [ ] 7.12 Write tests: track coasts for N frames then gets deleted
- [ ] 7.13 Write tests: track identity preserved through coast and re-association
- [ ] 7.14 Write tests: 10,000 track create/delete cycles — no ID collisions

## 8. Stone Soup Bridge (thresh-bridge)

- [ ] 8.1 Set up PyO3 with pyo3-build-config, gated behind `stonesoup` Cargo feature
- [ ] 8.2 Implement nalgebra-to-numpy type conversion (DVector → numpy array, DMatrix → numpy 2D array)
- [ ] 8.3 Implement Measurement → Stone Soup Detection conversion
- [ ] 8.4 Implement wrapper for Stone Soup JPDA data associator
- [ ] 8.5 Implement wrapper for Stone Soup MHT tracker
- [ ] 8.6 Implement wrapper for Stone Soup IMM filter
- [ ] 8.7 Implement wrapper for Stone Soup Gaussian Mixture PHD filter
- [ ] 8.8 Handle GIL management: acquire for Python calls, release during Rust compute
- [ ] 8.9 Implement graceful error when Python/Stone Soup is not installed at runtime
- [ ] 8.10 Write tests: build without `stonesoup` feature succeeds, core tracking works
- [ ] 8.11 Write integration tests: JPDA via bridge matches Stone Soup's own output on reference scenario

## 9. Synthetic Data Generation (thresh-synth)

- [ ] 9.1 Implement trajectory generator base: initial state, time step, duration, segment list
- [ ] 9.2 Implement CV trajectory segment
- [ ] 9.3 Implement CA trajectory segment
- [ ] 9.4 Implement CTRV maneuver trajectory segment with configurable turn rate
- [ ] 9.5 Implement ballistic trajectory segment with gravity and optional drag
- [ ] 9.6 Implement multi-segment trajectory stitching with smooth transitions
- [ ] 9.7 Implement radar measurement generator: range/azimuth/elevation noise, P_d, Poisson clutter
- [ ] 9.8 Implement RCS-dependent detection probability via radar equation
- [ ] 9.9 Implement EO/IR measurement generator: angular noise, FOV constraints, IR-signature-dependent P_d
- [ ] 9.10 Implement ADS-B message generator: 1 Hz position with NACp quantization and dropout
- [ ] 9.11 Implement multi-target scenario composer: N targets × M sensors, time-ordered output stream
- [ ] 9.12 Implement scenario serialization to JSON for reproducible test cases
- [ ] 9.13 Write tests: CV trajectory matches analytical position at each timestep
- [ ] 9.14 Write tests: radar measurement noise statistics match configured sigma over 10K samples
- [ ] 9.15 Write tests: multi-target scenario with 50 targets generates coherent data

## 10. Evaluation Metrics (thresh-eval)

- [ ] 10.1 Implement ground-truth to track matching via Hungarian assignment at each frame
- [ ] 10.2 Implement MOTA computation: 1 - (FN + FP + IDSW) / GT
- [ ] 10.3 Implement MOTP computation: average localization error for matched pairs
- [ ] 10.4 Implement IDF1 computation: optimal global trajectory matching, 2*IDTP / (2*IDTP + IDFP + IDFN)
- [ ] 10.5 Implement HOTA computation: sqrt(DetA * AssA) integrated over IoU thresholds 0.05 to 0.95
- [ ] 10.6 Implement per-threshold HOTA breakdown (DetA, AssA at each alpha)
- [ ] 10.7 Implement AMOTA: MOTA averaged over multiple recall thresholds
- [ ] 10.8 Implement per-class metric breakdown
- [ ] 10.9 Implement JSON report output with all metrics keyed by name and class
- [ ] 10.10 Implement human-readable table output (terminal-formatted)
- [ ] 10.11 Write tests: perfect tracking yields MOTA=1.0, HOTA=1.0
- [ ] 10.12 Write tests: known ID switch scenario yields correct IDSW count and MOTA penalty
- [ ] 10.13 Write tests: HOTA decomposition — high DetA + low AssA reflects in overall score

## 11. Integration and End-to-End Testing

- [ ] 11.1 Create an end-to-end integration test: synth scenario → tracker (KF + Hungarian) → eval metrics
- [ ] 11.2 Create a multi-sensor integration test: radar + EO/IR → centralized fusion → tracker → eval
- [ ] 11.3 Create a class-specific tracking test: mixed aerodynamic + ballistic targets with appropriate models
- [ ] 11.4 Benchmark tracker throughput: measure Hz for 50-target scenario with UKF + Hungarian
- [ ] 11.5 Document example usage in README.md with minimal code snippets
- [ ] 11.6 Add CI configuration (GitHub Actions) for `cargo test`, `cargo clippy`, `cargo fmt --check`
