## Context

Thresh is a greenfield Rust project for transformer-based multi-sensor fusion multi-object tracking, targeting heterogeneous aerospace targets (UAVs through ballistic missiles). The autonomous driving community has proven these architectures (TransFusion, BEVFusion, CenterPoint) at 70+ mAP on nuScenes, but no open-source Rust implementation exists for the defense/aerospace domain. The project leverages Stone Soup (Python) as a dependency for advanced algorithms (JPDA, MHT, IMM) until Rust-native implementations are ready.

Key constraints:
- Hybrid architecture: transformer detection + classical Bayesian state estimation
- Modular ONNX pipeline (no monolithic model) — each component independently certifiable
- Target dynamics span 10^0 to 10^4 m/s requiring class-specific motion models
- No public defense tracking datasets — synthetic data pipeline required
- Stone Soup dependency (not reimplementation) for complex algorithms

## Goals / Non-Goals

**Goals:**
- Implement core Kalman filter family (KF, EKF, UKF) in pure Rust with nalgebra
- Build Hungarian assignment and Mahalanobis-gated data association in Rust
- Create multi-sensor fusion framework (centralized, information filter, covariance intersection)
- Integrate ONNX Runtime for modular transformer inference pipeline
- Bridge to Stone Soup via PyO3 for JPDA, MHT, IMM (feature-gated)
- Provide synthetic data generation for radar, EO/IR, and ADS-B sensors
- Implement MOT evaluation metrics (MOTA, IDF1, HOTA, AMOTA)
- Structure as a Cargo workspace with well-defined crate boundaries

**Non-Goals:**
- Training transformer models (this is inference and classical tracking only)
- Reimplementing Stone Soup algorithms in Rust (deferred — use the bridge)
- Building a GUI or visualization (CLI/library only for now)
- Real sensor hardware integration (synthetic data and ONNX models only)
- Achieving DO-178C or MIL-STD certification (design for certifiability, don't certify)
- End-to-end learned tracking (transformer track queries) — keep classical state estimation

## Decisions

### 1. Cargo workspace with per-capability crates

**Decision:** Structure as a Cargo workspace with separate crates:
- `thresh-core` — common types: state vectors, measurements, covariance matrices, time
- `thresh-filter` — KF, EKF, UKF implementations with motion model traits
- `thresh-association` — Hungarian, Mahalanobis gating, cost matrix construction
- `thresh-fusion` — centralized fusion, information filter, covariance intersection
- `thresh-inference` — ONNX Runtime integration, pipeline orchestration
- `thresh-tracker` — track management, lifecycle, class-specific heads
- `thresh-bridge` — PyO3 Stone Soup bridge (feature-gated)
- `thresh-synth` — synthetic data generation
- `thresh-eval` — evaluation metrics
- `thresh` — top-level binary/integration crate

**Rationale:** Clean separation of concerns, independent compilation, optional features via Cargo feature flags. Users who only need filters don't pull in ONNX Runtime or PyO3.

**Alternatives considered:**
- Single crate with modules: simpler initially but poor compile times and forced dependency on ONNX/Python for all users
- Separate repos: too much overhead for a cohesive system

### 2. nalgebra for linear algebra

**Decision:** Use `nalgebra` for all matrix operations (state vectors, covariances, Jacobians).

**Rationale:** nalgebra is the de facto Rust linear algebra library — statically sized matrices for known dimensions (6x6 covariance), dynamically sized for variable-dimension cases. Strong type safety prevents dimension mismatches at compile time. Joseph-form covariance update maps directly to nalgebra matrix operations.

**Alternatives considered:**
- `ndarray`: better for tensor-like data but weaker type-level dimension checking
- Raw arrays: error-prone, no BLAS integration

### 3. Motion model trait with generic filter

**Decision:** Define a `MotionModel` trait:
```rust
trait MotionModel {
    type State: StateVector;
    fn predict(&self, state: &Self::State, dt: f64) -> Self::State;
    fn jacobian(&self, state: &Self::State, dt: f64) -> DMatrix<f64>;
    fn process_noise(&self, dt: f64) -> DMatrix<f64>;
}
```
Filters (KF, EKF, UKF) are generic over `MotionModel`. Built-in models: CV, CA, CTRV, CoordinatedTurn, Ballistic.

**Rationale:** Users can implement custom dynamics (orbital, hypersonic with drag) without modifying filter code. The UKF doesn't need the Jacobian method — it's only required by EKF. Use a separate `LinearModel` sub-trait for KF.

### 4. ONNX Runtime via `ort` crate

**Decision:** Use the `ort` crate (official Rust bindings for ONNX Runtime) for transformer inference. Each pipeline stage is a separate `ort::Session`.

**Rationale:** `ort` provides safe Rust bindings, supports CUDA/TensorRT execution providers, handles dynamic shapes. The modular pipeline approach (separate ONNX files per stage) aligns with defense certifiability requirements and the research consensus (no published system uses a single ONNX for full fusion tracking).

**Alternatives considered:**
- `tract`: pure Rust ONNX runtime, but lacking GPU acceleration and TensorRT
- `tch-rs` (PyTorch C++ bindings): ties to PyTorch rather than ONNX, less portable

### 5. Stone Soup bridge via PyO3 (feature-gated)

**Decision:** Use PyO3 to call Stone Soup from Rust, gated behind a `stonesoup` Cargo feature. The bridge converts nalgebra types to numpy arrays and Stone Soup State objects. Core functionality works without Python.

**Rationale:** Stone Soup provides battle-tested JPDA, MHT, IMM implementations. Reimplementing these in Rust is a major effort better deferred. The feature gate ensures the core library has zero Python dependency for users who don't need advanced algorithms.

### 6. Synthetic data over real dataset dependency

**Decision:** Build synthetic data generation rather than depending on nuScenes/Waymo.

**Rationale:** No public defense tracking dataset exists. Synthetic data with configurable target dynamics (Mach 0.5 to Mach 20), RCS profiles, and sensor characteristics is the only path. This also enables controlled evaluation of tracker performance across specific scenarios. The synthetic pipeline generates time-ordered ground-truth + measurement streams directly consumable by the tracker and evaluation modules.

### 7. Measurement types via enum + trait dispatch

**Decision:** Use an enum for measurement sources with trait-based dispatch:
```rust
enum Measurement {
    Radar { range: f64, azimuth: f64, elevation: f64, range_rate: Option<f64> },
    EoIr { azimuth: f64, elevation: f64 },
    AdsB { lat: f64, lon: f64, alt: f64, velocity: Option<Vec3> },
}
```
Each variant knows its observation matrix H and noise covariance R.

**Rationale:** Heterogeneous sensor support without dynamic dispatch overhead. The enum is exhaustive — adding a new sensor type is a compile error until all match arms are handled.

## Risks / Trade-offs

**[Risk] ONNX model compatibility across architectures** → Mitigation: Support opset 17, test with reference BEVFusion/TransFusion exports. Provide example model conversion scripts in Python.

**[Risk] PyO3 GIL contention in multi-threaded tracker** → Mitigation: Batch Stone Soup calls, release GIL during Rust compute. Profile GIL hold times. Long-term: replace with Rust-native implementations.

**[Risk] nalgebra dynamic matrix performance for large state vectors** → Mitigation: Use statically-sized matrices (`SMatrix<f64, 6, 6>`) for common cases, fall back to dynamic only for user-defined variable-dimension models.

**[Risk] No defense-domain validation data** → Mitigation: Synthetic data pipeline covers the gap. Design scenarios that stress the tracker's heterogeneous target handling. Cross-validate against Stone Soup's reference implementations on identical synthetic scenarios.

**[Risk] CTRV singularity at omega → 0** → Mitigation: Implement the L'Hôpital-rule degenerate case explicitly in the CTRV model, switching to straight-line CV when |omega| < epsilon.

**[Trade-off] Hybrid vs end-to-end** → We deliberately choose hybrid (transformer detection + classical filtering) over end-to-end learned tracking. This sacrifices potential AMOTA gains (~10 points based on nuScenes benchmarks) in exchange for certifiable uncertainty quantification and 200+ Hz state propagation. The research consensus supports this for defense applications.

**[Trade-off] Stone Soup dependency vs pure Rust** → Using Stone Soup accelerates development but introduces Python runtime dependency for advanced features. The feature gate makes this opt-in, and the long-term plan is Rust-native replacements.

## Open Questions

- What specific ONNX model architectures should be supported first? BEVFusion and TransFusion are the most documented for export.
- Should the IMM filter be the first algorithm ported from Stone Soup to Rust, given its importance for maneuvering targets?
- What coordinate system standard should the common tracking frame use? ENU (East-North-Up) is standard for defense; ECEF for global tracking.
- Should we support distributed/federated tracking across multiple fusion nodes from the start, or add it later?
