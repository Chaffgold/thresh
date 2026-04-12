# Performance Profiling and Optimization

## What

Profile the tracker and association hot paths, add criterion benchmarks for critical operations, and optimize the highest-impact bottlenecks including Hungarian assignment, covariance update, and Mahalanobis gating. The goal is to make thresh viable for large-N scenarios (100+ simultaneous targets with high clutter density) without changing the mathematical operations.

## Why

No profiling or performance optimization has been done on the codebase. The current implementation prioritizes correctness and readability, which is appropriate for an early-stage project, but leaves significant performance on the table. Hungarian assignment is O(n^3), covariance updates involve dense matrix multiplications, and the tracker step loop processes tracks sequentially. For operationally relevant scenarios with 100+ targets and dense clutter, the tracker step becomes the throughput bottleneck. nalgebra already supports SIMD, and rayon can parallelize independent track updates, but neither is being leveraged.

## How

- Add criterion micro-benchmarks for the hot paths: Hungarian assignment (10, 50, 100, 500 targets), Kalman filter predict/update cycle, Mahalanobis distance computation, and full tracker step
- Profile end-to-end tracker runs with cargo-flamegraph on the synthetic benchmark scenarios to identify actual bottlenecks vs. assumed ones
- Enable nalgebra SIMD features for matrix operations in the filter and association crates
- Add rayon parallelism for independent track predict/update operations in the tracker step loop
- Optimize data layout for cache friendliness: consider struct-of-arrays for track state storage if profiling shows cache miss overhead
- Establish a performance CI gate using criterion's comparison mode to catch regressions

## Out of scope

- GPU offload via CUDA or Metal
- Distributed processing across multiple machines
- Algorithm changes (e.g., replacing Hungarian with auction algorithm) -- keep the same mathematical operations, just make them faster
- Optimizing the ONNX inference path (that belongs to thresh-inference)

## Affected crates

- thresh-association: Hungarian algorithm optimization, parallel gating
- thresh-filter: KF predict/update SIMD enablement, cache-friendly state layout
- thresh-tracker: parallel track update loop, benchmark harness
- Cargo.toml (workspace): rayon and criterion dependency additions
