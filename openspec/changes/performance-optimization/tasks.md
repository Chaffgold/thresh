# Performance Profiling and Optimization — Tasks

## 1. Benchmark infrastructure

- [x] 1.1 Add `criterion` as a workspace dev-dependency in root `Cargo.toml`
- [x] 1.2 Create `crates/thresh-association/benches/hungarian.rs` with `hungarian_10x10`, `hungarian_100x100`, `hungarian_500x500` benchmarks using seeded random cost matrices
- [x] 1.3 Create `crates/thresh-filter/benches/kalman.rs` with `kf_predict_6d` and `kf_update_6d` benchmarks
- [x] 1.4 Create `crates/thresh-association/benches/mahalanobis.rs` with `mahalanobis_6d` benchmark
- [x] 1.5 Create `crates/thresh-tracker/benches/tracker_step.rs` with `tracker_step_10`, `tracker_step_50`, `tracker_step_200` benchmarks

## 2. Profiling

- [ ] 2.1 Generate flamegraphs for `tracker_step_200` using `cargo flamegraph` and identify top-5 hot spots
- [ ] 2.2 Document profiling results: which functions consume >5% of frame time, what the actual bottleneck ordering is

## 3. Hungarian optimization

- [x] 3.1 Implement `HungarianSolver` struct with pre-allocated flat `Vec<f64>` cost buffer and reusable marking arrays
- [x] 3.2 Implement `HungarianSolver::solve(&mut self, cost: &[Vec<f64>], gate: f64) -> AssignmentResult` reusing internal buffers
- [x] 3.3 Keep existing `hungarian_assignment` free function as a convenience wrapper around `HungarianSolver`
- [x] 3.4 Benchmark `HungarianSolver::solve` vs old `hungarian_assignment` at 100x100 and 500x500; verify improvement

## 4. Rayon parallelism

- [x] 4.1 Add `rayon` as an optional dependency in `crates/thresh-tracker/Cargo.toml` behind `parallel` feature gate
- [x] 4.2 Parallelize `predict_all` with `rayon::par_iter_mut` when `parallel` feature is enabled, with a threshold of 32 tracks
- [x] 4.3 Parallelize cost matrix row computation in `build_track_cost_matrix` with `rayon::par_iter` and collect
- [x] 4.4 Benchmark `tracker_step_200` with and without `parallel` feature; verify net improvement

## 5. Cache-friendly storage (conditional)

- [ ] 5.1 Analyze flamegraph for cache miss indicators in `predict_all` and `build_track_cost_matrix`
- [ ] 5.2 If warranted: prototype SoA track storage with states as `DMatrix` columns and covariances in contiguous `Vec<f64>`
- [ ] 5.3 If warranted: benchmark SoA vs AoS for `tracker_step_200`

## 6. CI performance gate

- [ ] 6.1 Add CI job that runs `cargo bench` on `develop` pushes and stores baseline results as artifacts
- [ ] 6.2 Add PR check that compares benchmark results against baseline and flags >10% regressions
- [ ] 6.3 Document benchmark methodology and regression thresholds in `docs/reference/`
