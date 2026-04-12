# Performance Profiling and Optimization — Design

## Context

No profiling or benchmarking has been done on the codebase. The current implementation prioritizes correctness. The Hungarian algorithm in `thresh-association/src/hungarian.rs` is O(n^3) using `Vec<Vec<f64>>` (row-major, heap-allocated per row). The tracker in `thresh-tracker/src/tracker.rs` processes tracks sequentially in `step()`: predict all, build cost matrix, Hungarian, update matched, lifecycle, birth. For 100+ targets with dense clutter, the per-frame cost is dominated by the O(n^3) assignment and the sequential predict/update loops. nalgebra supports SIMD but it is not explicitly leveraged. No parallelism exists.

## Goals / Non-Goals

**Goals:**
- Add `criterion` benchmarks for hot paths: Hungarian assignment, KF predict/update, Mahalanobis gating, full tracker step
- Profile with `cargo-flamegraph` to identify actual bottlenecks before optimizing
- Reduce allocation in Hungarian per-step (reuse buffers)
- Add `rayon` parallelism for independent track predictions behind a `parallel` feature gate
- Evaluate and implement cache-friendly SoA layout for track states if profiling warrants it
- Establish performance CI gate to catch regressions

**Non-Goals:**
- GPU offload (CUDA, Metal)
- Distributed processing across machines
- Algorithm replacement (e.g., auction algorithm instead of Hungarian)
- Optimizing ONNX inference (belongs to thresh-inference)

## Decisions

### Benchmark suite with criterion

Add `criterion` as a workspace dev-dependency. Benchmark targets:

| Benchmark | Crate | What it measures |
|---|---|---|
| `hungarian_10x10` | thresh-association | Baseline small assignment |
| `hungarian_100x100` | thresh-association | Medium realistic scenario |
| `hungarian_500x500` | thresh-association | Stress test |
| `kf_predict_6d` | thresh-filter | Single KF predict on 6D state |
| `kf_update_6d` | thresh-filter | Single KF update on 6D state |
| `mahalanobis_6d` | thresh-association | Single Mahalanobis distance |
| `tracker_step_10` | thresh-tracker | Full step with 10 targets |
| `tracker_step_50` | thresh-tracker | Full step with 50 targets |
| `tracker_step_200` | thresh-tracker | Full step with 200 targets |

Each benchmark uses deterministic synthetic data (seeded RNG) for reproducibility.

### Profile-first approach

Before any optimization, generate flamegraphs for `tracker_step_200` to identify where time actually goes. Document findings in a profiling report. Only optimize paths that account for >5% of total frame time.

### Hungarian allocation reduction

The current `build_square_cost` allocates a new `Vec<Vec<f64>>` every call. Introduce a `HungarianSolver` struct that owns reusable buffers:

```rust
pub struct HungarianSolver {
    dim: usize,
    cost: Vec<f64>,       // flat dim*dim, row-major
    row_assign: Vec<Option<usize>>,
    col_assign: Vec<Option<usize>>,
    // marking arrays
    row_covered: Vec<bool>,
    col_covered: Vec<bool>,
    visited_col: Vec<bool>,
    parent_row: Vec<Option<usize>>,
}
```

The existing `hungarian_assignment` free function remains as a convenience wrapper that creates a temporary solver. The `HungarianSolver::solve(&mut self, cost: &[Vec<f64>], gate: f64) -> AssignmentResult` method reuses internal buffers across calls. The flat `Vec<f64>` layout improves cache locality over `Vec<Vec<f64>>`.

### Rayon parallelism

Add `rayon` as an optional dependency behind a `parallel` feature gate on thresh-tracker. Parallelize:
- `predict_all`: each track's predict is independent (reads shared F/Q, writes own state)
- Cost matrix row computation: each row (track) can compute Mahalanobis distances to all detections independently

Do NOT parallelize Hungarian (inherently sequential) or the update step (needs mutable access to track vec by index). Use `rayon::par_iter_mut` for predict and `rayon::par_iter` with `map`/`collect` for cost matrix rows.

The `parallel` feature is off by default. When enabled, `step()` automatically uses parallel iterators. No API change.

### Cache-friendly track storage (conditional)

Currently tracks are `Vec<Track>` where each `Track` contains a `DVector<f64>` (heap-allocated state) and a `DMatrix<f64>` (heap-allocated covariance). For 200 tracks, iterating over predictions causes 200 pointer chases for states + 200 for covariances.

SoA alternative: store all states as columns of a single `DMatrix<f64>` and all covariances in a contiguous `Vec<f64>` block. This is a significant refactor. Decision: only pursue if flamegraph shows cache misses in predict_all account for >10% of frame time. Otherwise, the simpler per-track layout is kept.

### Performance CI gate

Add a CI job that runs `criterion` benchmarks and compares against a baseline. Use `critcmp` or criterion's built-in comparison. Flag PRs that regress any benchmark by >10%. Store baseline results as a CI artifact on `develop`.

## Risks / Trade-offs

- **Rayon overhead for small N**: Thread pool overhead can make parallel predict slower than sequential for <20 tracks. Add a threshold: only use par_iter when `tracks.len() > 32`.
- **SoA refactor scope**: Converting from AoS to SoA touches every track access site. High risk of introducing bugs. Only pursue with strong profiling evidence.
- **Benchmark stability in CI**: Criterion results vary with CI runner load. Use relative comparisons and generous regression thresholds (10%) to avoid false positives.
- **Flat cost matrix breaks API**: Changing `Vec<Vec<f64>>` to flat `Vec<f64>` in Hungarian requires updating all callers that build cost matrices. Mitigate by keeping the `Vec<Vec<f64>>` public API and converting internally.

## Open Questions

1. Should the `parallel` feature be workspace-wide or per-crate?
2. What is the minimum track count where rayon parallelism provides net benefit on typical hardware?
3. Should we add `iai` (instruction-count-based) benchmarks for deterministic CI, or is criterion sufficient?
