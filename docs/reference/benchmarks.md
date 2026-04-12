# Benchmark Methodology

## What We Benchmark

The thresh workspace includes Criterion benchmarks covering the core computational hot paths:

- **Hungarian assignment** (`thresh-association`): 10x10, 100x100, and 500x500 cost matrices using seeded random inputs. Measures the assignment solver that dominates association cost.
- **Kalman filter predict/update** (`thresh-filter`): 6D state predict and update cycles. Measures the per-track state estimation cost.
- **Mahalanobis distance** (`thresh-association`): 6D gating computation used during association.
- **Tracker step** (`thresh-tracker`): Full predict-associate-update-lifecycle cycle with 10, 50, and 200 simultaneous tracks. End-to-end frame processing latency.

## How to Run

```sh
cargo bench --workspace
```

Individual crate benchmarks:

```sh
cargo bench -p thresh-association
cargo bench -p thresh-filter
cargo bench -p thresh-tracker
```

Criterion reports are written to `target/criterion/` with HTML output viewable in a browser.

## CI Integration

On pushes to `develop`, the `criterion-benchmarks` job in `.github/workflows/benchmarks.yml` runs `cargo bench --workspace -- --output-format bencher` and uploads the results as a build artifact. Results are retained for 90 days.

## Regression Policy

A performance regression exceeding **10%** on any benchmark flags a review. When investigating regressions:

1. Compare the Criterion report against the previous baseline (`target/criterion/`).
2. Profile with `cargo flamegraph` on the regressing benchmark.
3. Document the root cause and whether the regression is acceptable (e.g., correctness fix) or needs optimization.
