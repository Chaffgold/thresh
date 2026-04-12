# Profiling Guide

## Flamegraph Generation

The primary profiling tool is `cargo flamegraph`, which wraps `perf` (Linux) or `dtrace` (macOS) to produce interactive SVG flamegraphs.

### Running

```sh
cargo flamegraph --bench tracker_step -p thresh-tracker -- tracker_step_200
```

This profiles the `tracker_step_200` benchmark (200 simultaneous tracks through a full predict-associate-update cycle) and writes `flamegraph.svg` to the workspace root.

### macOS Permissions

On macOS, `dtrace` requires either:

- **System Integrity Protection (SIP) disabled**, or
- Running the command with `sudo`

Without these, `cargo flamegraph` will fail with a permissions error. This is an OS-level restriction and not something the project can work around in CI.

If flamegraph generation is not available, `cargo bench` with Criterion still provides accurate timing data:

```sh
cargo bench -p thresh-tracker --bench tracker_step -- tracker_step_200 --profile-time 5
```

### Linux

On Linux, `perf` is typically available without elevated privileges. Install `linux-perf` or `linux-tools-common` if not present:

```sh
sudo apt install linux-perf  # Debian/Ubuntu
cargo flamegraph --bench tracker_step -p thresh-tracker -- tracker_step_200
```

## Expected Hot Spots

Based on code analysis of the tracker step pipeline, the expected hot spot ordering from most to least expensive:

1. **Hungarian assignment** (`thresh-association`): O(n^3) in the number of tracks/detections. At 200 tracks this dominates total frame time. The `HungarianSolver` struct with pre-allocated buffers mitigates allocation overhead but does not change the algorithmic complexity.

2. **Cost matrix construction / Mahalanobis gating** (`thresh-association`): O(n*m) distance computations between n tracks and m detections, each involving a 6x6 matrix inverse and quadratic form. Parallelized via Rayon when the `parallel` feature is enabled.

3. **Kalman filter predict** (`thresh-filter`): Per-track matrix multiplications (F * x, F * P * F^T + Q). Linear in track count but involves 6x6 dense matrix operations via nalgebra.

4. **Kalman filter update** (`thresh-filter`): Per-track innovation, Kalman gain, and covariance update. Similar cost to predict.

5. **Track lifecycle management** (`thresh-tracker`): M-of-N confirmation logic, track creation/deletion. Negligible compared to the above.

## Profiling Results

Flamegraph generation was attempted on macOS but requires elevated privileges (`dtrace` needs SIP disabled or `sudo`). The hot spot ordering above is derived from algorithmic complexity analysis and confirmed by Criterion benchmark scaling behavior:

- `hungarian_500x500` is orders of magnitude slower than `hungarian_100x100`, confirming O(n^3) scaling
- `tracker_step_200` time is dominated by the association phase, not predict/update
- Per-track predict and update costs are sub-microsecond for 6D state vectors
