# Python Package Export via PyO3 — Design

## Context

thresh-bridge currently provides inbound PyO3 bridges that call Python libraries (Stone Soup, JSBSim) from Rust. There is no outbound binding that lets Python users call thresh's tracking pipeline. The ML and tracking research community works primarily in Python/Jupyter. Without Python bindings, thresh is inaccessible to this audience. The project already has PyO3 experience from thresh-bridge.

## Goals / Non-Goals

**Goals:**
- New `crates/thresh-py/` crate with `#[pymodule]` and `cdylib` crate-type
- Build with `maturin`; add `pyproject.toml` for pip-installable wheels
- Expose: `MultiObjectTracker` (step, get_tracks), `KalmanFilter` (predict, update), `compute_mot_metrics`, `run_scenario`
- numpy interop: accept `np.ndarray` as detection input, return tracks as structured numpy arrays
- Minimal API surface — wrap user-facing workflow, not every internal type
- Python pytest suite under `tests/python/`
- Pre-built wheels for Linux (manylinux), macOS (x86_64 + ARM), Windows

**Non-Goals:**
- Full Stone Soup compatibility or drop-in replacement API
- Real-time streaming from Python (batch API only for initial release)
- GUI, visualization, or plotting
- Exposing filter internals (matrix math, motion model traits)

## Decisions

### Crate structure

```
crates/thresh-py/
  Cargo.toml          # cdylib, pyo3 + numpy deps
  pyproject.toml      # maturin build config
  src/
    lib.rs            # #[pymodule] entry point
    tracker.rs        # PyMultiObjectTracker wrapper
    filter.rs         # PyKalmanFilter wrapper
    metrics.rs        # compute_mot_metrics binding
    scenario.rs       # run_scenario binding
    conversions.rs    # numpy <-> nalgebra conversion utilities
```

The `thresh-py` crate depends on `thresh-tracker`, `thresh-filter`, `thresh-eval`, and `thresh-synth`. It does NOT depend on `thresh-inference` or `thresh-bridge` to avoid pulling in ONNX Runtime or the inbound PyO3 bridges.

### Python API surface

```python
import thresh

# Simple API
tracker = thresh.MultiObjectTracker(
    measurement_noise=10.0,
    gate_threshold=50.0,
)
tracks = tracker.step(detections_np_array, dt=0.1)  # np.ndarray (N, 3) -> list[dict]

# Filter API
kf = thresh.KalmanFilter(state_dim=6, measurement_dim=3)
kf.predict(F, Q)
kf.update(z, H, R)
state = kf.state   # np.ndarray
covariance = kf.covariance  # np.ndarray

# Metrics
results = thresh.compute_mot_metrics(ground_truth, hypotheses)
# returns dict: {"mota": 0.85, "motp": 12.3, "idf1": 0.78, ...}

# Scenario runner
tracks = thresh.run_scenario(scenario_config_dict)
```

### numpy interop

Use the `numpy` crate (PyO3 companion) for zero-copy array I/O where possible. Conversions in `conversions.rs`:
- `ndarray_to_dvectors(arr: &PyArray2<f64>) -> Vec<DVector<f64>>`: each row becomes a detection vector
- `tracks_to_ndarray(tracks: &[Track]) -> PyArray2<f64>`: columns are [track_id, x, y, z, vx, vy, vz, confidence]
- `dmatrix_to_ndarray(m: &DMatrix<f64>) -> PyArray2<f64>`: for covariance/state matrix export

For track output, return a list of Python dicts (not a custom class) for maximum interop with pandas/matplotlib. Each dict: `{"id": int, "state": np.ndarray, "covariance": np.ndarray, "class": str, "status": str}`.

### Maturin build

`pyproject.toml` uses maturin as the build backend. The package name on PyPI will be `thresh`. Maturin handles:
- Compiling the Rust `cdylib`
- Packaging as a wheel with the correct platform tags
- `maturin develop` for local development installs

CI publishes wheels via `maturin build --release` with cross-compilation for:
- `manylinux_2_28` x86_64
- `macos-13` x86_64
- `macos-14` ARM64
- `windows-latest` x86_64

### Error handling

Rust `Result` types are converted to Python exceptions. Define a custom `thresh.ThreshError` exception class. Panics are caught by PyO3 and converted to `RuntimeError`. All public methods return Python-native types or raise exceptions; no Rust `Result` leaks to the Python side.

## Risks / Trade-offs

- **API stability**: The Python API is a public contract. Changes to internal Rust types (e.g., `Track` fields) require updating the binding layer. Mitigate by keeping the binding surface minimal and using dicts for output.
- **numpy version compatibility**: The `numpy` crate tracks numpy's C API. Major numpy releases (e.g., numpy 2.0) may require crate updates. Pin numpy crate version and test against numpy 1.x and 2.x.
- **Wheel size**: Statically linking thresh + nalgebra + dependencies produces wheels of 5-15 MB. Acceptable for a tracking library.
- **No streaming**: Batch-only API means Python users cannot do real-time tracking. This is intentional for the initial release; streaming bindings are a future extension after the streaming Rust API stabilizes.

## Open Questions

1. Should the Python package name be `thresh` or `thresh-py` on PyPI?
2. Should we expose `Detection3D` as a Python class, or keep it as a dict?
3. Should `run_scenario` accept a file path to a YAML config or a Python dict?
