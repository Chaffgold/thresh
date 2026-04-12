# Python Package Export via PyO3 — Tasks

## 1. Crate setup

- [ ] 1.1 Create `crates/thresh-py/Cargo.toml` with `crate-type = ["cdylib"]`, `pyo3` and `numpy` dependencies, and deps on `thresh-tracker`, `thresh-filter`, `thresh-eval`, `thresh-synth`
- [ ] 1.2 Create `crates/thresh-py/pyproject.toml` with maturin build backend, package name `thresh`, and minimum Python version 3.9
- [ ] 1.3 Create `crates/thresh-py/src/lib.rs` with `#[pymodule]` entry point registering submodules
- [ ] 1.4 Add `thresh-py` to workspace members in root `Cargo.toml`

## 2. Conversion utilities

- [ ] 2.1 Implement `crates/thresh-py/src/conversions.rs` with `ndarray_to_dvectors` for converting `np.ndarray` (N, M) to `Vec<DVector<f64>>`
- [ ] 2.2 Implement `tracks_to_list_of_dicts` converting `Vec<Track>` to Python list of dicts with id, state, covariance, class, status
- [ ] 2.3 Implement `dmatrix_to_pyarray` and `dvector_to_pyarray` for nalgebra-to-numpy conversion

## 3. Tracker binding

- [ ] 3.1 Implement `PyMultiObjectTracker` wrapper in `crates/thresh-py/src/tracker.rs` with `#[pyclass]`
- [ ] 3.2 Implement `PyMultiObjectTracker::new(measurement_noise, gate_threshold)` constructor
- [ ] 3.3 Implement `PyMultiObjectTracker::step(detections: PyArray2<f64>, dt: f64) -> Vec<PyDict>` method
- [ ] 3.4 Implement `PyMultiObjectTracker::confirmed_count()` and `alive_count()` properties

## 4. Filter binding

- [ ] 4.1 Implement `PyKalmanFilter` wrapper in `crates/thresh-py/src/filter.rs` with `#[pyclass]`
- [ ] 4.2 Implement `predict(F: PyArray2, Q: PyArray2)` and `update(z: PyArray1, H: PyArray2, R: PyArray2)` methods
- [ ] 4.3 Expose `state` and `covariance` as numpy array properties

## 5. Metrics and scenario bindings

- [ ] 5.1 Implement `compute_mot_metrics` function in `crates/thresh-py/src/metrics.rs` accepting ground truth and hypothesis arrays, returning a dict of metric values
- [ ] 5.2 Implement `run_scenario` function in `crates/thresh-py/src/scenario.rs` accepting a config dict and returning track results

## 6. Error handling

- [ ] 6.1 Define `ThreshError` custom Python exception class via `pyo3::create_exception!`
- [ ] 6.2 Implement `From<ThreshError>` for `PyErr` mapping Rust error variants to the custom exception

## 7. Testing and CI

- [ ] 7.1 Create `tests/python/test_tracker.py` with pytest tests: create tracker, step with numpy arrays, verify track output structure
- [ ] 7.2 Create `tests/python/test_filter.py` with pytest tests: KF predict/update cycle, verify state dimensions
- [ ] 7.3 Create `tests/python/test_metrics.py` with pytest tests: compute metrics on known ground truth, verify MOTA/MOTP values
- [ ] 7.4 Add CI job: `maturin develop && pytest tests/python/`
- [ ] 7.5 Add CI job: build wheels for Linux, macOS (x86_64 + ARM), Windows using `maturin build --release`
