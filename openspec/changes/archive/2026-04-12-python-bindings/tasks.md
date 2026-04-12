# Python Package Export via PyO3 — Tasks

## 1. Crate setup

- [x] 1.1 Create `crates/thresh-py/Cargo.toml` with `crate-type = ["cdylib"]`, `pyo3` and `numpy` dependencies, and deps on `thresh-tracker`, `thresh-filter`, `thresh-eval`, `thresh-synth`
- [x] 1.2 Create `crates/thresh-py/pyproject.toml` with maturin build backend, package name `thresh`, and minimum Python version 3.9
- [x] 1.3 Create `crates/thresh-py/src/lib.rs` with `#[pymodule]` entry point registering submodules
- [x] 1.4 Add `thresh-py` to workspace members in root `Cargo.toml`

## 2. Conversion utilities

- [x] 2.1 Implement `crates/thresh-py/src/conversions.rs` with `lists_to_dvectors`, `dvector_to_list`, `dmatrix_to_lists`, `lists_to_dmatrix` conversion utilities
- [x] 2.2 Implement roundtrip-tested conversion functions between Python list types and nalgebra types
- [x] 2.3 Implement `dmatrix_to_lists` and `dvector_to_list` for nalgebra-to-Python conversion

## 3. Tracker binding

- [x] 3.1 Implement `PyMultiObjectTracker` wrapper in `crates/thresh-py/src/tracker.rs` with `#[pyclass]`
- [x] 3.2 Implement `PyMultiObjectTracker::new(measurement_noise, gate_threshold)` constructor
- [x] 3.3 Implement `PyMultiObjectTracker::step(detections: PyArray2<f64>, dt: f64) -> Vec<PyDict>` method
- [x] 3.4 Implement `PyMultiObjectTracker::confirmed_count()` and `alive_count()` properties

## 4. Filter binding

- [x] 4.1 Implement `PyKalmanFilter` wrapper in `crates/thresh-py/src/filter.rs` with `#[pyclass]`
- [x] 4.2 Implement `predict(f, q)` and `update(z, h, r)` methods with dimension validation
- [x] 4.3 Expose `state` and `covariance` as Python-accessible getter properties

## 5. Metrics and scenario bindings

- [x] 5.1 Implement `compute_mot_metrics` function in `crates/thresh-py/src/eval.rs` accepting ground truth and hypothesis arrays, returning (mota, motp, id_switches) tuple
- [x] 5.2 Implement `run_scenario` placeholder function in `crates/thresh-py/src/scenario.rs` (returns ThreshError directing to CLI)

## 6. Error handling

- [x] 6.1 Define `ThreshError` custom Python exception class via `pyo3::create_exception!` in `errors.rs`
- [x] 6.2 Implement `thresh_err()` helper for converting string errors to `ThreshError`-wrapped `PyErr`; used in filter.rs and scenario.rs

## 7. Testing and CI

- [x] 7.1 Create `tests/python/test_tracker.py` with pytest tests: create tracker, step with numpy arrays, verify track output structure
- [x] 7.2 Create `tests/python/test_filter.py` with pytest tests: KF predict/update cycle, verify state dimensions
- [x] 7.3 Create `tests/python/test_metrics.py` with pytest tests: compute metrics on known ground truth, verify MOTA/MOTP values
- [x] 7.4 Add CI job: `maturin develop && pytest tests/python/`
- [x] 7.5 Add CI job: build wheels for Linux, macOS (x86_64 + ARM), Windows using `maturin build --release`
