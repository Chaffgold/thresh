# Python Package Export via PyO3

## What

Expose the tracker, filter, and evaluation modules as a Python package (`thresh-py`) so researchers can call `thresh.track(detections)` from Jupyter notebooks and Python scripts without writing Rust. This provides an outbound Rust-to-Python API, complementing the existing inbound PyO3 bridges that call Python libraries from Rust.

## Why

The existing PyO3 bridges in thresh-bridge are inbound: they call Stone Soup, JSBSim, and nuScenes from Rust. An outbound binding that exposes thresh's tracking pipeline as a Python-callable library would make the framework accessible to the ML and computer vision research community that works primarily in Python. Researchers evaluating tracking algorithms typically prototype in Python/Jupyter. Without Python bindings, thresh is invisible to this community regardless of its technical merits.

## How

- Create a new `thresh-py` crate with `#[pymodule]` exposing the core tracking API: `MultiObjectTracker`, filter types (KF, EKF, UKF), evaluation metrics, and the scenario runner
- Use the `numpy` crate for zero-copy array I/O so detection matrices and track state arrays pass between Python and Rust without serialization overhead
- Expose a high-level `thresh.track(detections, timestamps)` function for simple use cases alongside the full builder API for advanced configuration
- Build and publish as a pip-installable wheel via maturin, with pre-built wheels for Linux (manylinux), macOS (x86_64 + ARM), and Windows
- Add Python integration tests using pytest that exercise the full tracking pipeline from numpy arrays through to MOT metric output

## Out of scope

- Full Stone Soup compatibility layer or drop-in replacement API
- Real-time streaming from Python (batch API only for initial release)
- GUI, visualization, or plotting (users bring their own matplotlib)
- Exposing internal filter math (only the tracking pipeline and metrics APIs)

## Affected crates

- thresh-py (new crate): PyO3 module, numpy interop, wheel packaging via maturin
- thresh-tracker: public API stabilization and documentation for binding surface
- thresh-eval: Python-friendly metric computation API returning dicts/numpy arrays
