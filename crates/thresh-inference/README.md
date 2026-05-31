# thresh-inference

Detection front-end for the thresh tracking stack: a `DetectionPipeline` trait,
3D NMS + confidence filtering, and detector backends (a pure-Rust SafeTensors
DETR decoder and an ONNX Runtime path).

## `onnx` feature (off by default)

Enables the `ort`-backed `session::OnnxModel` runner and
`detection::OnnxDetector`. `OnnxDetector::detect` loads a `.onnx` detector and
decodes the three-output contract into `Detection3D`s, then confidence-filters
and runs NMS:

- `boxes (1, 100, 7)` f32 `[x, y, z, L, W, H, yaw]`
- `scores (1, 100, 1)` f32
- `classes (1, 100, 1)` int64 — optional; a 2-output model decodes with all
  `class_id = 0` (backward compatible).

```sh
cargo test -p thresh-inference --features onnx   # auto-downloads ORT binaries
```

The `ort` / ONNX-Runtime dependency is heavy (auto-downloads ~200 MB of
binaries), so `onnx` is opt-in and its tests run in a dedicated, path-filtered
CI job rather than the default build. The committed `test-data/models/*.onnx`
are random-weight shape-contract stubs — see `test-data/models/MODEL_CARD.md`.
