//! Run an ONNX model on a deterministic fixture and print its output as JSON
//! (flight-data-training-pipeline, Phase 8.3).
//!
//! Used by `python/eval/onnx_parity.py` to assert the Rust (`thresh-inference`
//! / `ort`) and Python (`onnxruntime`) runtimes agree on the same input. Built
//! behind the `onnx` feature.
//!
//! ```sh
//! cargo run -p thresh --features onnx --bin onnx-infer -- \
//!   test-data/models/imm_mode_classifier.onnx 1 10 12
//! ```
//!
//! Args: `<model.onnx> <dim0> <dim1> ...` (the full input shape). The input
//! tensor is filled with the **same deterministic ramp** the Python side uses:
//! `x[i] = sin(i * 0.1)`, row-major over the flattened shape. Output is printed
//! as `{"output":[...]}` (the first output, flattened f32).

use thresh::inference::session::OnnxModel;

/// Deterministic fixture value at flat index `i` — must match `onnx_parity.py`.
fn fixture_value(i: usize) -> f32 {
    (i as f32 * 0.1).sin()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("usage: onnx-infer <model.onnx> <dim0> [dim1 ...]");
        std::process::exit(2);
    }
    let model_path = &args[1];
    let shape: Vec<i64> = args[2..]
        .iter()
        .map(|s| s.parse::<i64>())
        .collect::<Result<_, _>>()?;
    let numel: i64 = shape.iter().product();

    let input: Vec<f32> = (0..numel as usize).map(fixture_value).collect();

    let mut model = OnnxModel::load(model_path)?;
    let output = model.run_f32(&input, &shape)?;

    let body = output
        .iter()
        .map(|v| format!("{v:.8}"))
        .collect::<Vec<_>>()
        .join(",");
    println!("{{\"output\":[{body}]}}");
    Ok(())
}
