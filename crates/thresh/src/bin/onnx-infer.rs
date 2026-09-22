//! Run an ONNX model on a deterministic fixture and print its output as JSON
//! (flight-data-training-pipeline, Phase 8.3).
//!
//! Used by `python/eval/onnx_parity.py` to assert the Rust (`thresh-inference`
//! / `ort`) and Python (`onnxruntime`) runtimes agree on the same input. Built
//! behind the `onnx` feature.
//!
//! ```sh
//! cargo run -p thresh --features onnx --bin onnx-infer -- \
//!   test-data/models/imm_mode_classifier.onnx 1 10 13
//! ```
//!
//! Args: `<model.onnx> <dim0> <dim1> ...` (the full input shape). The input
//! tensor is filled with the **same deterministic ramp** the Python side uses:
//! `x[i] = sin(i * 0.1)`, row-major over the flattened shape. For the IMM
//! `(batch, 10, 13)` contract the final column is elapsed seconds (default 0.1,
//! override with `--elapsed-seconds <dt>`), starting with zero per batch.
//! Output is printed
//! as `{"output":[...]}` (the first output, flattened f32).

use thresh::inference::session::OnnxModel;

/// Deterministic fixture value at flat index `i` — must match `onnx_parity.py`
/// exactly. The sine is computed in **f64** and then cast to f32, mirroring
/// Python's `float32(math.sin(i * 0.1))`, so the two runtimes see identical
/// inputs (an f32 sine could round differently and cause spurious mismatches).
fn fixture_value(i: usize) -> f32 {
    (i as f64 * 0.1).sin() as f32
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("usage: onnx-infer <model.onnx> <dim0> [dim1 ...]");
        std::process::exit(2);
    }
    let model_path = &args[1];
    let timing_index = args.iter().position(|arg| arg == "--elapsed-seconds");
    let shape_end = timing_index.unwrap_or(args.len());
    let shape: Vec<i64> = args[2..shape_end]
        .iter()
        .map(|s| s.parse::<i64>())
        .collect::<Result<_, _>>()?;
    let numel: i64 = shape.iter().product();

    let elapsed = match timing_index {
        Some(index) => args
            .get(index + 1)
            .ok_or("missing elapsed seconds")?
            .parse::<f32>()?,
        None => 0.1,
    };
    if !elapsed.is_finite() || elapsed < 0.0 {
        return Err("elapsed seconds must be finite and nonnegative".into());
    }
    let mut input: Vec<f32> = (0..numel as usize).map(fixture_value).collect();
    if shape.len() == 3 && shape[1..] == [10, 13] {
        for (row_index, row) in input.as_chunks_mut::<13>().0.iter_mut().enumerate() {
            row[12] = if row_index % 10 == 0 { 0.0 } else { elapsed };
        }
    }

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
