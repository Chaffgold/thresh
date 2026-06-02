"""Cross-language ONNX parity check (flight-data-training-pipeline, task 8.3).

Runs the same deterministic fixture through the **Python** runtime
(`onnxruntime`) and the **Rust** runtime (`thresh-inference` / `ort`, via the
`onnx-infer` binary) and asserts the outputs match within a tolerance. This
guards against the two runtimes silently diverging on the IMM-classifier
contract (the IMM stub has a single `(1, 10, 12) → (1, 4)` output, so the
parity check is unambiguous).

The fixture is ``x[i] = sin(i * 0.1)`` row-major over the input shape — the same
ramp `crates/thresh/src/bin/onnx-infer.rs::fixture_value` fills. Requires the
``training`` extra (onnxruntime); the Rust side needs ``--features onnx``.

    uv run --extra training python -m eval.onnx_parity
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = REPO_ROOT / "test-data" / "models" / "imm_mode_classifier.onnx"
DEFAULT_SHAPE = (1, 10, 12)
TOLERANCE = 1e-5


def fixture(shape: tuple[int, ...]) -> np.ndarray:
    """Deterministic input ramp, identical to the Rust `fixture_value`."""
    numel = math.prod(shape)
    flat = np.array([math.sin(i * 0.1) for i in range(numel)], dtype=np.float32)
    return flat.reshape(shape)


def python_output(model_path: Path, shape: tuple[int, ...]) -> np.ndarray:
    """Run the model under onnxruntime; return the first output, flattened."""
    import onnxruntime as ort

    sess = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    (out,) = sess.run(None, {input_name: fixture(shape)})
    return np.asarray(out, dtype=np.float32).reshape(-1)


def rust_output(
    model_path: Path, shape: tuple[int, ...], *, repo_root: Path = REPO_ROOT
) -> np.ndarray:
    """Run the model via the Rust `onnx-infer` binary; return its output array."""
    cmd = [
        "cargo", "run", "-q", "-p", "thresh", "--features", "onnx", "--bin", "onnx-infer",
        "--", str(model_path), *[str(d) for d in shape],
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=repo_root)
    if proc.returncode != 0:
        raise RuntimeError(
            f"onnx-infer exited {proc.returncode}\n--- stdout ---\n{proc.stdout}\n"
            f"--- stderr ---\n{proc.stderr}"
        )
    line = next((ln for ln in proc.stdout.splitlines() if ln.strip().startswith("{")), None)
    if line is None:
        raise RuntimeError(
            f"onnx-infer produced no JSON output line\n--- stdout ---\n{proc.stdout}\n"
            f"--- stderr ---\n{proc.stderr}"
        )
    return np.asarray(json.loads(line)["output"], dtype=np.float32)


def check_parity(
    model_path: Path, shape: tuple[int, ...], *, tolerance: float = TOLERANCE
) -> float:
    """Assert Python and Rust outputs agree within `tolerance`. Returns the max
    absolute difference."""
    py = python_output(model_path, shape)
    rs = rust_output(model_path, shape)
    if py.shape != rs.shape:
        raise AssertionError(f"output length differs: python {py.shape} vs rust {rs.shape}")
    max_diff = float(np.max(np.abs(py - rs))) if py.size else 0.0
    if max_diff > tolerance:
        raise AssertionError(
            f"ONNX parity FAILED: max |python - rust| = {max_diff:.2e} > {tolerance:.0e}\n"
            f"  python: {py}\n  rust:   {rs}"
        )
    return max_diff


def main() -> None:
    parser = argparse.ArgumentParser(description="Python/Rust ONNX output parity check.")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--shape", type=int, nargs="+", default=list(DEFAULT_SHAPE))
    parser.add_argument("--tolerance", type=float, default=TOLERANCE)
    args = parser.parse_args()

    max_diff = check_parity(args.model, tuple(args.shape), tolerance=args.tolerance)
    print(f"ONNX parity OK: max |python - rust| = {max_diff:.2e} (tol {args.tolerance:.0e})")


if __name__ == "__main__":
    main()
