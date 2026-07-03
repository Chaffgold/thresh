"""IMM classifier holdout accuracy: exported ONNX vs analytic labels (task 6.8).

Runs the exported classifier over windows built from an *independent*
holdout dataset (e.g. a different capture region than training, mirroring
holdout_split's region discipline) and reports overall accuracy plus
per-class recall — the overall number satisfies the 6.8 ≥ 0.70 gate, the
per-class breakdown keeps a majority-class-only classifier honest.

    uv run --extra training python -m eval.imm_accuracy \\
        --data ../python/data-training/imm-holdout.parquet \\
        --model ../test-data/models/imm_mode_classifier.onnx
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from training.imm_dataset import build_windows

MODE_NAMES = ("cv", "ca", "ctrv", "coord_turn")
"""Label names index-aligned to the `cv_ca_ctrv_ct` IMM bank (label_index 0..4)."""


def evaluate_onnx(model_path: Path, x: np.ndarray, y: np.ndarray) -> dict[str, object]:
    """Accuracy + per-class recall of the ONNX classifier on ``(x, y)``."""
    import onnxruntime as ort  # heavy import kept local (training extra)

    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    probs = np.asarray(session.run(None, {input_name: x.astype(np.float32)})[0])
    preds = probs.argmax(axis=1)

    per_class: dict[str, dict[str, float | int]] = {}
    for cls, name in enumerate(MODE_NAMES):
        mask = y == cls
        support = int(mask.sum())
        recall = float((preds[mask] == cls).mean()) if support else float("nan")
        per_class[name] = {"support": support, "recall": recall}

    return {
        "accuracy": float((preds == y).mean()),
        "n_windows": int(y.shape[0]),
        "per_class": per_class,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="IMM classifier holdout accuracy (task 6.8).")
    parser.add_argument("--data", required=True, type=Path, help="holdout imm-samples Parquet")
    parser.add_argument("--model", required=True, type=Path, help="classifier ONNX path")
    args = parser.parse_args()

    windows = build_windows(args.data)
    if len(windows) == 0:
        raise SystemExit(f"no windows built from {args.data}")
    print(json.dumps(evaluate_onnx(args.model, windows.x, windows.y), indent=2), flush=True)


if __name__ == "__main__":
    main()
