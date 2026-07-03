"""Export the trained Track A detector to ONNX and verify it (task 7.6).

Wraps the model with the inference head ([`DetectorExportWrapper`]) and
`torch.onnx.export`s the three-output contract — `boxes` ``(batch, 100, 7)``,
`scores` ``(batch, 100, 1)`` in [0, 1], `classes` ``(batch, 100, 1)`` int64 —
then runs it under onnxruntime to assert shapes and value ranges. The result
replaces the random-weight `test_detector.onnx` stub once the exit criterion
(task 7.9) is met.

Requires the ``training`` optional extra (torch, onnx, onnxruntime).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from training.detector_dataset import BOX_DIM, MAX_BOXES, NUM_CLASSES, NUM_POINTS, POINT_DIM
from training.detector_model import DetectorExportWrapper, DetectorModel


def export(checkpoint: Path, out_path: Path, *, opset: int = 17) -> None:
    """Export ``checkpoint`` to a three-output ONNX detector at ``out_path``."""
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=True)
    core = DetectorModel(d_model=int(ckpt.get("d_model", 64)))
    core.load_state_dict(ckpt["state_dict"])
    core.eval()
    model = DetectorExportWrapper(core)
    model.eval()

    dummy = torch.zeros(1, NUM_POINTS, POINT_DIM, dtype=torch.float32)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        dummy,
        str(out_path),
        input_names=["point_cloud"],
        output_names=["boxes", "scores", "classes"],
        dynamic_axes={
            "point_cloud": {0: "batch"},
            "boxes": {0: "batch"},
            "scores": {0: "batch"},
            "classes": {0: "batch"},
        },
        opset_version=opset,
        # Legacy exporter: supports dynamic_axes, avoids the onnxscript dep.
        dynamo=False,
    )
    print(f"exported detector ONNX → {out_path}")


def verify(onnx_path: Path, *, batch: int = 2) -> None:
    """Run the exported detector under onnxruntime and assert the contract."""
    import onnxruntime as ort  # local import: only needed at verification time

    rng = np.random.default_rng(0)
    fixture = rng.standard_normal((batch, NUM_POINTS, POINT_DIM)).astype(np.float32)
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    boxes, scores, classes = sess.run(None, {"point_cloud": fixture})

    assert boxes.shape == (batch, MAX_BOXES, BOX_DIM), f"boxes {boxes.shape}"
    assert scores.shape == (batch, MAX_BOXES, 1), f"scores {scores.shape}"
    assert classes.shape == (batch, MAX_BOXES, 1), f"classes {classes.shape}"
    assert (scores >= 0.0).all() and (scores <= 1.0).all(), "scores must be in [0, 1]"
    assert (classes >= 0).all() and (classes < NUM_CLASSES).all(), "classes in [0, NUM_CLASSES)"
    print(
        f"verified {onnx_path}: boxes {boxes.shape}, scores in [0,1], classes in [0,{NUM_CLASSES})"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Export + verify the Track A detector ONNX.")
    parser.add_argument("--checkpoint", required=True, type=Path, help="trained .pt checkpoint")
    parser.add_argument("--out", required=True, type=Path, help="output .onnx path")
    parser.add_argument("--opset", type=int, default=17)
    args = parser.parse_args()

    export(args.checkpoint, args.out, opset=args.opset)
    verify(args.out)


if __name__ == "__main__":
    main()
