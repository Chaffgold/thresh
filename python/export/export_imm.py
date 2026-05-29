"""Export a trained IMM classifier to ONNX and verify it (tasks 6.4, 6.5).

Loads a checkpoint, wraps it with a softmax head (so the deployed graph emits a
probability distribution), and ``torch.onnx.export``s it with input
``(batch, WINDOW_LEN, FEATURE_DIM)`` = ``(batch, 10, 12)`` and output
``(batch, NUM_MODES)`` = ``(batch, 4)``. Then runs the exported model under
onnxruntime on a fixture batch and asserts the outputs are valid probabilities
(each row sums to 1).

The resulting ``imm_mode_classifier.onnx`` is what the Rust
``thresh_filter::imm_adapter::ImmModeAdapter`` loads at runtime; it replaces the
checked-in random-weight stub once a real checkpoint meets the exit criterion.

Requires the ``training`` optional extra (torch, onnx, onnxruntime).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from training.imm_dataset import FEATURE_DIM, NUM_MODES, WINDOW_LEN
from training.imm_model import ImmModeClassifier, SoftmaxClassifier


def export(checkpoint: Path, out_path: Path, *, opset: int = 17) -> None:
    """Export ``checkpoint`` to ONNX at ``out_path`` with a softmax head."""
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=True)
    core = ImmModeClassifier(hidden_dim=int(ckpt.get("hidden_dim", 64)))
    core.load_state_dict(ckpt["state_dict"])
    core.eval()
    model = SoftmaxClassifier(core)
    model.eval()

    dummy = torch.zeros(1, WINDOW_LEN, FEATURE_DIM, dtype=torch.float32)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        dummy,
        str(out_path),
        input_names=["features"],
        output_names=["mode_probs"],
        dynamic_axes={"features": {0: "batch"}, "mode_probs": {0: "batch"}},
        opset_version=opset,
        # Use the legacy TorchScript exporter: it supports `dynamic_axes` and
        # avoids the dynamo exporter's `onnxscript` dependency.
        dynamo=False,
    )
    print(f"exported ONNX → {out_path}")


def verify(onnx_path: Path, *, batch: int = 3) -> None:
    """Run the exported model under onnxruntime and assert valid probabilities."""
    import onnxruntime as ort  # local import: only needed at verification time

    rng = np.random.default_rng(0)
    fixture = rng.standard_normal((batch, WINDOW_LEN, FEATURE_DIM)).astype(np.float32)
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    (out,) = sess.run(None, {"features": fixture})

    assert out.shape == (batch, NUM_MODES), f"unexpected output shape {out.shape}"
    row_sums = out.sum(axis=1)
    assert np.allclose(row_sums, 1.0, atol=1e-5), f"rows must sum to 1, got {row_sums}"
    assert (out >= 0.0).all() and (out <= 1.0).all(), "probabilities must be in [0, 1]"
    print(f"verified {onnx_path}: output {out.shape}, rows sum to 1.0 (±1e-5)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export + verify the IMM classifier ONNX.")
    parser.add_argument("--checkpoint", required=True, type=Path, help="Trained .pt checkpoint")
    parser.add_argument("--out", required=True, type=Path, help="Output .onnx path")
    parser.add_argument("--opset", type=int, default=17)
    args = parser.parse_args()

    export(args.checkpoint, args.out, opset=args.opset)
    verify(args.out)


if __name__ == "__main__":
    main()
