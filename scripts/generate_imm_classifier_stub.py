#!/usr/bin/env python3
"""Generate a tiny synthetic ONNX stub for the Track B IMM mode classifier.

Input:  features   (1, 10, 12)  float32  — 10 consecutive 12-dim filter-state
                                            projections (state + cov diagonal).
Output: mode_probs (1, 4)       float32  — softmaxed probabilities over
                                            [CV, CA, CTRV, coord_turn].

Random weights; for shape-contract testing only (< 100 KB). The real checkpoint
is produced by ``python/training/train_imm.py`` + ``python/export/export_imm.py``
once trained, and replaces this stub. Mirrors ``generate_test_model.py``.
"""

from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

# ---------------------------------------------------------------------------
# Dimensions — must match thresh_filter::imm::CLASSIFIER_FEATURE_DIM (12),
# imm_adapter::WINDOW_LEN (10) and NUM_MODES (4).
# ---------------------------------------------------------------------------
BATCH = 1
WINDOW_LEN = 10
FEATURE_DIM = 12
NUM_MODES = 4

rng = np.random.default_rng(42)


def _rand_init(name: str, shape: tuple[int, ...]) -> onnx.TensorProto:
    """Create a random float32 initializer."""
    data = (rng.standard_normal(shape) * 0.1).astype(np.float32)
    return numpy_helper.from_array(data, name=name)


def build_model() -> onnx.ModelProto:
    # Linear head weights: (FEATURE_DIM, NUM_MODES) and bias (NUM_MODES,).
    w = _rand_init("w", (FEATURE_DIM, NUM_MODES))
    b = _rand_init("b", (NUM_MODES,))

    # pooled = mean over the time axis → (1, 12)
    pool = helper.make_node(
        "ReduceMean", ["features"], ["pooled"], axes=[1], keepdims=0
    )
    # logits = pooled @ w + b → (1, 4)
    matmul = helper.make_node("MatMul", ["pooled", "w"], ["mm"])
    add = helper.make_node("Add", ["mm", "b"], ["logits"])
    # mode_probs = softmax(logits) → (1, 4), sums to 1 per row
    softmax = helper.make_node("Softmax", ["logits"], ["mode_probs"], axis=1)

    input_info = helper.make_tensor_value_info(
        "features", TensorProto.FLOAT, [BATCH, WINDOW_LEN, FEATURE_DIM]
    )
    output_info = helper.make_tensor_value_info(
        "mode_probs", TensorProto.FLOAT, [BATCH, NUM_MODES]
    )

    graph = helper.make_graph(
        nodes=[pool, matmul, add, softmax],
        name="imm_mode_classifier_stub",
        inputs=[input_info],
        outputs=[output_info],
        initializer=[w, b],
    )

    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    return model


def main() -> None:
    out_dir = Path(__file__).resolve().parent.parent / "test-data" / "models"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "imm_mode_classifier.onnx"

    model = build_model()
    onnx.save(model, str(out_path))

    size_kb = out_path.stat().st_size / 1024
    print(f"Saved {out_path}  ({size_kb:.1f} KB)")
    assert size_kb < 100, f"Model too large: {size_kb:.1f} KB (limit 100 KB)"
    print("Model passes onnx.checker.check_model -- OK")


if __name__ == "__main__":
    main()
