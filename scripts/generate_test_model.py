#!/usr/bin/env python3
"""Generate a tiny synthetic ONNX test model mimicking an RT-DETR-like detection head.

Input:  point_cloud  (1, 1000, 4)  float32  — batch of 1000 points with [x, y, z, intensity]
Outputs:
  boxes   (1, 100, 7)  float32  — [x, y, z, length, width, height, yaw] per detection
  scores  (1, 100, 1)  float32  — confidence score per detection

The model uses random weights and is intended for shape-testing only (< 1 MB).
"""

from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

# ---------------------------------------------------------------------------
# Dimensions
# ---------------------------------------------------------------------------
BATCH = 1
NUM_POINTS = 1000
POINT_DIM = 4       # x, y, z, intensity
HIDDEN_DIM = 32     # small hidden layer
NUM_DETECTIONS = 100
BOX_DIM = 7         # x, y, z, length, width, height, yaw
SCORE_DIM = 1

rng = np.random.default_rng(42)


def _rand_init(name: str, shape: tuple[int, ...]) -> onnx.TensorProto:
    """Create a random float32 initializer."""
    data = (rng.standard_normal(shape) * 0.01).astype(np.float32)
    return numpy_helper.from_array(data, name=name)


def build_model() -> onnx.ModelProto:
    # --- Initializers (weights) ---------------------------------------------------
    # Linear 1: (POINT_DIM, HIDDEN_DIM)  — applied across last dim of input
    w1 = _rand_init("w1", (POINT_DIM, HIDDEN_DIM))
    b1 = _rand_init("b1", (HIDDEN_DIM,))

    # Linear 2 for boxes:  (HIDDEN_DIM, BOX_DIM)
    w2_box = _rand_init("w2_box", (HIDDEN_DIM, BOX_DIM))
    b2_box = _rand_init("b2_box", (BOX_DIM,))

    # Linear 2 for scores: (HIDDEN_DIM, SCORE_DIM)
    w2_score = _rand_init("w2_score", (HIDDEN_DIM, SCORE_DIM))
    b2_score = _rand_init("b2_score", (SCORE_DIM,))

    # --- Graph nodes --------------------------------------------------------------
    # hidden = relu(input @ w1 + b1)   shape: (1, 1000, 32)
    matmul1 = helper.make_node("MatMul", ["point_cloud", "w1"], ["mm1"])
    add1 = helper.make_node("Add", ["mm1", "b1"], ["pre_relu"])
    relu1 = helper.make_node("Relu", ["pre_relu"], ["hidden"])

    # boxes_raw = hidden @ w2_box + b2_box   shape: (1, 1000, 7)
    matmul_box = helper.make_node("MatMul", ["hidden", "w2_box"], ["mm_box"])
    add_box = helper.make_node("Add", ["mm_box", "b2_box"], ["boxes_full"])

    # scores_raw = sigmoid(hidden @ w2_score + b2_score)  shape: (1, 1000, 1)
    matmul_score = helper.make_node("MatMul", ["hidden", "w2_score"], ["mm_score"])
    add_score = helper.make_node("Add", ["mm_score", "b2_score"], ["scores_pre_sig"])
    sigmoid_score = helper.make_node("Sigmoid", ["scores_pre_sig"], ["scores_full"])

    # Slice to first NUM_DETECTIONS along axis=1 to get (1, 100, *)
    # axes, starts, ends, steps as constant tensors
    axes_init = numpy_helper.from_array(np.array([1], dtype=np.int64), "slice_axes")
    starts_init = numpy_helper.from_array(np.array([0], dtype=np.int64), "slice_starts")
    ends_init = numpy_helper.from_array(
        np.array([NUM_DETECTIONS], dtype=np.int64), "slice_ends"
    )

    slice_box = helper.make_node(
        "Slice",
        ["boxes_full", "slice_starts", "slice_ends", "slice_axes"],
        ["boxes"],
    )
    slice_score = helper.make_node(
        "Slice",
        ["scores_full", "slice_starts", "slice_ends", "slice_axes"],
        ["scores"],
    )

    # --- Input / output specs -----------------------------------------------------
    input_info = helper.make_tensor_value_info(
        "point_cloud", TensorProto.FLOAT, [BATCH, NUM_POINTS, POINT_DIM]
    )
    output_boxes = helper.make_tensor_value_info(
        "boxes", TensorProto.FLOAT, [BATCH, NUM_DETECTIONS, BOX_DIM]
    )
    output_scores = helper.make_tensor_value_info(
        "scores", TensorProto.FLOAT, [BATCH, NUM_DETECTIONS, SCORE_DIM]
    )

    # --- Assemble graph -----------------------------------------------------------
    graph = helper.make_graph(
        nodes=[
            matmul1, add1, relu1,
            matmul_box, add_box,
            matmul_score, add_score, sigmoid_score,
            slice_box, slice_score,
        ],
        name="test_detector",
        inputs=[input_info],
        outputs=[output_boxes, output_scores],
        initializer=[
            w1, b1, w2_box, b2_box, w2_score, b2_score,
            axes_init, starts_init, ends_init,
        ],
    )

    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    return model


def main() -> None:
    out_dir = Path(__file__).resolve().parent.parent / "test-data" / "models"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "test_detector.onnx"

    model = build_model()
    onnx.save(model, str(out_path))

    size_kb = out_path.stat().st_size / 1024
    print(f"Saved {out_path}  ({size_kb:.1f} KB)")
    assert size_kb < 1024, f"Model too large: {size_kb:.1f} KB (limit 1 MB)"
    print("Model passes onnx.checker.check_model -- OK")


if __name__ == "__main__":
    main()
