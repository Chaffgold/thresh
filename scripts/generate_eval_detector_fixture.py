#!/usr/bin/env python3
"""Generate a constant one-detection ONNX fixture for bounded evaluator tests.

This is fabricated test data, not a trained model. It ignores point-cloud
values and always emits one confidence-1 box at sensor-ENU [5000, 0, 1000] m;
the remaining 99 scores are zero. No external data or training weights are
used. It must never serve as an accuracy baseline or replace test_detector.onnx.

Run with the python/ training-extra environment from any working directory.
"""

from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


def build_model() -> onnx.ModelProto:
    boxes = np.zeros((1, 100, 7), dtype=np.float32)
    boxes[0, 0] = [5000.0, 0.0, 1000.0, 10.0, 10.0, 10.0, 0.0]
    scores = np.zeros((1, 100, 1), dtype=np.float32)
    scores[0, 0, 0] = 1.0
    classes = np.ones((1, 100, 1), dtype=np.int64)
    nodes = [
        helper.make_node("Constant", [], [name], value=numpy_helper.from_array(value))
        for name, value in [("boxes", boxes), ("scores", scores), ("classes", classes)]
    ]
    graph = helper.make_graph(
        nodes, "synthetic_single_detection_eval_fixture",
        [helper.make_tensor_value_info("point_cloud", TensorProto.FLOAT, [1, 1000, 4])],
        [
            helper.make_tensor_value_info("boxes", TensorProto.FLOAT, [1, 100, 7]),
            helper.make_tensor_value_info("scores", TensorProto.FLOAT, [1, 100, 1]),
            helper.make_tensor_value_info("classes", TensorProto.INT64, [1, 100, 1]),
        ],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=9)
    model.doc_string = "Wholly synthetic constant fixture. Not trained; not an accuracy baseline."
    onnx.checker.check_model(model)
    return model


if __name__ == "__main__":
    output = Path(__file__).resolve().parents[1] / "test-data/models/eval_single_detection.onnx"
    onnx.save(build_model(), output)
    print(f"Wrote synthetic evaluation fixture: {output}")
