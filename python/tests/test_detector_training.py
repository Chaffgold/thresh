"""Smoke tests for the detector model, set-prediction loss, and ONNX export
(tasks 7.3/7.6). Skipped without the ``training`` extra (torch/scipy/onnx).

    uv run --extra training pytest tests/test_detector_training.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("scipy")

import torch

from training.detector_dataset import (
    BOX_DIM,
    MAX_BOXES,
    NUM_CLASSES,
    NUM_POINTS,
    POINT_DIM,
    DetectorSamples,
)
from training.detector_model import DetectorModel
from training.train_detector import set_prediction_loss, train


def _tiny_samples(n: int = 4) -> DetectorSamples:
    """A few snapshots each with one valid GT box (cluster near a point)."""
    rng = np.random.default_rng(0)
    pcs = (rng.standard_normal((n, NUM_POINTS, POINT_DIM)) * 0.1).astype(np.float32)
    boxes = np.zeros((n, MAX_BOXES, BOX_DIM), dtype=np.float32)
    valid = np.zeros((n, MAX_BOXES), dtype=np.bool_)
    classes = np.zeros((n, MAX_BOXES), dtype=np.int64)
    for i in range(n):
        boxes[i, 0] = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 0.1], dtype=np.float32)
        valid[i, 0] = True
        classes[i, 0] = 1
    return DetectorSamples(
        point_clouds=pcs,
        gt_boxes=boxes,
        gt_valid=valid,
        gt_classes=classes,
        trajectory_ids=np.arange(n, dtype=np.uint32),
    )


def test_model_forward_shapes() -> None:
    model = DetectorModel(d_model=16)
    boxes, scores, class_logits = model(torch.zeros(2, NUM_POINTS, POINT_DIM))
    assert boxes.shape == (2, MAX_BOXES, BOX_DIM)
    assert scores.shape == (2, MAX_BOXES, 1)
    assert class_logits.shape == (2, MAX_BOXES, NUM_CLASSES)


def test_set_prediction_loss_computes_and_backprops() -> None:
    model = DetectorModel(d_model=16)
    boxes, scores, class_logits = model(torch.zeros(1, NUM_POINTS, POINT_DIM))
    gt_boxes = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 0.1]])
    gt_classes = torch.tensor([1])
    loss = set_prediction_loss(boxes[0], scores[0], class_logits[0], gt_boxes, gt_classes)
    assert torch.isfinite(loss)
    loss.backward()  # gradients flow
    # Empty-GT case is also valid (objectness-only).
    empty = set_prediction_loss(
        boxes[0].detach(),
        scores[0].detach(),
        class_logits[0].detach(),
        torch.zeros(0, BOX_DIM),
        torch.zeros(0, dtype=torch.long),
    )
    assert torch.isfinite(empty)


def test_smoke_train_reduces_loss() -> None:
    samples = _tiny_samples(4)
    _, loss_short = train(samples, epochs=1, d_model=16, seed=0)
    _, loss_long = train(samples, epochs=20, d_model=16, seed=0)
    assert np.isfinite(loss_short) and np.isfinite(loss_long)
    # Trivially-separable single-box task: more epochs should not increase loss.
    assert loss_long <= loss_short + 1e-3


def test_export_and_onnxruntime_verify(tmp_path: Path) -> None:
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    from export.export_detector import export, verify

    model, _ = train(_tiny_samples(4), epochs=2, d_model=16, seed=0)
    ckpt = tmp_path / "detector.pt"
    torch.save({"state_dict": model.state_dict(), "d_model": 16}, ckpt)

    onnx_path = tmp_path / "test_detector.onnx"
    export(ckpt, onnx_path)
    assert onnx_path.exists()
    verify(onnx_path, batch=2)  # asserts the 3-output contract + ranges
