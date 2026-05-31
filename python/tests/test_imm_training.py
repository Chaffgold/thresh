"""Smoke tests for the torch training + ONNX export path (tasks 6.2-6.5).

Skipped entirely when the ``training`` extra (torch / onnxruntime) is not
installed, so the default CI lane stays torch-free. Run them with:

    uv run --extra training pytest tests/test_imm_training.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")

import torch

from training.imm_dataset import FEATURE_DIM, NUM_MODES, WINDOW_LEN, ImmWindows
from training.imm_model import ImmModeClassifier
from training.train_imm import train


def _separable_windows() -> ImmWindows:
    """Six trajectories: three of class CV (0) with low features, three of class
    coord_turn (3) with high features — trivially separable so a tiny model can
    learn it in a few epochs."""
    rng = np.random.default_rng(0)
    xs: list[np.ndarray] = []
    ys: list[int] = []
    tids: list[int] = []
    for traj in range(6):
        label = 0 if traj < 3 else 3
        base = 0.0 if label == 0 else 5.0
        for _ in range(8):
            noise = rng.standard_normal((WINDOW_LEN, FEATURE_DIM)) * 0.1
            xs.append((noise + base).astype(np.float32))
            ys.append(label)
            tids.append(traj)
    return ImmWindows(
        x=np.asarray(xs, dtype=np.float32),
        y=np.asarray(ys, dtype=np.int64),
        trajectory_ids=np.asarray(tids, dtype=np.uint32),
    )


def test_model_forward_shape() -> None:
    model = ImmModeClassifier(hidden_dim=8)
    out = model(torch.zeros(2, WINDOW_LEN, FEATURE_DIM))
    assert out.shape == (2, NUM_MODES)


def test_smoke_train_runs_and_learns() -> None:
    windows = _separable_windows()
    model, acc = train(windows, epochs=8, hidden_dim=16, seed=0)
    assert 0.0 <= acc <= 1.0
    # The task is trivially separable; a few epochs should beat chance (0.5 for
    # the two populated classes). Keep the bar low — this is a smoke test, not
    # the real exit criterion.
    assert acc >= 0.5
    out = model(torch.from_numpy(windows.x[:3]))
    assert out.shape == (3, NUM_MODES)


def test_export_and_onnxruntime_verify(tmp_path: Path) -> None:
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    from export.export_imm import export, verify

    windows = _separable_windows()
    model, _ = train(windows, epochs=3, hidden_dim=8, seed=0)
    ckpt = tmp_path / "imm.pt"
    torch.save({"state_dict": model.state_dict(), "hidden_dim": 8}, ckpt)

    onnx_path = tmp_path / "imm_mode_classifier.onnx"
    export(ckpt, onnx_path)
    assert onnx_path.exists()
    # `verify` asserts shape (batch, 4) and rows summing to 1.0.
    verify(onnx_path, batch=4)
