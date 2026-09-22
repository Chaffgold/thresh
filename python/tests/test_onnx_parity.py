"""Test the cross-language ONNX parity check (task 8.3).

Skipped unless ``onnxruntime`` is installed (the ``training`` extra) AND the
committed IMM-classifier stub exists. The Rust side is invoked via cargo, so
this also requires a working Rust toolchain — guarded so the default torch-free
CI lane (no onnxruntime) skips cleanly; it runs in the onnx-tests job.

    uv run --extra training pytest tests/test_onnx_parity.py
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("onnxruntime")

from eval.onnx_parity import DEFAULT_MODEL, DEFAULT_SHAPE, fixture, python_output

pytestmark = pytest.mark.skipif(not DEFAULT_MODEL.exists(), reason="IMM stub not present")


def test_fixture_is_deterministic_and_shaped() -> None:
    a = fixture(DEFAULT_SHAPE)
    b = fixture(DEFAULT_SHAPE)
    assert a.shape == DEFAULT_SHAPE
    assert a.dtype == np.float32
    assert np.array_equal(a, b)  # deterministic
    assert abs(float(a.reshape(-1)[0]) - 0.0) < 1e-7  # sin(0) == 0


def test_python_runtime_produces_distribution() -> None:
    out = python_output(DEFAULT_MODEL, DEFAULT_SHAPE)
    assert out.shape == (4,)
    assert abs(float(out.sum()) - 1.0) < 1e-4  # softmax over 4 modes
    assert (out >= 0.0).all()


@pytest.mark.skipif(
    __import__("shutil").which("cargo") is None, reason="cargo (Rust) not available"
)
@pytest.mark.parametrize("elapsed_seconds", [0.1, 1.0, 0.3])
def test_python_rust_parity_within_tolerance(elapsed_seconds: float) -> None:
    from eval.onnx_parity import check_parity

    # Builds + runs the Rust `onnx-infer` binary; asserts agreement within 1e-5.
    max_diff = check_parity(DEFAULT_MODEL, DEFAULT_SHAPE, elapsed_seconds=elapsed_seconds)
    assert max_diff <= 1e-5


def test_elapsed_feature_changes_reach_the_onnx_model() -> None:
    slow = python_output(DEFAULT_MODEL, DEFAULT_SHAPE, elapsed_seconds=1.0)
    fast = python_output(DEFAULT_MODEL, DEFAULT_SHAPE, elapsed_seconds=0.1)
    assert not np.allclose(slow, fast, atol=1e-5)


@pytest.mark.skipif(
    __import__("shutil").which("cargo") is None, reason="cargo (Rust) not available"
)
def test_rust_constructor_rejects_legacy_12_wide_onnx(tmp_path: Path) -> None:
    onnx = pytest.importorskip("onnx")
    from eval.onnx_parity import REPO_ROOT

    legacy = onnx.load(str(DEFAULT_MODEL))
    legacy.graph.input[0].type.tensor_type.shape.dim[2].dim_value = 12
    weights = legacy.graph.initializer[0]
    old_weights = onnx.numpy_helper.to_array(weights)[:12].copy()
    weights.CopyFrom(onnx.numpy_helper.from_array(old_weights, name="w"))
    onnx.checker.check_model(legacy)
    path = tmp_path / "legacy-12.onnx"
    onnx.save(legacy, str(path))
    proc = subprocess.run([
        "cargo", "run", "-q", "-p", "thresh", "--features", "learned-imm",
        "--bin", "eval-tracker", "--", "--learned-imm", "--model", str(path), "--json",
    ], cwd=REPO_ROOT, capture_output=True, text=True, check=False)
    assert proc.returncode != 0
    assert "13" in proc.stderr and "re-export" in proc.stderr
