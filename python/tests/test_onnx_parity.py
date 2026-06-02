"""Test the cross-language ONNX parity check (task 8.3).

Skipped unless ``onnxruntime`` is installed (the ``training`` extra) AND the
committed IMM-classifier stub exists. The Rust side is invoked via cargo, so
this also requires a working Rust toolchain — guarded so the default torch-free
CI lane (no onnxruntime) skips cleanly; it runs in the onnx-tests job.

    uv run --extra training pytest tests/test_onnx_parity.py
"""

from __future__ import annotations

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
def test_python_rust_parity_within_tolerance() -> None:
    from eval.onnx_parity import check_parity

    # Builds + runs the Rust `onnx-infer` binary; asserts agreement within 1e-5.
    max_diff = check_parity(DEFAULT_MODEL, DEFAULT_SHAPE)
    assert max_diff <= 1e-5
