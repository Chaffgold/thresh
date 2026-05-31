"""Tests for the torch-free detector dataset reader (task 7.2). Run in CI."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from training.detector_dataset import (
    BOX_DIM,
    MAX_BOXES,
    NUM_POINTS,
    POINT_DIM,
    load_detector_samples,
    valid_boxes,
)

# Repo-root sample produced by the Rust `gen-detector-dataset` binary.
SAMPLE_PARQUET = (
    Path(__file__).resolve().parents[2]
    / "test-data"
    / "training"
    / "detector"
    / "detector-samples.parquet"
)


def _write_detector_parquet(path: Path, n: int = 2) -> None:
    pcs = [[0.1] * (NUM_POINTS * POINT_DIM) for _ in range(n)]
    boxes = [[1.0] * (MAX_BOXES * BOX_DIM) for _ in range(n)]
    valid = [[i == 0] + [False] * (MAX_BOXES - 1) for i in range(n)]  # row 0 has 1 valid
    classes = [[2] + [0] * (MAX_BOXES - 1) for _ in range(n)]
    table = pa.table(
        {
            "trajectory_id": pa.array(list(range(n)), pa.uint32()),
            "snapshot_index": pa.array(list(range(n)), pa.uint64()),
            "time_s": pa.array([float(i) for i in range(n)], pa.float64()),
            "point_cloud": pa.array(pcs, pa.list_(pa.float32())),
            "gt_boxes": pa.array(boxes, pa.list_(pa.float32())),
            "gt_valid": pa.array(valid, pa.list_(pa.bool_())),
            "gt_classes": pa.array(classes, pa.list_(pa.int64())),
        }
    )
    pq.write_table(table, path)


def test_load_shapes(tmp_path: Path) -> None:
    path = tmp_path / "detector.parquet"
    _write_detector_parquet(path, n=3)
    s = load_detector_samples(path)
    assert len(s) == 3
    assert s.point_clouds.shape == (3, NUM_POINTS, POINT_DIM)
    assert s.point_clouds.dtype == np.float32
    assert s.gt_boxes.shape == (3, MAX_BOXES, BOX_DIM)
    assert s.gt_valid.shape == (3, MAX_BOXES)
    assert s.gt_valid.dtype == np.bool_
    assert s.gt_classes.shape == (3, MAX_BOXES)
    assert s.gt_classes.dtype == np.int64


def test_valid_boxes_filters_by_mask(tmp_path: Path) -> None:
    path = tmp_path / "detector.parquet"
    _write_detector_parquet(path, n=2)
    s = load_detector_samples(path)
    boxes, classes = valid_boxes(s, 0)  # row 0 has exactly one valid box
    assert boxes.shape == (1, BOX_DIM)
    assert classes.tolist() == [2]
    # Row 1 has no valid boxes.
    boxes1, classes1 = valid_boxes(s, 1)
    assert boxes1.shape == (0, BOX_DIM)
    assert classes1.shape == (0,)


@pytest.mark.skipif(not SAMPLE_PARQUET.exists(), reason="committed sample parquet not present")
def test_committed_sample_loads() -> None:
    s = load_detector_samples(SAMPLE_PARQUET)
    assert len(s) > 0
    assert s.point_clouds.shape[1:] == (NUM_POINTS, POINT_DIM)
    assert s.gt_boxes.shape[1:] == (MAX_BOXES, BOX_DIM)
    # Class indices are in range.
    assert s.gt_classes.min() >= 0 and s.gt_classes.max() < 5
