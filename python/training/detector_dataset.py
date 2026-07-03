"""Track A detector training dataset (task 7.2).

Torch-free (numpy + pyarrow) reader of the Parquet produced by the Rust
``gen-detector-dataset`` binary
(``test-data/training/detector/detector-samples.parquet``). Each row is one
synth snapshot: a `(point_cloud, gt_boxes, gt_valid, gt_classes)` tuple in the
ONNX detector shapes. Runs in the default CI lane (no torch needed); the torch
training loop in ``train_detector.py`` consumes these arrays.

Shapes mirror the Rust contract (crates/thresh-synth, scripts/generate_test_model.py):
point clouds are ``1000 x 4``, boxes ``100 x 7``, validity/classes length 100.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

NUM_POINTS = 1000
POINT_DIM = 4  # x, y, z, intensity
MAX_BOXES = 100
BOX_DIM = 7  # x, y, z, L, W, H, yaw
NUM_CLASSES = 5
SCENE_SCALE_M = 100_000.0
"""Constant scene normalization scale (metres).

Real ingested scenes span ±1e4 to 1e5 m (the synth clutter cube half-extent is
100 km), which the detector cannot regress in raw units — the first real GPU
run scored mAP 0.0 (design.md findings §4). The model consumes xyz divided by
this scale and predicts normalized boxes; training scales ground truth to
match and the export wrapper multiplies boxes back, so the Parquet dataset
and the ONNX contract both stay in raw metres. Constant (not per-snapshot
statistics) so the exported graph needs no data-dependent branches."""


@dataclass
class DetectorSamples:
    """A materialised set of detector training snapshots.

    Attributes:
        point_clouds: ``(N, 1000, 4)`` float32.
        gt_boxes: ``(N, 100, 7)`` float32.
        gt_valid: ``(N, 100)`` bool validity mask.
        gt_classes: ``(N, 100)`` int64 class indices.
        trajectory_ids: ``(N,)`` uint32 source scene ids.
    """

    point_clouds: np.ndarray
    gt_boxes: np.ndarray
    gt_valid: np.ndarray
    gt_classes: np.ndarray
    trajectory_ids: np.ndarray

    def __len__(self) -> int:
        return int(self.point_clouds.shape[0])


def load_detector_samples(path: Path | str) -> DetectorSamples:
    """Read the detector-samples Parquet at ``path`` into stacked numpy arrays.

    Raises ``ValueError`` if a column's flattened length disagrees with the
    expected ONNX contract dimensions.
    """
    cols = pq.read_table(path).to_pydict()
    n = len(cols["trajectory_id"])

    def _stack(column: str, width: int, dtype: type) -> np.ndarray:
        rows = cols[column]
        arr = np.asarray([list(r) for r in rows], dtype=dtype)
        if arr.shape != (n, width):
            raise ValueError(f"column {column!r} has shape {arr.shape}, expected {(n, width)}")
        return arr

    point_clouds = _stack("point_cloud", NUM_POINTS * POINT_DIM, np.float32).reshape(
        n, NUM_POINTS, POINT_DIM
    )
    gt_boxes = _stack("gt_boxes", MAX_BOXES * BOX_DIM, np.float32).reshape(n, MAX_BOXES, BOX_DIM)
    gt_valid = _stack("gt_valid", MAX_BOXES, np.bool_)
    gt_classes = _stack("gt_classes", MAX_BOXES, np.int64)
    trajectory_ids = np.asarray(cols["trajectory_id"], dtype=np.uint32)

    return DetectorSamples(
        point_clouds=point_clouds,
        gt_boxes=gt_boxes,
        gt_valid=gt_valid,
        gt_classes=gt_classes,
        trajectory_ids=trajectory_ids,
    )


def valid_boxes(samples: DetectorSamples, index: int) -> tuple[np.ndarray, np.ndarray]:
    """Return the valid ground-truth boxes and their class indices for snapshot
    ``index`` — i.e. the rows where the validity mask is set.

    Returns ``(boxes (M, 7) float32, classes (M,) int64)`` where ``M`` is the
    number of valid boxes in that snapshot.
    """
    mask = samples.gt_valid[index]
    return samples.gt_boxes[index][mask], samples.gt_classes[index][mask]
