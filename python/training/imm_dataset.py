"""Sliding-window dataset for the IMM mode classifier (task 6.1).

Reads the Parquet produced in Phase 5 by the Rust ``gen-imm-dataset`` binary
(``test-data/training/imm-classifier/imm-samples.parquet``) and builds
fixed-length windows of consecutive filter-state feature vectors, each paired
with the analytic mode label at the *end* of the window.

This module is intentionally **torch-free** (numpy + pyarrow only) so it can be
unit-tested in the default CI lane without the heavy ``training`` extras. The
windows it returns are consumed by ``imm_model.py`` / ``train_imm.py``.

Feature width and window length mirror the Rust constants
``thresh_filter::imm::CLASSIFIER_FEATURE_DIM`` (12) and
``thresh_filter::imm_adapter::WINDOW_LEN`` (10).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

#: Filter-state feature width (state[6] + covariance-diagonal[6]).
FEATURE_DIM = 12
#: Number of consecutive timesteps per classifier input window.
WINDOW_LEN = 10
#: Number of IMM modes: CV, CA, CTRV, coord_turn.
NUM_MODES = 4


@dataclass
class ImmWindows:
    """A materialised set of sliding windows.

    Attributes:
        x: ``(N, WINDOW_LEN, FEATURE_DIM)`` float32 input windows.
        y: ``(N,)`` int64 mode-label index at the end of each window.
        trajectory_ids: ``(N,)`` uint32 source trajectory id (for grouped splits).
    """

    x: np.ndarray
    y: np.ndarray
    trajectory_ids: np.ndarray

    def __len__(self) -> int:
        return int(self.x.shape[0])


def _grouped_rows(path: Path | str) -> dict[tuple[int, int], list[tuple[float, list[float], int]]]:
    """Read the Parquet and group ``(time_s, feature, label_index)`` rows by
    ``(trajectory_id, track_id)``."""
    table = pq.read_table(path)
    cols = table.to_pydict()
    groups: dict[tuple[int, int], list[tuple[float, list[float], int]]] = {}
    for traj, track, time_s, feature, label_index in zip(
        cols["trajectory_id"],
        cols["track_id"],
        cols["time_s"],
        cols["feature"],
        cols["label_index"],
        strict=True,
    ):
        feats = list(feature)
        if len(feats) != FEATURE_DIM:
            raise ValueError(
                f"feature width {len(feats)} != expected {FEATURE_DIM}; "
                "regenerate the dataset (CLASSIFIER_FEATURE_DIM mismatch)"
            )
        groups.setdefault((int(traj), int(track)), []).append(
            (float(time_s), feats, int(label_index))
        )
    return groups


def build_windows(path: Path | str, window: int = WINDOW_LEN) -> ImmWindows:
    """Build sliding windows of length ``window`` from the Parquet at ``path``.

    Windows never span two tracks (or two trajectories): each ``(trajectory_id,
    track_id)`` group is sorted by time and slid independently. Groups shorter
    than ``window`` contribute no windows. The label of a window is the mode
    label at its final timestep.
    """
    groups = _grouped_rows(path)
    xs: list[list[list[float]]] = []
    ys: list[int] = []
    tids: list[int] = []
    for (traj, _track), rows in groups.items():
        rows.sort(key=lambda r: r[0])  # by time_s
        feats = [r[1] for r in rows]
        labels = [r[2] for r in rows]
        for start in range(len(rows) - window + 1):
            xs.append(feats[start : start + window])
            ys.append(labels[start + window - 1])
            tids.append(traj)

    if not xs:
        return ImmWindows(
            x=np.empty((0, window, FEATURE_DIM), dtype=np.float32),
            y=np.empty((0,), dtype=np.int64),
            trajectory_ids=np.empty((0,), dtype=np.uint32),
        )
    return ImmWindows(
        x=np.asarray(xs, dtype=np.float32),
        y=np.asarray(ys, dtype=np.int64),
        trajectory_ids=np.asarray(tids, dtype=np.uint32),
    )


def trajectory_split(
    windows: ImmWindows, train_fraction: float = 0.8, seed: int = 42
) -> tuple[np.ndarray, np.ndarray]:
    """Split window indices into train/test by *trajectory* (not by row), so no
    trajectory leaks across the split. Deterministic given ``seed``.

    Returns ``(train_indices, test_indices)``.
    """
    unique = np.unique(windows.trajectory_ids)
    rng = np.random.default_rng(seed)
    shuffled = unique.copy()
    rng.shuffle(shuffled)
    n_train = max(1, round(len(shuffled) * train_fraction))
    train_trajs = set(shuffled[:n_train].tolist())
    train_mask = np.array([tid in train_trajs for tid in windows.trajectory_ids], dtype=bool)
    train_idx = np.nonzero(train_mask)[0]
    test_idx = np.nonzero(~train_mask)[0]
    return train_idx, test_idx
