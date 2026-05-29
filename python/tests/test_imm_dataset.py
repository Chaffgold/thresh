"""Tests for the torch-free IMM sliding-window dataset (task 6.1).

These run in the default CI lane (no torch needed)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from training.imm_dataset import (
    FEATURE_DIM,
    NUM_MODES,
    WINDOW_LEN,
    ImmWindows,
    build_windows,
    trajectory_split,
)

# Repo-root sample produced by the Rust `gen-imm-dataset` binary (Phase 5).
SAMPLE_PARQUET = Path("../test-data/training/imm-classifier/imm-samples.parquet")


def _write_parquet(path: Path, trajectories: dict[int, tuple[int, int]]) -> None:
    """Write a synthetic IMM-samples Parquet.

    ``trajectories`` maps ``trajectory_id -> (num_rows, label_index)``; each
    trajectory is a single track with monotonically increasing time.
    """
    traj_ids: list[int] = []
    track_ids: list[int] = []
    times: list[float] = []
    feats: list[list[float]] = []
    labels: list[int] = []
    for traj, (n, label) in trajectories.items():
        for i in range(n):
            traj_ids.append(traj)
            track_ids.append(traj * 1000)
            times.append(float(i))
            feats.append([float(traj) + 0.01 * i] * FEATURE_DIM)
            labels.append(label)
    table = pa.table(
        {
            "trajectory_id": pa.array(traj_ids, pa.uint32()),
            "track_id": pa.array(track_ids, pa.uint64()),
            "time_s": pa.array(times, pa.float64()),
            "feature": pa.array(feats, pa.list_(pa.float64())),
            "label_index": pa.array(labels, pa.int32()),
        }
    )
    pq.write_table(table, path)


def test_build_windows_shapes_and_labels(tmp_path: Path) -> None:
    path = tmp_path / "synthetic.parquet"
    # Two trajectories: CV (label 0) and coord_turn (label 3), 15 rows each.
    _write_parquet(path, {0: (15, 0), 1: (15, 3)})

    windows = build_windows(path, window=WINDOW_LEN)
    # 15 - 10 + 1 = 6 windows per trajectory.
    assert len(windows) == 12
    assert windows.x.shape == (12, WINDOW_LEN, FEATURE_DIM)
    assert windows.x.dtype == np.float32
    assert set(windows.y.tolist()) == {0, 3}
    assert (windows.y == 0).sum() == 6
    assert (windows.y == 3).sum() == 6


def test_windows_never_span_trajectories(tmp_path: Path) -> None:
    path = tmp_path / "synthetic.parquet"
    _write_parquet(path, {0: (15, 0), 1: (15, 3)})
    windows = build_windows(path, window=WINDOW_LEN)
    # Each window's trajectory id is consistent with its label mapping above.
    for tid, label in zip(windows.trajectory_ids, windows.y, strict=True):
        expected = 0 if tid == 0 else 3
        assert label == expected


def test_short_tracks_yield_no_windows(tmp_path: Path) -> None:
    path = tmp_path / "short.parquet"
    _write_parquet(path, {0: (WINDOW_LEN - 1, 0)})  # too short
    windows = build_windows(path, window=WINDOW_LEN)
    assert len(windows) == 0
    assert windows.x.shape == (0, WINDOW_LEN, FEATURE_DIM)


def test_trajectory_split_is_disjoint_and_deterministic() -> None:
    # 5 trajectories, 4 windows each.
    x = np.zeros((20, WINDOW_LEN, FEATURE_DIM), dtype=np.float32)
    y = np.zeros((20,), dtype=np.int64)
    tids = np.repeat(np.arange(5, dtype=np.uint32), 4)
    windows = ImmWindows(x=x, y=y, trajectory_ids=tids)

    train_idx, test_idx = trajectory_split(windows, train_fraction=0.8, seed=42)
    train_trajs = set(tids[train_idx].tolist())
    test_trajs = set(tids[test_idx].tolist())
    assert train_trajs.isdisjoint(test_trajs), "no trajectory may leak across the split"
    assert train_trajs | test_trajs == set(range(5))
    # Deterministic given the seed.
    again = trajectory_split(windows, train_fraction=0.8, seed=42)
    assert np.array_equal(train_idx, again[0])


@pytest.mark.skipif(not SAMPLE_PARQUET.exists(), reason="committed sample parquet not present")
def test_committed_sample_loads() -> None:
    windows = build_windows(SAMPLE_PARQUET, window=WINDOW_LEN)
    assert len(windows) > 0
    assert windows.x.shape[1:] == (WINDOW_LEN, FEATURE_DIM)
    assert windows.y.min() >= 0 and windows.y.max() < NUM_MODES
