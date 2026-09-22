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
# Anchored to this file (not the CWD) so the test runs regardless of where
# pytest is launched from.
SAMPLE_PARQUET = (
    Path(__file__).resolve().parents[2]
    / "test-data"
    / "training"
    / "imm-classifier"
    / "imm-samples.parquet"
)


def _write_parquet(
    path: Path, trajectories: dict[int, tuple[int, int]], *, dt: float = 1.0
) -> None:
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
            times.append(float(i) * dt)
            feats.append([float(traj) + 0.01 * i] * (FEATURE_DIM - 1) + [0.0 if i == 0 else dt])
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


def test_identical_states_at_1hz_and_10hz_have_distinct_timing(tmp_path: Path) -> None:
    slow, fast = tmp_path / "1hz.parquet", tmp_path / "10hz.parquet"
    _write_parquet(slow, {0: (12, 0)}, dt=1.0)
    _write_parquet(fast, {0: (12, 0)}, dt=0.1)
    a, b = build_windows(slow), build_windows(fast)
    np.testing.assert_array_equal(a.x[:, :, :-1], b.x[:, :, :-1])
    assert a.x[0, 0, -1] == b.x[0, 0, -1] == 0.0
    np.testing.assert_allclose(a.x[0, 1:, -1], 1.0)
    np.testing.assert_allclose(b.x[0, 1:, -1], 0.1)
    # Moving the window forward must not reset its first interval.
    assert a.x[1, 0, -1] == 1.0
    assert b.x[1, 0, -1] == np.float32(0.1)


def _write_timed_rows(path: Path, times: list[float], elapsed: list[float]) -> None:
    pq.write_table(pa.table({
        "trajectory_id": [0] * len(times),
        "track_id": [1] * len(times),
        "time_s": times,
        "feature": [[0.0] * (FEATURE_DIM - 1) + [dt] for dt in elapsed],
        "label_index": [0] * len(times),
    }), path)


def test_missing_observations_keep_the_elapsed_gap(tmp_path: Path) -> None:
    path = tmp_path / "gap.parquet"
    _write_timed_rows(path, [0.1, 0.4, 0.5], [0.0, 0.3, 0.1])
    windows = build_windows(path, window=2)
    np.testing.assert_allclose(windows.x[:, :, -1], [[0.0, 0.3], [0.3, 0.1]])


def test_same_time_updates_preserve_zero_intervals(tmp_path: Path) -> None:
    path = tmp_path / "same-time.parquet"
    _write_timed_rows(path, [0.1, 0.1, 0.4], [0.0, 0.0, 0.3])
    windows = build_windows(path, window=2)
    np.testing.assert_allclose(windows.x[:, :, -1], [[0.0, 0.0], [0.0, 0.3]])


def test_out_of_order_rows_are_sorted_but_intervals_are_not_rewritten(tmp_path: Path) -> None:
    path = tmp_path / "unordered.parquet"
    _write_timed_rows(path, [0.4, 0.1], [0.3, 0.0])
    np.testing.assert_allclose(build_windows(path, window=2).x[0, :, -1], [0.0, 0.3])
    _write_timed_rows(path, [0.4, 0.1], [0.0, 0.3])
    with pytest.raises(ValueError, match="elapsed_seconds"):
        build_windows(path, window=2)


@pytest.mark.parametrize(("times", "elapsed"), [
    ([0.0, 0.0], [0.0, 0.1]),
    ([0.0, float("nan")], [0.0, 0.1]),
    ([0.0, float("inf")], [0.0, 0.1]),
    ([0.0, 0.1], [0.1, 0.1]),
    ([0.0, 0.1], [0.0, 1.0]),
    ([0.0, 0.1], [0.0, -0.1]),
    ([0.0, 0.1], [0.0, float("inf")]),
])
def test_rejects_invalid_timestamps_and_intervals(
    tmp_path: Path, times: list[float], elapsed: list[float]
) -> None:
    path = tmp_path / "bad-time.parquet"
    _write_timed_rows(path, times, elapsed)
    with pytest.raises(ValueError):
        build_windows(path, window=2)


def test_rejects_legacy_width_with_regeneration_message(tmp_path: Path) -> None:
    path = tmp_path / "legacy.parquet"
    pq.write_table(pa.table({
        "trajectory_id": [0], "track_id": [1], "time_s": [0.0],
        "feature": [[0.0] * 12], "label_index": [0],
    }), path)
    with pytest.raises(ValueError, match=r"regenerate.*elapsed_seconds"):
        build_windows(path)


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
