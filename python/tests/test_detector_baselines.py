"""Unit tests for the classical/oracle detector baselines (task 7.10).

Torch-free: hand-built point clouds exercise the numpy DBSCAN, the
cluster-centroid baseline, and the GT-segmented oracle in the default CI lane.
"""

from __future__ import annotations

import numpy as np

from eval.detector_baselines import (
    classical_detections,
    dbscan_labels,
    oracle_detections,
)
from training.classes import TargetClass


def _cluster(centre: tuple[float, float, float], n: int, spread: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    points = rng.normal(loc=centre, scale=spread, size=(n, 3))
    intensity = np.full((n, 1), 0.8)
    return np.hstack([points, intensity])


def _scatter(n: int, half_extent: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    points = rng.uniform(-half_extent, half_extent, size=(n, 3))
    intensity = np.full((n, 1), 0.1)
    return np.hstack([points, intensity])


class TestDbscan:
    def test_tight_cluster_vs_far_scatter(self) -> None:
        cloud = np.vstack([_cluster((50.0, 0.0, 0.0), 16, 5.0, seed=1), _scatter(30, 5e4, seed=2)])
        labels = dbscan_labels(cloud[:, :3], eps=15.0, min_samples=4)
        assert set(labels[:16]) == {0}  # the 16 returns form one cluster
        assert (labels[16:] == -1).all()  # sparse clutter is all noise

    def test_below_min_samples_is_noise(self) -> None:
        cloud = _cluster((0.0, 0.0, 0.0), 3, 1.0, seed=3)
        labels = dbscan_labels(cloud[:, :3], eps=15.0, min_samples=4)
        assert (labels == -1).all()

    def test_empty_input(self) -> None:
        labels = dbscan_labels(np.zeros((0, 3)), eps=15.0, min_samples=4)
        assert labels.shape == (0,)


class TestClassicalDetections:
    def test_finds_target_cluster_centroid(self) -> None:
        target = _cluster((50.0, -20.0, 10.0), 16, 5.0, seed=4)
        cloud = np.vstack([target, _scatter(100, 5e4, seed=5)])
        det = classical_detections(cloud)
        assert det["boxes"].shape == (1, 7)
        np.testing.assert_allclose(det["boxes"][0, :3], target[:, :3].mean(axis=0), atol=1e-9)
        assert det["scores"][0] == 1.0  # 16/16 expected returns
        assert det["classes"][0] == int(TargetClass.OTHER)

    def test_no_cluster_no_detection(self) -> None:
        det = classical_detections(_scatter(100, 5e4, seed=6))
        assert det["boxes"].shape == (0, 7)
        assert det["scores"].shape == (0,)
        assert det["classes"].shape == (0,)

    def test_partial_cluster_scores_below_one(self) -> None:
        cloud = np.vstack([_cluster((0.0, 0.0, 0.0), 8, 3.0, seed=7), _scatter(50, 5e4, seed=8)])
        det = classical_detections(cloud)
        assert det["boxes"].shape == (1, 7)
        assert abs(float(det["scores"][0]) - 0.5) < 1e-9  # 8/16


class TestOracleDetections:
    def _gt(self, centres: list[list[float]], classes: list[int]) -> tuple[np.ndarray, ...]:
        n = len(centres)
        gt_boxes = np.zeros((n, 7), dtype=np.float32)
        for j, c in enumerate(centres):
            gt_boxes[j, :3] = c
            gt_boxes[j, 3:6] = (10.0, 10.0, 5.0)
        gt_valid = np.ones(n, dtype=bool)
        gt_classes = np.array(classes, dtype=np.int64)
        return gt_boxes, gt_valid, gt_classes

    def test_segments_per_target_and_keeps_gt_class(self) -> None:
        a = _cluster((100.0, 0.0, 0.0), 16, 4.0, seed=9)
        b = _cluster((-200.0, 50.0, 0.0), 16, 4.0, seed=10)
        cloud = np.vstack([a, b, _scatter(50, 5e4, seed=11)])
        gt_boxes, gt_valid, gt_classes = self._gt(
            [[100.0, 0.0, 0.0], [-200.0, 50.0, 0.0]], [1, 2]
        )
        det = oracle_detections(cloud, gt_boxes, gt_valid, gt_classes)
        assert det["boxes"].shape == (2, 7)
        np.testing.assert_allclose(det["classes"], [1, 2])
        # Each centroid is the mean of the returns within the oracle radius —
        # far closer to truth than any single 4 m-noise return.
        assert np.linalg.norm(det["boxes"][0, :3] - [100.0, 0.0, 0.0]) < 3.0
        assert np.linalg.norm(det["boxes"][1, :3] - [-200.0, 50.0, 0.0]) < 3.0
        assert (det["scores"] == 1.0).all()

    def test_target_without_returns_is_missed(self) -> None:
        cloud = _scatter(50, 5e4, seed=12)
        gt_boxes, gt_valid, gt_classes = self._gt([[0.0, 0.0, 0.0]], [1])
        det = oracle_detections(cloud, gt_boxes, gt_valid, gt_classes)
        assert det["boxes"].shape == (0, 7)
