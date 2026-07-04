"""Classical and oracle detector baselines (task 7.10, design Decision 26).

Two torch-free detection sources that emit the same per-snapshot detection
dict as ``eval.detector_map`` (``boxes (N, 7)`` / ``scores (N,)`` /
``classes (N,)``), so all sources score under identical matching:

- **Classical baseline** — plain-numpy DBSCAN over point positions, one
  detection per cluster at the member centroid, ranked by cluster size.
  This is the bar the learned detector must beat (Decision 26): position-only
  clustering is a fair classical competitor because the synth scenes pad with
  uniform clutter over a ±100 km cube, so accidental clutter clusters are
  vanishingly rare while a target's ~16 returns (5 m noise) form one tight
  cluster. It has no classifier, so every detection carries the majority
  prior class (``OTHER``).
- **Oracle ceiling** — centroid of the returns within ``ORACLE_RADIUS_M`` of
  each ground-truth centre (GT-segmented, the CRLB-achieving estimator), with
  ground-truth class and extent. No real detector can out-localize it; it
  locates every run on the bracket ``stub … classical … learned … oracle``.
"""

from __future__ import annotations

import numpy as np

from training.classes import TargetClass
from training.detector_dataset import DetectorSamples

# ~2x the typical inter-return distance (returns differ by sigma*sqrt(2) = 7 m
# per axis at the 5 m return noise), small enough that uniform clutter over the
# +/-100 km cube essentially never links.
DBSCAN_EPS_M = 15.0
DBSCAN_MIN_SAMPLES = 4
# 3 sigma at the 5 m return noise: captures ~97% of a target's returns.
ORACLE_RADIUS_M = 15.0
# `returns_per_target` default in thresh-synth's TrajectoryRadarConfig.
EXPECTED_CLUSTER_SIZE = 16
# Detector box prior, matching training.detector_model.
PRIOR_DIMS_M = (10.0, 10.0, 5.0)


def _empty_detections() -> dict[str, np.ndarray]:
    return {
        "boxes": np.zeros((0, 7), dtype=np.float64),
        "scores": np.zeros((0,), dtype=np.float64),
        "classes": np.zeros((0,), dtype=np.int64),
    }


def dbscan_labels(points: np.ndarray, eps: float, min_samples: int) -> np.ndarray:
    """Plain-numpy DBSCAN over ``(N, 3)`` points; returns ``-1`` for noise.

    O(N^2) neighbourhood matrix — fine at the (1000, 4) snapshot contract.
    """
    n = points.shape[0]
    labels = np.full(n, -1, dtype=np.int64)
    if n == 0:
        return labels
    deltas = points[:, None, :] - points[None, :, :]
    neighbours = np.einsum("ijk,ijk->ij", deltas, deltas) <= eps * eps
    core = neighbours.sum(axis=1) >= min_samples
    cluster = 0
    for seed in range(n):
        if labels[seed] != -1 or not core[seed]:
            continue
        labels[seed] = cluster
        stack = [seed]
        while stack:
            member = stack.pop()
            if not core[member]:
                continue  # border points join a cluster but never expand it
            for neighbour in np.nonzero(neighbours[member])[0]:
                if labels[neighbour] == -1:
                    labels[neighbour] = cluster
                    stack.append(int(neighbour))
        cluster += 1
    return labels


def classical_detections(
    point_cloud: np.ndarray,
    *,
    eps: float = DBSCAN_EPS_M,
    min_samples: int = DBSCAN_MIN_SAMPLES,
    expected_cluster_size: int = EXPECTED_CLUSTER_SIZE,
) -> dict[str, np.ndarray]:
    """DBSCAN-centroid detections for one ``(1000, 4)`` snapshot."""
    xyz = point_cloud[:, :3].astype(np.float64)
    labels = dbscan_labels(xyz, eps, min_samples)
    boxes: list[list[float]] = []
    scores: list[float] = []
    for cluster in range(int(labels.max()) + 1):
        members = xyz[labels == cluster]
        centre = members.mean(axis=0)
        boxes.append([*centre.tolist(), *PRIOR_DIMS_M, 0.0])
        scores.append(min(1.0, members.shape[0] / expected_cluster_size))
    if not boxes:
        return _empty_detections()
    return {
        "boxes": np.array(boxes, dtype=np.float64),
        "scores": np.array(scores, dtype=np.float64),
        "classes": np.full(len(boxes), int(TargetClass.OTHER), dtype=np.int64),
    }


def oracle_detections(
    point_cloud: np.ndarray,
    gt_boxes: np.ndarray,
    gt_valid: np.ndarray,
    gt_classes: np.ndarray,
    *,
    radius: float = ORACLE_RADIUS_M,
) -> dict[str, np.ndarray]:
    """GT-segmented centroid detections (the CRLB ceiling) for one snapshot.

    For each valid ground-truth box, the centroid of the returns within
    ``radius`` of the true centre, with ground-truth class and extent. A
    target with no returns in range is a miss (recall < 1 stays possible).
    """
    xyz = point_cloud[:, :3].astype(np.float64)
    boxes: list[list[float]] = []
    classes: list[int] = []
    for gt, cls in zip(gt_boxes[gt_valid], gt_classes[gt_valid], strict=True):
        selected = xyz[np.linalg.norm(xyz - gt[:3].astype(np.float64), axis=1) <= radius]
        if selected.shape[0] == 0:
            continue
        centre = selected.mean(axis=0)
        boxes.append([*centre.tolist(), *gt[3:7].astype(np.float64).tolist()])
        classes.append(int(cls))
    if not boxes:
        return _empty_detections()
    return {
        "boxes": np.array(boxes, dtype=np.float64),
        "scores": np.ones(len(boxes), dtype=np.float64),
        "classes": np.array(classes, dtype=np.int64),
    }


def classical_detections_batch(
    samples: DetectorSamples, **kwargs: float | int
) -> list[dict[str, np.ndarray]]:
    """Classical baseline over every snapshot in ``samples``."""
    return [
        classical_detections(samples.point_clouds[i], **kwargs)  # type: ignore[arg-type]
        for i in range(len(samples))
    ]


def oracle_detections_batch(
    samples: DetectorSamples, **kwargs: float
) -> list[dict[str, np.ndarray]]:
    """Oracle ceiling over every snapshot in ``samples``."""
    return [
        oracle_detections(
            samples.point_clouds[i],
            samples.gt_boxes[i],
            samples.gt_valid[i],
            samples.gt_classes[i],
            **kwargs,  # type: ignore[arg-type]
        )
        for i in range(len(samples))
    ]
