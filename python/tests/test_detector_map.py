"""Unit tests for the detector holdout evaluator (task 7.9 metrics).

Torch- and onnxruntime-free: exercises the pure-numpy IoU / AP / postprocess /
matching / aggregation pieces with hand-built detections, so it runs in the
default CI lane. The ONNX inference path is exercised by the real evaluation
run.
"""

from __future__ import annotations

from typing import cast

import numpy as np

from eval.detector_map import (
    average_precision,
    distance_gated_ap,
    evaluate,
    iou_3d_axis_aligned,
    postprocess,
)
from training.detector_dataset import DetectorSamples


def _box(cx: float, cy: float, cz: float, d: float = 2.0) -> list[float]:
    return [cx, cy, cz, d, d, d, 0.0]


class TestIou:
    def test_identical_boxes_iou_one(self) -> None:
        gt = np.array(_box(0.0, 0.0, 0.0))
        boxes = np.array([_box(0.0, 0.0, 0.0)])
        assert abs(float(iou_3d_axis_aligned(boxes, gt)[0]) - 1.0) < 1e-9

    def test_disjoint_boxes_iou_zero(self) -> None:
        gt = np.array(_box(0.0, 0.0, 0.0))
        boxes = np.array([_box(10.0, 10.0, 10.0)])
        assert float(iou_3d_axis_aligned(boxes, gt)[0]) == 0.0

    def test_half_overlap(self) -> None:
        # Shift by half the box size along x: intersection 1/2 vol, union 3/2.
        gt = np.array(_box(0.0, 0.0, 0.0))
        boxes = np.array([_box(1.0, 0.0, 0.0)])
        assert abs(float(iou_3d_axis_aligned(boxes, gt)[0]) - 1.0 / 3.0) < 1e-9


class TestAveragePrecision:
    def test_perfect_detector_ap_one(self) -> None:
        flags = np.array([True, True])
        scores = np.array([0.9, 0.8])
        assert abs(average_precision(flags, scores, total_gt=2) - 1.0) < 1e-9

    def test_all_misses_ap_zero(self) -> None:
        flags = np.array([False, False])
        scores = np.array([0.9, 0.8])
        assert average_precision(flags, scores, total_gt=2) == 0.0

    def test_no_gt_is_nan(self) -> None:
        assert np.isnan(average_precision(np.array([]), np.array([]), total_gt=0))


def _detections(
    boxes: list[list[float]], scores: list[float], classes: list[int]
) -> dict[str, np.ndarray]:
    return {
        "boxes": np.array(boxes),
        "scores": np.array(scores),
        "classes": np.array(classes, dtype=np.int64),
    }


class TestPostprocess:
    def test_confidence_filter_drops_low_scores(self) -> None:
        det = _detections([_box(0.0, 0.0, 0.0), _box(10.0, 0.0, 0.0)], [0.9, 0.4], [1, 2])
        out = postprocess(det, score_threshold=0.5, nms_iou=0.4)
        assert out["boxes"].shape == (1, 7)
        assert out["classes"].tolist() == [1]

    def test_nms_suppresses_near_duplicates(self) -> None:
        # Boxes 1 and 2 nearly coincide (IoU ≈ 0.9 > 0.4): only the higher
        # score survives; the distant third box is untouched.
        det = _detections(
            [_box(0.0, 0.0, 0.0), _box(0.1, 0.0, 0.0), _box(50.0, 0.0, 0.0)],
            [0.8, 0.9, 0.6],
            [1, 2, 3],
        )
        out = postprocess(det, score_threshold=0.5, nms_iou=0.4)
        assert out["scores"].tolist() == [0.9, 0.6]
        assert out["classes"].tolist() == [2, 3]

    def test_nms_is_class_agnostic(self) -> None:
        # Coincident boxes with different classes still suppress each other,
        # mirroring the Rust decode.
        det = _detections([_box(0.0, 0.0, 0.0), _box(0.0, 0.0, 0.0)], [0.9, 0.8], [1, 2])
        out = postprocess(det, score_threshold=0.5, nms_iou=0.4)
        assert out["boxes"].shape[0] == 1
        assert out["classes"].tolist() == [1]

    def test_empty_when_all_below_threshold(self) -> None:
        det = _detections([_box(0.0, 0.0, 0.0)], [0.3], [1])
        out = postprocess(det, score_threshold=0.5, nms_iou=0.4)
        assert out["boxes"].shape[0] == 0
        assert out["scores"].shape[0] == 0
        assert out["classes"].shape[0] == 0

    def test_output_sorted_by_descending_score(self) -> None:
        det = _detections(
            [_box(0.0, 0.0, 0.0), _box(50.0, 0.0, 0.0), _box(100.0, 0.0, 0.0)],
            [0.6, 0.9, 0.7],
            [1, 2, 3],
        )
        out = postprocess(det, score_threshold=0.5, nms_iou=0.4)
        assert out["scores"].tolist() == [0.9, 0.7, 0.6]
        assert out["classes"].tolist() == [2, 3, 1]


def _samples_one_snapshot(gt_centres: list[list[float]], classes: list[int]) -> DetectorSamples:
    n_boxes = 100
    gt_boxes = np.zeros((1, n_boxes, 7), dtype=np.float32)
    gt_valid = np.zeros((1, n_boxes), dtype=bool)
    gt_classes = np.zeros((1, n_boxes), dtype=np.int64)
    for j, (centre, cls) in enumerate(zip(gt_centres, classes, strict=True)):
        gt_boxes[0, j] = np.array(_box(*centre), dtype=np.float32)
        gt_valid[0, j] = True
        gt_classes[0, j] = cls
    return DetectorSamples(
        point_clouds=np.zeros((1, 1000, 4), dtype=np.float32),
        gt_boxes=gt_boxes,
        gt_valid=gt_valid,
        gt_classes=gt_classes,
        trajectory_ids=np.zeros(1, dtype=np.uint32),
    )


class TestDistanceGatedAp:
    def test_exact_hit_scores_one_at_every_gate(self) -> None:
        samples = _samples_one_snapshot([[0.0, 0.0, 0.0]], [1])
        detections = [_detections([_box(0.0, 0.0, 0.0)], [0.9], [1])]
        result = distance_gated_ap(samples, detections, noise_sigma_m=5.0)
        assert result["micro_distance_ap"] == 1.0
        assert result["distance_ap_per_gate"] == {"2.5m": 1.0, "5m": 1.0, "10m": 1.0}

    def test_three_metre_miss_passes_only_wider_gates(self) -> None:
        # 3 m offset: outside the 2.5 m gate, inside 5 m and 10 m → micro 2/3.
        samples = _samples_one_snapshot([[0.0, 0.0, 0.0]], [1])
        detections = [_detections([_box(3.0, 0.0, 0.0)], [0.9], [1])]
        result = distance_gated_ap(samples, detections, noise_sigma_m=5.0)
        per_gate = cast(dict[str, float], result["distance_ap_per_gate"])
        assert per_gate["2.5m"] == 0.0
        assert per_gate["5m"] == 1.0
        assert per_gate["10m"] == 1.0
        assert abs(cast(float, result["micro_distance_ap"]) - 2.0 / 3.0) < 1e-9

    def test_matching_is_class_agnostic(self) -> None:
        # Wrong class must not prevent a distance match (micro pools classes).
        samples = _samples_one_snapshot([[0.0, 0.0, 0.0]], [1])
        detections = [_detections([_box(0.0, 0.0, 0.0)], [0.9], [3])]
        result = distance_gated_ap(samples, detections, noise_sigma_m=5.0)
        assert result["micro_distance_ap"] == 1.0

    def test_high_scoring_false_positive_halves_ap(self) -> None:
        # FP outranks the TP: precision at full recall is 0.5 at every gate.
        samples = _samples_one_snapshot([[0.0, 0.0, 0.0]], [1])
        detections = [
            _detections(
                [_box(500.0, 0.0, 0.0), _box(0.0, 0.0, 0.0)], [0.9, 0.8], [1, 1]
            )
        ]
        result = distance_gated_ap(samples, detections, noise_sigma_m=5.0)
        assert abs(cast(float, result["micro_distance_ap"]) - 0.5) < 1e-9

    def test_one_gt_matches_at_most_once(self) -> None:
        # Two detections on one GT: the higher score takes it, the other is FP.
        samples = _samples_one_snapshot([[0.0, 0.0, 0.0]], [1])
        detections = [
            _detections([_box(0.0, 0.0, 0.0), _box(1.0, 0.0, 0.0)], [0.9, 0.8], [1, 1])
        ]
        result = distance_gated_ap(samples, detections, noise_sigma_m=5.0)
        # TP first: AP stays 1.0 under all-points interpolation.
        assert result["micro_distance_ap"] == 1.0

    def test_evaluate_includes_distance_metrics(self) -> None:
        samples = _samples_one_snapshot([[0.0, 0.0, 0.0]], [1])
        detections = [_detections([_box(0.0, 0.0, 0.0)], [0.9], [1])]
        result = evaluate(samples, detections)
        assert result["micro_distance_ap"] == 1.0
        assert "distance_ap_per_gate" in result


class TestEvaluate:
    def test_perfect_predictions_score_perfectly(self) -> None:
        samples = _samples_one_snapshot([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]], [1, 2])
        detections = [
            {
                "boxes": np.array([_box(0.0, 0.0, 0.0), _box(10.0, 0.0, 0.0)]),
                "scores": np.array([0.9, 0.8]),
                "classes": np.array([1, 2], dtype=np.int64),
            }
        ]
        result = evaluate(samples, detections)
        assert abs(float(result["map_50"]) - 1.0) < 1e-9  # type: ignore[arg-type]
        assert result["class_accuracy"] == 1.0
        assert result["matched_detections"] == 2

    def test_wrong_classes_zero_map_but_boxes_still_match(self) -> None:
        samples = _samples_one_snapshot([[0.0, 0.0, 0.0]], [1])
        detections = [
            {
                "boxes": np.array([_box(0.0, 0.0, 0.0)]),
                "scores": np.array([0.9]),
                "classes": np.array([3], dtype=np.int64),  # wrong class
            }
        ]
        result = evaluate(samples, detections)
        # Per-class AP for class 1 sees zero predictions → AP 0.
        assert float(result["map_50"]) == 0.0  # type: ignore[arg-type]
        # Class-agnostic box match still lands, but the class is wrong.
        assert result["matched_detections"] == 1
        assert result["class_accuracy"] == 0.0
