"""Detector holdout evaluation: mAP@0.5 (3D IoU) + class accuracy (task 7.9).

Evaluates the *exported ONNX* (the artifact `thresh-inference` deploys, not
the torch checkpoint) against a held-out detector-samples Parquet — per
design Decision 11: train on one region, evaluate on another.

    uv run --extra training python -m eval.detector_map \\
        --data ../python/data-training/detector-holdout.parquet \\
        --model ../test-data/models/test_detector.onnx

IoU is axis-aligned 3D (the same proxy the training loss and Decision 23
use — yaw is ignored by design). AP uses all-points interpolation per class;
mAP averages over classes that appear in the ground truth. Class accuracy is
computed over class-agnostic IoU≥0.5 matches so it is independent of the
classifier head's effect on matching.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from training.detector_dataset import DetectorSamples, load_detector_samples

IOU_THRESHOLD = 0.5


def _interval_overlap(
    centre_a: np.ndarray, size_a: np.ndarray, centre_b: float, size_b: float
) -> np.ndarray:
    lo = np.maximum(centre_a - size_a / 2.0, centre_b - size_b / 2.0)
    hi = np.minimum(centre_a + size_a / 2.0, centre_b + size_b / 2.0)
    return np.maximum(hi - lo, 0.0)


def iou_3d_axis_aligned(boxes: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """IoU of ``boxes`` (N, 7) against one ``gt`` box (7,), yaw ignored."""
    inter = (
        _interval_overlap(boxes[:, 0], boxes[:, 3], float(gt[0]), float(gt[3]))
        * _interval_overlap(boxes[:, 1], boxes[:, 4], float(gt[1]), float(gt[4]))
        * _interval_overlap(boxes[:, 2], boxes[:, 5], float(gt[2]), float(gt[5]))
    )
    vol_boxes = np.abs(boxes[:, 3] * boxes[:, 4] * boxes[:, 5])
    vol_gt = abs(float(gt[3]) * float(gt[4]) * float(gt[5]))
    union = np.maximum(vol_boxes + vol_gt - inter, 1e-6)
    return np.clip(inter / union, 0.0, 1.0)


def run_onnx(model_path: Path, point_clouds: np.ndarray) -> list[dict[str, np.ndarray]]:
    """Run the detector ONNX snapshot-by-snapshot (batch 1, the CI contract)."""
    import onnxruntime as ort  # heavy import kept local (training extra)

    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    detections: list[dict[str, np.ndarray]] = []
    for i in range(point_clouds.shape[0]):
        raw = session.run(None, {input_name: point_clouds[i : i + 1].astype(np.float32)})
        boxes = np.asarray(raw[0])
        scores = np.asarray(raw[1])
        classes = np.asarray(raw[2])
        detections.append(
            {
                "boxes": boxes[0],
                "scores": scores[0].reshape(-1),
                "classes": classes[0].reshape(-1).astype(np.int64),
            }
        )
    return detections


def postprocess(
    det: dict[str, np.ndarray], *, score_threshold: float, nms_iou: float
) -> dict[str, np.ndarray]:
    """Confidence filter + class-agnostic greedy NMS, mirroring the deployed
    Rust decode (`thresh_inference::DetectorConfig`: confidence 0.5, NMS IoU
    0.4). Raw query outputs are 100 near-duplicates per snapshot; evaluating
    them directly floods precision with false positives that deployment
    never emits."""
    keep = det["scores"] >= score_threshold
    boxes, scores, classes = det["boxes"][keep], det["scores"][keep], det["classes"][keep]
    order = np.argsort(-scores)
    boxes, scores, classes = boxes[order], scores[order], classes[order]
    kept: list[int] = []
    for i in range(boxes.shape[0]):
        if all(float(iou_3d_axis_aligned(boxes[i : i + 1], boxes[j])[0]) < nms_iou for j in kept):
            kept.append(i)
    idx = np.array(kept, dtype=np.int64)
    return {"boxes": boxes[idx], "scores": scores[idx], "classes": classes[idx]}


def average_precision(
    matched_flags: np.ndarray,
    scores: np.ndarray,
    total_gt: int,
) -> float:
    """All-points-interpolated AP from per-detection match flags and scores."""
    if total_gt == 0:
        return float("nan")
    if matched_flags.size == 0:
        return 0.0
    order = np.argsort(-scores)
    tp = matched_flags[order].astype(np.float64)
    fp = 1.0 - tp
    cum_tp = np.cumsum(tp)
    cum_fp = np.cumsum(fp)
    recall = cum_tp / total_gt
    precision = cum_tp / np.maximum(cum_tp + cum_fp, 1e-9)
    # All-points interpolation: precision envelope integrated over recall.
    ap = 0.0
    for threshold in np.unique(recall):
        mask = recall >= threshold
        ap += float(precision[mask].max()) if mask.any() else 0.0
    return ap / np.unique(recall).size if np.unique(recall).size else 0.0


def _greedy_match(
    detections: dict[str, np.ndarray],
    gt_boxes: np.ndarray,
    keep: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Greedily match kept detections (desc score) to GT at IoU≥0.5.

    Returns ``(matched_flags, matched_gt_index)`` aligned to the kept
    detections in descending-score order.
    """
    order = np.argsort(-detections["scores"][keep])
    det_boxes = detections["boxes"][keep][order]
    flags = np.zeros(det_boxes.shape[0], dtype=bool)
    gt_index = np.full(det_boxes.shape[0], -1, dtype=np.int64)
    taken = np.zeros(gt_boxes.shape[0], dtype=bool)
    for d in range(det_boxes.shape[0]):
        if gt_boxes.shape[0] == 0:
            break
        ious = iou_3d_axis_aligned(gt_boxes, det_boxes[d])
        ious[taken] = -1.0
        best = int(np.argmax(ious))
        if ious[best] >= IOU_THRESHOLD:
            flags[d] = True
            gt_index[d] = best
            taken[best] = True
    return flags, gt_index


def evaluate(
    samples: DetectorSamples, detections: list[dict[str, np.ndarray]]
) -> dict[str, object]:
    """Compute mAP@0.5 per class + class-agnostic-match class accuracy."""
    classes_in_gt = sorted(
        {int(c) for i in range(len(samples)) for c in samples.gt_classes[i][samples.gt_valid[i]]}
    )

    per_class_ap: dict[str, float] = {}
    for cls in classes_in_gt:
        flags_all: list[np.ndarray] = []
        scores_all: list[np.ndarray] = []
        total_gt = 0
        for i, det in enumerate(detections):
            gt_mask = samples.gt_valid[i] & (samples.gt_classes[i] == cls)
            gt_boxes = samples.gt_boxes[i][gt_mask]
            total_gt += int(gt_boxes.shape[0])
            keep = det["classes"] == cls
            flags, _ = _greedy_match(det, gt_boxes, keep)
            flags_all.append(flags)
            scores_all.append(np.sort(det["scores"][keep])[::-1])
        per_class_ap[str(cls)] = average_precision(
            np.concatenate(flags_all), np.concatenate(scores_all), total_gt
        )

    # Class accuracy over class-agnostic matches.
    correct = 0
    matched = 0
    for i, det in enumerate(detections):
        gt_mask = samples.gt_valid[i]
        gt_boxes = samples.gt_boxes[i][gt_mask]
        gt_classes = samples.gt_classes[i][gt_mask]
        keep = np.ones(det["classes"].shape[0], dtype=bool)
        order = np.argsort(-det["scores"])
        flags, gt_index = _greedy_match(det, gt_boxes, keep)
        det_classes = det["classes"][order]
        for d in range(flags.shape[0]):
            if flags[d]:
                matched += 1
                correct += int(det_classes[d] == gt_classes[gt_index[d]])

    valid_aps = [ap for ap in per_class_ap.values() if not np.isnan(ap)]
    return {
        "map_50": float(np.mean(valid_aps)) if valid_aps else 0.0,
        "per_class_ap": per_class_ap,
        "class_accuracy": (correct / matched) if matched else 0.0,
        "matched_detections": matched,
        "n_snapshots": len(samples),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Detector holdout mAP@0.5 (task 7.9).")
    parser.add_argument("--data", required=True, type=Path, help="holdout detector-samples Parquet")
    parser.add_argument("--model", required=True, type=Path, help="detector ONNX path")
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=0.5,
        help="confidence filter, matching the Rust DetectorConfig default (0.5)",
    )
    parser.add_argument(
        "--nms-iou",
        type=float,
        default=0.4,
        help="class-agnostic NMS IoU, matching the Rust DetectorConfig default (0.4)",
    )
    args = parser.parse_args()

    samples = load_detector_samples(args.data)
    if len(samples) == 0:
        raise SystemExit(f"no detector samples in {args.data}")
    detections = [
        postprocess(det, score_threshold=args.score_threshold, nms_iou=args.nms_iou)
        for det in run_onnx(args.model, samples.point_clouds)
    ]
    result = evaluate(samples, detections)
    result["detections_after_nms"] = int(sum(d["boxes"].shape[0] for d in detections))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
