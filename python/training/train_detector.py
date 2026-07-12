"""Train the Track A detector with a DETR set-prediction loss (task 7.3).

The loss is the standard DETR recipe adapted to 3D: a Hungarian matcher pairs
predicted queries to ground-truth boxes (cost = L1 box + class mismatch), then
the matched pairs incur an L1 box loss, a 3D-IoU loss, and a class
cross-entropy, while an objectness BCE pushes matched queries' scores to 1 and
the rest to 0.

The 3D-IoU term is an axis-aligned IoU (yaw-ignored, matching the Rust
`iou_3d`); the full GIoU-3D enclosing-box term is deferred to the real training
run (task 7.5), as is the GPU-scale fine-tune. This scaffold is CPU-smoke-only.

Per design Decision 7 this is a local / offline script (training is a non-CI
goal). Requires the ``training`` optional extra (torch, scipy).
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn

from training.detector_dataset import SCENE_SCALE_M, DetectorSamples, load_detector_samples
from training.detector_model import DIMS_PRIOR_M, DetectorModel

# Per-component L1 units (normalized space). Dividing box errors by these puts
# them at O(1) exactly when they are box-sized: a 100 m centre error or a
# 10 m dims error contributes ~1.0 to the loss. Without this the first real
# run stalled at 460 m centre error — box terms at 1e-3 normalized were
# invisible next to the saturated IoU and CE terms.
CENTRE_UNIT = 100.0 / SCENE_SCALE_M
DIMS_UNIT = DIMS_PRIOR_M / SCENE_SCALE_M

# Explicit cost/loss weights (DETR balances box vs. class with named
# coefficients rather than relying on implicit normalisation). The matcher cost
# and the training loss share the same box/class balance.
MATCH_COST_BOX = 1.0
MATCH_COST_CLASS = 1.0
LOSS_W_BOX = 1.0
LOSS_W_IOU = 1.0
LOSS_W_CLASS = 1.0
LOSS_W_OBJ = 1.0


def axis_aligned_iou_3d(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Axis-aligned 3D IoU between matched boxes ``a``/``b`` (``(M, 7)`` each,
    ``[x, y, z, L, W, H, yaw]``; yaw ignored). Returns ``(M,)`` in ``[0, 1]``."""

    def overlap(
        ca: torch.Tensor, ea: torch.Tensor, cb: torch.Tensor, eb: torch.Tensor
    ) -> torch.Tensor:
        amin, amax = ca - ea / 2, ca + ea / 2
        bmin, bmax = cb - eb / 2, cb + eb / 2
        return (torch.min(amax, bmax) - torch.max(amin, bmin)).clamp(min=0.0)

    ix = overlap(a[:, 0], a[:, 3], b[:, 0], b[:, 3])
    iy = overlap(a[:, 1], a[:, 4], b[:, 1], b[:, 4])
    iz = overlap(a[:, 2], a[:, 5], b[:, 2], b[:, 5])
    inter = ix * iy * iz
    vol_a = (a[:, 3] * a[:, 4] * a[:, 5]).abs()
    vol_b = (b[:, 3] * b[:, 4] * b[:, 5]).abs()
    union = (vol_a + vol_b - inter).clamp(min=1e-6)
    return (inter / union).clamp(0.0, 1.0)


def _match(
    pred_boxes: torch.Tensor,
    class_logits: torch.Tensor,
    gt_boxes: torch.Tensor,
    gt_classes: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Hungarian match queries → GT (cost = L1 box + class-mismatch). Returns
    ``(query_idx, gt_idx)`` long tensors of length ``M``."""
    with torch.no_grad():
        # Centre-distance cost in centre-prior units (a 100 m error costs ~1,
        # comparable to the class cost's [0, 1]); dims/yaw are excluded from
        # matching — assignment should be about *where*, not box shape.
        box_cost = torch.cdist(pred_boxes[:, :3], gt_boxes[:, :3], p=1) / (3.0 * CENTRE_UNIT)
        class_prob = torch.softmax(class_logits, dim=-1)  # (Q, C)
        class_cost = 1.0 - class_prob[:, gt_classes]  # (Q, M)
        cost = (MATCH_COST_BOX * box_cost + MATCH_COST_CLASS * class_cost).cpu().numpy()
    qi, gi = linear_sum_assignment(cost)
    return (
        torch.as_tensor(qi, dtype=torch.long, device=pred_boxes.device),
        torch.as_tensor(gi, dtype=torch.long, device=pred_boxes.device),
    )


def set_prediction_loss(
    boxes: torch.Tensor,
    score_logits: torch.Tensor,
    class_logits: torch.Tensor,
    gt_boxes: torch.Tensor,
    gt_classes: torch.Tensor,
) -> torch.Tensor:
    """DETR-style set-prediction loss for one snapshot.

    ``boxes`` ``(Q, 7)``, ``score_logits`` ``(Q, 1)``, ``class_logits``
    ``(Q, C)``; ``gt_boxes`` ``(M, 7)``, ``gt_classes`` ``(M,)``.
    """
    num_queries = boxes.shape[0]
    obj_target = torch.zeros(num_queries, 1, device=boxes.device)

    if gt_boxes.shape[0] == 0:
        # No ground truth: every query should be "no object".
        return nn.functional.binary_cross_entropy_with_logits(score_logits, obj_target)

    # The matcher returns min(Q, M) pairs. By construction M ≤ num_queries here
    # (gt_valid has MAX_BOXES=100 slots and num_queries defaults to MAX_BOXES),
    # so every ground-truth box is matched; assert the invariant defensively.
    assert gt_boxes.shape[0] <= num_queries, (
        "more GT boxes than queries (unmatched GT would be unsupervised)"
    )

    qi, gi = _match(boxes, class_logits, gt_boxes, gt_classes)
    matched_pred, matched_gt = boxes[qi], gt_boxes[gi]
    # L1 supervises all 7 DoF including yaw (the only yaw supervision); the IoU
    # term is an axis-aligned proxy that ignores yaw by design (Decision 23).
    # Centre/dims errors are expressed in box-prior units (see CENTRE_UNIT /
    # DIMS_UNIT) so the loss keeps gradient pressure down to metre scale.
    diff = (matched_pred - matched_gt).abs()
    # Yaw is periodic: wrap its residual to [-pi, pi] so a prediction near +pi
    # against a target near -pi costs ~0, not ~2*pi (torch.remainder returns
    # values in [0, 2*pi) for a positive divisor, so this holds for all inputs).
    yaw_res = (
        torch.remainder(matched_pred[:, 6] - matched_gt[:, 6] + math.pi, 2 * math.pi) - math.pi
    )
    box_l1 = (
        diff[:, :3].sum(dim=1) / (3.0 * CENTRE_UNIT)
        + diff[:, 3:6].sum(dim=1) / (3.0 * DIMS_UNIT)
        + yaw_res.abs()
    ).mean() / 3.0
    iou_loss = (1.0 - axis_aligned_iou_3d(matched_pred, matched_gt)).mean()
    class_loss = nn.functional.cross_entropy(class_logits[qi], gt_classes[gi])
    obj_target[qi] = 1.0
    obj_loss = nn.functional.binary_cross_entropy_with_logits(score_logits, obj_target)
    return (
        LOSS_W_BOX * box_l1
        + LOSS_W_IOU * iou_loss
        + LOSS_W_CLASS * class_loss
        + LOSS_W_OBJ * obj_loss
    )


def train(
    samples: DetectorSamples,
    *,
    epochs: int = 10,
    d_model: int = 32,
    lr: float = 1e-3,
    seed: int = 42,
    batch_size: int = 32,
    device: str | None = None,
) -> tuple[DetectorModel, float]:
    """Train the detector on ``samples`` with shuffled mini-batches.

    Returns ``(model, mean_final_epoch_loss)``; the model is moved back to
    CPU so export stays device-agnostic. The original scaffold ran one
    full-batch step per epoch, which only works for the tiny checked-in
    sample — real acquisition datasets need mini-batching.
    """
    resolved = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.manual_seed(seed)
    model = DetectorModel(d_model=d_model).to(resolved)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # Cosine decay to ~0: at a fixed 1e-3 the loss plateaus on optimizer
    # noise (~3.6 m centre error) while IoU@0.5 on aircraft-sized boxes
    # needs ~1.5-2 m — the tail of training must anneal.
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    point_clouds = torch.from_numpy(samples.point_clouds)
    n = len(samples)
    rng = np.random.default_rng(seed)

    final_loss = 0.0
    for epoch in range(epochs):
        model.train()
        order = rng.permutation(n)
        epoch_loss = 0.0
        for start in range(0, n, batch_size):
            idx = order[start : start + batch_size]
            batch_pcs = point_clouds[idx].to(resolved)
            all_boxes, all_scores, all_class_logits = model(batch_pcs)
            total = batch_pcs.new_zeros(())
            for k, i in enumerate(idx):
                mask = samples.gt_valid[i]
                gt_boxes = torch.from_numpy(samples.gt_boxes[i][mask]).float().to(resolved)
                # The model predicts scene-normalized boxes (see SCENE_SCALE_M);
                # supervise in the same space so the loss is unit-scale.
                gt_boxes = torch.cat([gt_boxes[..., :6] / SCENE_SCALE_M, gt_boxes[..., 6:]], dim=-1)
                gt_classes = torch.from_numpy(samples.gt_classes[i][mask]).long().to(resolved)
                total = total + set_prediction_loss(
                    all_boxes[k], all_scores[k], all_class_logits[k], gt_boxes, gt_classes
                )
            loss = total / max(len(idx), 1)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item()) * len(idx)
        scheduler.step()
        final_loss = epoch_loss / max(n, 1)
        print(f"epoch {epoch + 1}/{epochs}: loss {final_loss:.4f}", flush=True)

    return model.cpu(), final_loss


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the Track A detector (scaffold).")
    parser.add_argument("--data", required=True, type=Path, help="detector-samples Parquet path")
    parser.add_argument("--out", required=True, type=Path, help="output checkpoint (.pt) path")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="torch device (default: cuda when available, else cpu)",
    )
    args = parser.parse_args()

    samples = load_detector_samples(args.data)
    if len(samples) == 0:
        raise SystemExit(f"no detector samples in {args.data}")

    model, final_loss = train(
        samples,
        epochs=args.epochs,
        d_model=args.d_model,
        seed=args.seed,
        batch_size=args.batch_size,
        device=args.device,
    )
    torch.save({"state_dict": model.state_dict(), "d_model": args.d_model}, args.out)
    print(f"trained on {len(samples)} snapshots; final loss = {final_loss:.4f}")
    print(f"saved checkpoint → {args.out}")
    print(
        "NOTE: this is the Phase 7.3 scaffold on synthetic data; the Track A "
        "exit criterion (7.9: mAP@0.5 ≥ 0.30 + MOTA) needs the real GPU run (7.5)."
    )


if __name__ == "__main__":
    main()
