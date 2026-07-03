"""Track A detector model (task 7.3) — a minimal 3DETR-style set predictor.

A small point encoder produces per-point features; a fixed set of learned object
queries cross-attends over them (DETR-style) and three heads emit, per query, a
7-DoF box, an objectness score logit, and class logits. Output shapes match the
ONNX detector contract: boxes ``(B, 100, 7)``, scores ``(B, 100, 1)``, class
logits ``(B, 100, NUM_CLASSES)``.

This is the scaffold the real 3DETR fine-tune (task 7.5) replaces; it is small
enough to smoke-test on CPU. Requires the ``training`` optional extra (torch);
excluded from the default CI type-check (pyproject ``[tool.pyright].exclude``).
"""

from __future__ import annotations

import torch
from torch import nn

from training.detector_dataset import BOX_DIM, MAX_BOXES, NUM_CLASSES, POINT_DIM, SCENE_SCALE_M


class DetectorModel(nn.Module):
    """Minimal point-cloud set-prediction detector."""

    def __init__(
        self,
        point_dim: int = POINT_DIM,
        d_model: int = 64,
        num_queries: int = MAX_BOXES,
        num_classes: int = NUM_CLASSES,
        num_heads: int = 4,
    ) -> None:
        super().__init__()
        self.point_encoder = nn.Sequential(
            nn.Linear(point_dim, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )
        # Learned object queries (DETR-style).
        self.query_embed = nn.Parameter(torch.randn(num_queries, d_model) * 0.1)
        self.cross_attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.box_head = nn.Linear(d_model, BOX_DIM)
        self.score_head = nn.Linear(d_model, 1)
        self.class_head = nn.Linear(d_model, num_classes)

    def forward(self, point_cloud: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # point_cloud: (B, NUM_POINTS, point_dim) in raw metres; xyz is scene-
        # normalized here so the core learns unit-scale geometry, and the box
        # head predicts in the same normalized space (see SCENE_SCALE_M).
        point_cloud = torch.cat(
            [point_cloud[..., :3] / SCENE_SCALE_M, point_cloud[..., 3:]], dim=-1
        )
        memory = self.point_encoder(point_cloud)  # (B, NUM_POINTS, d_model)
        batch = point_cloud.shape[0]
        queries = self.query_embed.unsqueeze(0).expand(batch, -1, -1)  # (B, Q, d)
        decoded, _ = self.cross_attn(queries, memory, memory)  # (B, Q, d)
        boxes = self.box_head(decoded)  # (B, Q, 7)
        score_logits = self.score_head(decoded)  # (B, Q, 1)
        class_logits = self.class_head(decoded)  # (B, Q, num_classes)
        return boxes, score_logits, class_logits


class DetectorExportWrapper(nn.Module):
    """Inference wrapper producing the ONNX 3-output contract: boxes
    ``(B, 100, 7)`` float, scores ``(B, 100, 1)`` float in [0, 1] (sigmoid), and
    classes ``(B, 100, 1)`` int64 (argmax of the class logits)."""

    def __init__(self, core: DetectorModel) -> None:
        super().__init__()
        self.core = core

    def forward(self, point_cloud: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        boxes, score_logits, class_logits = self.core(point_cloud)
        # The core predicts scene-normalized boxes; the deployed contract is
        # raw metres (centre + dims scale, yaw unchanged).
        boxes = torch.cat([boxes[..., :6] * SCENE_SCALE_M, boxes[..., 6:]], dim=-1)
        scores = torch.sigmoid(score_logits)
        classes = class_logits.argmax(dim=-1, keepdim=True).to(torch.int64)
        return boxes, scores, classes
