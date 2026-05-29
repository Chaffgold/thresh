"""IMM mode-classifier model (task 6.2).

A small sequence model: a 2-layer GRU over the filter-state window followed by a
linear head producing four mode logits (CV, CA, CTRV, coord_turn). Training uses
the raw logits with cross-entropy; the ONNX export (``export_imm.py``) appends a
softmax so the deployed graph emits a probability distribution.

Requires the ``training`` optional extra (torch). Excluded from the default CI
lane's type-check; see pyproject ``[tool.pyright].exclude``.
"""

from __future__ import annotations

import torch
from torch import nn

from training.imm_dataset import FEATURE_DIM, NUM_MODES


class ImmModeClassifier(nn.Module):
    """2-layer GRU + linear head.

    Input:  ``(batch, window, FEATURE_DIM)`` filter-state windows.
    Output: ``(batch, NUM_MODES)`` raw logits.
    """

    def __init__(
        self,
        feature_dim: int = FEATURE_DIM,
        hidden_dim: int = 64,
        num_layers: int = 2,
        num_classes: int = NUM_MODES,
    ) -> None:
        super().__init__()
        self.gru = nn.GRU(
            input_size=feature_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )
        self.head = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, window, feature_dim)
        out, _ = self.gru(x)
        last = out[:, -1, :]  # final timestep hidden state, (batch, hidden_dim)
        return self.head(last)  # (batch, num_classes) logits


class SoftmaxClassifier(nn.Module):
    """Inference wrapper that softmaxes the core model's logits, so the exported
    ONNX graph emits a probability distribution that sums to 1 per row."""

    def __init__(self, core: ImmModeClassifier) -> None:
        super().__init__()
        self.core = core

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.core(x), dim=1)
