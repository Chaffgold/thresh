"""Train the IMM mode classifier (task 6.3).

Fixed-seed, trajectory-grouped train/test split + cross-entropy on the raw
logits. Saves the best checkpoint (by held-out accuracy) as a torch state dict.

Per design Decision 7, training does **not** run in CI; this is a local /
offline script. Run it against the full external dataset (or the checked-in
sample for a smoke run):

    uv run --extra training python -m training.train_imm \\
        --data ../test-data/training/imm-classifier/imm-samples.parquet \\
        --out best_imm.pt --epochs 30

Requires the ``training`` optional extra (torch).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch import nn

from training.imm_dataset import WINDOW_LEN, ImmWindows, build_windows, trajectory_split
from training.imm_model import ImmModeClassifier


def _accuracy(model: ImmModeClassifier, x: torch.Tensor, y: torch.Tensor) -> float:
    model.eval()
    with torch.no_grad():
        preds = model(x).argmax(dim=1)
    return float((preds == y).float().mean().item())


def train(
    windows: ImmWindows,
    *,
    epochs: int = 30,
    hidden_dim: int = 64,
    lr: float = 1e-3,
    batch_size: int = 64,
    seed: int = 42,
) -> tuple[ImmModeClassifier, float]:
    """Train on ``windows`` with a trajectory-grouped split. Returns the best
    model (by held-out accuracy) and that accuracy."""
    torch.manual_seed(seed)
    train_idx, test_idx = trajectory_split(windows, seed=seed)

    x = torch.from_numpy(windows.x)
    y = torch.from_numpy(windows.y)
    x_train, y_train = x[train_idx], y[train_idx]
    x_test, y_test = x[test_idx], y[test_idx]

    model = ImmModeClassifier(hidden_dim=hidden_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    best_acc = 0.0
    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    rng = np.random.default_rng(seed)
    n = x_train.shape[0]

    for _ in range(epochs):
        model.train()
        order = rng.permutation(n)
        for start in range(0, n, batch_size):
            batch = order[start : start + batch_size]
            optimizer.zero_grad()
            logits = model(x_train[batch])
            loss = loss_fn(logits, y_train[batch])
            loss.backward()
            optimizer.step()

        if test_idx.size:
            acc = _accuracy(model, x_test, y_test)
        else:
            acc = _accuracy(model, x_train, y_train)
        if acc >= best_acc:
            best_acc = acc
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    return model, best_acc


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the IMM mode classifier.")
    parser.add_argument("--data", required=True, type=Path, help="IMM-samples Parquet path")
    parser.add_argument("--out", required=True, type=Path, help="Output checkpoint (.pt) path")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--window", type=int, default=WINDOW_LEN)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    windows = build_windows(args.data, window=args.window)
    if len(windows) == 0:
        raise SystemExit(f"no windows built from {args.data} (need ≥{args.window} steps per track)")

    model, best_acc = train(windows, epochs=args.epochs, hidden_dim=args.hidden_dim, seed=args.seed)
    torch.save({"state_dict": model.state_dict(), "hidden_dim": args.hidden_dim}, args.out)
    print(f"trained on {len(windows)} windows; best held-out accuracy = {best_acc:.3f}")
    print(f"saved checkpoint → {args.out}")
    # Exit criterion 6.8 (first bullet): held-out accuracy ≥ 0.70.
    if best_acc < 0.70:
        print("WARNING: held-out accuracy below the 0.70 Track B exit criterion")


if __name__ == "__main__":
    main()
