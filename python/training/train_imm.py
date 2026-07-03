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


def _macro_recall(model: ImmModeClassifier, x: torch.Tensor, y: torch.Tensor) -> float:
    """Mean per-class recall over the classes present in ``y``.

    Used for checkpoint selection under ``--balanced``: plain accuracy
    selects majority-class behaviour on the heavily CV-skewed real data
    (a classifier that never predicts a turn still scores >0.9).
    """
    model.eval()
    with torch.no_grad():
        preds = model(x).argmax(dim=1)
    recalls = []
    for cls in torch.unique(y):
        mask = y == cls
        recalls.append(float((preds[mask] == cls).float().mean().item()))
    return float(sum(recalls) / len(recalls)) if recalls else 0.0


def train(
    windows: ImmWindows,
    *,
    epochs: int = 30,
    hidden_dim: int = 64,
    lr: float = 1e-3,
    batch_size: int = 64,
    seed: int = 42,
    device: str | None = None,
    balanced: bool = False,
    balance_power: float = 1.0,
) -> tuple[ImmModeClassifier, float]:
    """Train on ``windows`` with a trajectory-grouped split. Returns the best
    model (by held-out accuracy — macro recall under ``balanced`` — moved
    back to CPU for device-agnostic export) and its held-out accuracy."""
    resolved = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.manual_seed(seed)
    train_idx, test_idx = trajectory_split(windows, seed=seed)

    x = torch.from_numpy(windows.x)
    y = torch.from_numpy(windows.y)
    x_train, y_train = x[train_idx].to(resolved), y[train_idx].to(resolved)
    x_test, y_test = x[test_idx].to(resolved), y[test_idx].to(resolved)

    # Standardize features from the training set; the stats ride along in the
    # checkpoint and are baked into the exported graph (SoftmaxClassifier), so
    # the deployed contract still takes raw filter-state features.
    feature_mean = x_train.reshape(-1, x_train.shape[-1]).mean(dim=0)
    feature_std = x_train.reshape(-1, x_train.shape[-1]).std(dim=0).clamp(min=1e-6)
    x_train = (x_train - feature_mean) / feature_std
    if x_test.shape[0]:
        x_test = (x_test - feature_mean) / feature_std

    model = ImmModeClassifier(hidden_dim=hidden_dim).to(resolved)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    if balanced:
        # Inverse-frequency weights: real captures are >90% CV, and an
        # unweighted CE happily learns "always CV" (0 turn recall). The
        # exponent tempers the correction (1.0 = full inverse frequency,
        # 0.5 = sqrt — full weighting over-predicts the minority classes).
        counts = torch.bincount(y_train, minlength=4).float().clamp(min=1.0)
        weights = (y_train.shape[0] / (4.0 * counts)) ** balance_power
        loss_fn = nn.CrossEntropyLoss(weight=weights.to(resolved))
    else:
        loss_fn = nn.CrossEntropyLoss()

    best_metric = 0.0
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

        x_eval, y_eval = (x_test, y_test) if test_idx.size else (x_train, y_train)
        metric = (
            _macro_recall(model, x_eval, y_eval) if balanced else _accuracy(model, x_eval, y_eval)
        )
        if metric >= best_metric:
            best_metric = metric
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    best_acc = _accuracy(
        model.to(resolved),
        x_test if test_idx.size else x_train,
        y_test if test_idx.size else y_train,
    )
    model = model.cpu()
    # Plain attributes (not buffers): saved alongside the state dict by main()
    # and consumed by export_imm to bake normalization into the ONNX graph.
    model.feature_mean = feature_mean.cpu()
    model.feature_std = feature_std.cpu()
    return model, best_acc


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the IMM mode classifier.")
    parser.add_argument("--data", required=True, type=Path, help="IMM-samples Parquet path")
    parser.add_argument("--out", required=True, type=Path, help="Output checkpoint (.pt) path")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--window", type=int, default=WINDOW_LEN)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="torch device (default: cuda when available, else cpu)",
    )
    parser.add_argument(
        "--balanced",
        action="store_true",
        help="inverse-frequency class weights + macro-recall checkpoint selection "
        "(real captures are >90%% CV; unweighted CE learns 'always CV')",
    )
    parser.add_argument(
        "--balance-power",
        type=float,
        default=1.0,
        help="exponent tempering the inverse-frequency weights (0.5 = sqrt)",
    )
    args = parser.parse_args()

    windows = build_windows(args.data, window=args.window)
    if len(windows) == 0:
        raise SystemExit(f"no windows built from {args.data} (need ≥{args.window} steps per track)")

    model, best_acc = train(
        windows,
        epochs=args.epochs,
        hidden_dim=args.hidden_dim,
        seed=args.seed,
        device=args.device,
        balanced=args.balanced,
        balance_power=args.balance_power,
    )
    torch.save(
        {
            "state_dict": model.state_dict(),
            "hidden_dim": args.hidden_dim,
            "feature_mean": getattr(model, "feature_mean", None),
            "feature_std": getattr(model, "feature_std", None),
        },
        args.out,
    )
    print(f"trained on {len(windows)} windows; best held-out accuracy = {best_acc:.3f}")
    print(f"saved checkpoint → {args.out}")
    # Exit criterion 6.8 (first bullet): held-out accuracy ≥ 0.70.
    if best_acc < 0.70:
        print("WARNING: held-out accuracy below the 0.70 Track B exit criterion")


if __name__ == "__main__":
    main()
