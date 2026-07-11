"""Train a compact fraud classifier on the public ULB credit-card dataset.

Expected input file:
  data/creditcard.csv

Expected columns:
  Time, V1..V28, Amount, Class
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
from torch import nn

from src.metrics import (
    binary_classification_metrics,
    model_size_mb,
    parameter_count,
    peak_memory_mb,
    reset_peak_memory,
)

FEATURE_COLUMNS = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]
TARGET_COLUMN = "Class"


class LogisticFraudModel(nn.Module):
    def __init__(self, num_features: int) -> None:
        super().__init__()
        self.linear = nn.Linear(num_features, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x).squeeze(1)


def load_dataset(csv_path: Path) -> tuple[torch.Tensor, torch.Tensor]:
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {csv_path}. Download creditcard.csv and place it there."
        )

    df = pd.read_csv(csv_path)
    missing = [column for column in FEATURE_COLUMNS + [TARGET_COLUMN] if column not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns: {missing}")

    x = torch.tensor(df[FEATURE_COLUMNS].values, dtype=torch.float32)
    y = torch.tensor(df[TARGET_COLUMN].values, dtype=torch.float32)
    return x, y


def split_train_test(
    x: torch.Tensor,
    y: torch.Tensor,
    train_ratio: float = 0.8,
    seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create a deterministic stratified split for the binary target."""
    if not 0.0 < train_ratio < 1.0:
        raise ValueError("train_ratio must be between 0 and 1")
    if x.shape[0] != y.shape[0]:
        raise ValueError("x and y must contain the same number of rows")

    generator = torch.Generator().manual_seed(seed)
    train_parts: list[torch.Tensor] = []
    test_parts: list[torch.Tensor] = []

    for class_value in (0.0, 1.0):
        indices = torch.nonzero(y == class_value, as_tuple=False).squeeze(1)
        if indices.numel() < 2:
            raise ValueError(f"Class {int(class_value)} requires at least two rows")

        shuffled = indices[torch.randperm(indices.numel(), generator=generator)]
        split_at = max(1, min(int(train_ratio * indices.numel()), indices.numel() - 1))
        train_parts.append(shuffled[:split_at])
        test_parts.append(shuffled[split_at:])

    train_idx = torch.cat(train_parts)
    test_idx = torch.cat(test_parts)
    train_idx = train_idx[torch.randperm(train_idx.numel(), generator=generator)]
    test_idx = test_idx[torch.randperm(test_idx.numel(), generator=generator)]
    return x[train_idx], y[train_idx], x[test_idx], y[test_idx]


def normalize_from_training(
    x_train: torch.Tensor,
    x_test: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    mean = x_train.mean(dim=0)
    std = x_train.std(dim=0, unbiased=False).clamp_min(1e-6)
    return (x_train - mean) / std, (x_test - mean) / std


def train(csv_path: Path, epochs: int, learning_rate: float, seed: int) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    x, y = load_dataset(csv_path)
    x_train, y_train, x_test, y_test = split_train_test(x, y, seed=seed)
    x_train, x_test = normalize_from_training(x_train, x_test)

    x_train = x_train.to(device)
    y_train = y_train.to(device)
    x_test = x_test.to(device)
    y_test = y_test.to(device)

    torch.manual_seed(seed)
    model = LogisticFraudModel(x_train.shape[1]).to(device)

    negative_count = (y_train == 0).sum()
    positive_count = (y_train == 1).sum()
    pos_weight = negative_count / positive_count.clamp_min(1)

    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    reset_peak_memory(device)

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        logits = model(x_train)
        loss = loss_fn(logits, y_train)
        loss.backward()
        optimizer.step()

        if epoch == 1 or epoch % 5 == 0 or epoch == epochs:
            model.eval()
            with torch.no_grad():
                test_logits = model(x_test)
                metrics = binary_classification_metrics(test_logits, y_test)
            print(
                f"epoch={epoch:03d} loss={loss.item():.4f} "
                f"accuracy={metrics.accuracy:.4f} precision={metrics.precision:.4f} "
                f"recall={metrics.recall:.4f} f1={metrics.f1:.4f}"
            )

    print("Device:", device)
    print("Rows:", x.shape[0])
    print("Fraud labels:", int(y.sum().item()))
    print("Model parameters:", parameter_count(model))
    print("Model size MB:", f"{model_size_mb(model):.6f}")
    print("Peak CUDA memory MB:", f"{peak_memory_mb(device):.2f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the public credit-card fraud example.")
    parser.add_argument("--csv", type=Path, default=Path("data/creditcard.csv"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args.csv, args.epochs, args.learning_rate, args.seed)
