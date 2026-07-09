"""Train a compact GPU-friendly fraud classifier on the public ULB/Kaggle credit-card fraud dataset.

Expected input file:
  data/creditcard.csv

Dataset columns are expected to include:
  Time, V1..V28, Amount, Class

The dataset is intentionally not committed to this repository. Download it from the public dataset host and place it under data/creditcard.csv.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

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
    missing = [col for col in FEATURE_COLUMNS + [TARGET_COLUMN] if col not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns: {missing}")

    x = torch.tensor(df[FEATURE_COLUMNS].values, dtype=torch.float32)
    y = torch.tensor(df[TARGET_COLUMN].values, dtype=torch.float32)

    # Standardize features for stable optimization.
    x = (x - x.mean(dim=0)) / (x.std(dim=0) + 1e-6)
    return x, y


def split_train_test(x: torch.Tensor, y: torch.Tensor, train_ratio: float = 0.8):
    generator = torch.Generator().manual_seed(42)
    indices = torch.randperm(x.shape[0], generator=generator)
    split = int(train_ratio * x.shape[0])
    train_idx, test_idx = indices[:split], indices[split:]
    return x[train_idx], y[train_idx], x[test_idx], y[test_idx]


def train(csv_path: Path, epochs: int, learning_rate: float) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    x, y = load_dataset(csv_path)
    x_train, y_train, x_test, y_test = split_train_test(x, y)

    x_train = x_train.to(device)
    y_train = y_train.to(device)
    x_test = x_test.to(device)
    y_test = y_test.to(device)

    model = LogisticFraudModel(x_train.shape[1]).to(device)

    negative_count = (y_train == 0).sum()
    positive_count = (y_train == 1).sum()
    pos_weight = negative_count / (positive_count + 1e-6)

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
    parser = argparse.ArgumentParser(description="Run public credit-card fraud GPU example.")
    parser.add_argument("--csv", type=Path, default=Path("data/creditcard.csv"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args.csv, args.epochs, args.learning_rate)
