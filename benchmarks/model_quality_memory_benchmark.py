"""Benchmark fraud models by accuracy and memory footprint across CPU and GPU.

Expected input file:
  data/creditcard.csv

This benchmark compares a compact logistic model against a wider MLP on:
  - accuracy
  - precision
  - recall
  - F1 score
  - parameter count
  - model size in MB
  - peak CUDA memory in MB when CUDA is available

The benchmark can run on CPU, GPU, or both. It does not report runtime speed.
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


class CompactLogisticModel(nn.Module):
    def __init__(self, num_features: int) -> None:
        super().__init__()
        self.linear = nn.Linear(num_features, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x).squeeze(1)


class WiderMLPModel(nn.Module):
    def __init__(self, num_features: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(num_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x).squeeze(1)


def load_dataset(csv_path: Path) -> tuple[torch.Tensor, torch.Tensor]:
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset not found: {csv_path}")

    df = pd.read_csv(csv_path)
    missing = [column for column in FEATURE_COLUMNS + [TARGET_COLUMN] if column not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns: {missing}")

    x = torch.tensor(df[FEATURE_COLUMNS].values, dtype=torch.float32)
    y = torch.tensor(df[TARGET_COLUMN].values, dtype=torch.float32)
    x = (x - x.mean(dim=0)) / (x.std(dim=0) + 1e-6)
    return x, y


def split_train_test(x: torch.Tensor, y: torch.Tensor, train_ratio: float = 0.8):
    generator = torch.Generator().manual_seed(42)
    indices = torch.randperm(x.shape[0], generator=generator)
    split = int(train_ratio * x.shape[0])
    train_idx, test_idx = indices[:split], indices[split:]
    return x[train_idx], y[train_idx], x[test_idx], y[test_idx]


def build_models(num_features: int) -> list[tuple[str, nn.Module]]:
    return [
        ("Compact Logistic", CompactLogisticModel(num_features)),
        ("Wider MLP", WiderMLPModel(num_features)),
    ]


def train_model(model: nn.Module, x_train: torch.Tensor, y_train: torch.Tensor, epochs: int, lr: float) -> nn.Module:
    negative_count = (y_train == 0).sum()
    positive_count = (y_train == 1).sum()
    pos_weight = negative_count / (positive_count + 1e-6)

    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for _ in range(epochs):
        model.train()
        optimizer.zero_grad()
        logits = model(x_train)
        loss = loss_fn(logits, y_train)
        loss.backward()
        optimizer.step()

    return model


def evaluate_model(
    device_name: str,
    name: str,
    model: nn.Module,
    x_test: torch.Tensor,
    y_test: torch.Tensor,
) -> dict[str, float | int | str]:
    reset_peak_memory(device_name)
    model.eval()
    with torch.no_grad():
        logits = model(x_test)
        metrics = binary_classification_metrics(logits, y_test)

    return {
        "device": device_name,
        "model": name,
        "accuracy": metrics.accuracy,
        "precision": metrics.precision,
        "recall": metrics.recall,
        "f1": metrics.f1,
        "parameters": parameter_count(model),
        "model_size_mb": model_size_mb(model),
        "peak_memory_mb": peak_memory_mb(device_name),
    }


def resolve_devices(mode: str) -> list[str]:
    if mode == "cpu":
        return ["cpu"]
    if mode == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but torch.cuda.is_available() is False.")
        return ["cuda"]
    if mode == "both":
        devices = ["cpu"]
        if torch.cuda.is_available():
            devices.append("cuda")
        return devices
    raise ValueError(f"Unsupported device mode: {mode}")


def run_for_device(
    device_name: str,
    x: torch.Tensor,
    y: torch.Tensor,
    epochs: int,
    learning_rate: float,
) -> list[dict[str, float | int | str]]:
    torch.manual_seed(42)
    x_train, y_train, x_test, y_test = split_train_test(x, y)

    x_train = x_train.to(device_name)
    y_train = y_train.to(device_name)
    x_test = x_test.to(device_name)
    y_test = y_test.to(device_name)

    rows = []
    for name, model in build_models(x_train.shape[1]):
        model = model.to(device_name)
        reset_peak_memory(device_name)
        trained_model = train_model(model, x_train, y_train, epochs, learning_rate)
        rows.append(evaluate_model(device_name, name, trained_model, x_test, y_test))
    return rows


def print_markdown_table(rows: list[dict[str, float | int | str]]) -> None:
    print("| Device | Model | Accuracy | Precision | Recall | F1 | Parameters | Model size MB | Peak memory MB |")
    print("|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for row in rows:
        print(
            f"| {row['device']} | "
            f"{row['model']} | "
            f"{row['accuracy']:.4f} | "
            f"{row['precision']:.4f} | "
            f"{row['recall']:.4f} | "
            f"{row['f1']:.4f} | "
            f"{row['parameters']} | "
            f"{row['model_size_mb']:.6f} | "
            f"{row['peak_memory_mb']:.2f} |"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark model accuracy and memory footprint on CPU and GPU.")
    parser.add_argument("--csv", type=Path, default=Path("data/creditcard.csv"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda", "both"],
        default="both",
        help="Run on CPU, CUDA GPU, or both. Default runs CPU and GPU when CUDA is available.",
    )
    args = parser.parse_args()

    x, y = load_dataset(args.csv)
    devices = resolve_devices(args.device)

    all_results = []
    for device_name in devices:
        all_results.extend(run_for_device(device_name, x, y, args.epochs, args.learning_rate))

    print("Rows:", x.shape[0])
    print("Fraud labels:", int(y.sum().item()))
    print("Devices evaluated:", ", ".join(devices))
    if "cuda" not in devices:
        print("CUDA evaluated: No")
    print_markdown_table(all_results)


if __name__ == "__main__":
    main()
