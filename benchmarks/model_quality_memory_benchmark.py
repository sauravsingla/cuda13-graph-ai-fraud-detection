"""Benchmark fraud models across CPU, older CUDA GPU, and latest CUDA GPU.

Expected input file:
  data/creditcard.csv

The benchmark reports classification quality and model-footprint metrics that can
be reproduced across CPU and CUDA environments. Feature normalization is fitted
on the training partition only so the held-out set does not influence training.
"""

from __future__ import annotations

import argparse
import platform
from pathlib import Path

import pandas as pd
import torch
from torch import nn

from src.metrics import binary_classification_metrics, model_size_mb, parameter_count

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
    return x, y


def _class_indices(y: torch.Tensor, class_value: float) -> torch.Tensor:
    return torch.nonzero(y == class_value, as_tuple=False).squeeze(1)


def split_train_test(
    x: torch.Tensor,
    y: torch.Tensor,
    train_ratio: float = 0.8,
    seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create a deterministic stratified split for an imbalanced binary target."""
    if not 0.0 < train_ratio < 1.0:
        raise ValueError("train_ratio must be between 0 and 1")
    if x.shape[0] != y.shape[0]:
        raise ValueError("x and y must contain the same number of rows")

    generator = torch.Generator().manual_seed(seed)
    train_parts = []
    test_parts = []

    for class_value in (0.0, 1.0):
        indices = _class_indices(y, class_value)
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
    """Normalize both partitions using statistics fitted on the training data."""
    mean = x_train.mean(dim=0)
    std = x_train.std(dim=0, unbiased=False).clamp_min(1e-6)
    return (x_train - mean) / std, (x_test - mean) / std


def build_models(num_features: int) -> list[tuple[str, nn.Module]]:
    return [
        ("Compact Logistic", CompactLogisticModel(num_features)),
        ("Wider MLP", WiderMLPModel(num_features)),
    ]


def train_model(model: nn.Module, x_train: torch.Tensor, y_train: torch.Tensor, epochs: int, lr: float) -> nn.Module:
    negative_count = (y_train == 0).sum()
    positive_count = (y_train == 1).sum()
    pos_weight = negative_count / positive_count.clamp_min(1)

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


def environment_summary(label: str) -> dict[str, str]:
    gpu_name = "not-available"
    cuda_device_count = "0"
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        cuda_device_count = str(torch.cuda.device_count())

    return {
        "benchmark_label": label,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda": str(torch.version.cuda),
        "cuda_available": str(torch.cuda.is_available()),
        "cuda_device_count": cuda_device_count,
        "gpu_name": gpu_name,
    }


def evaluate_model(
    benchmark_label: str,
    device_name: str,
    name: str,
    model: nn.Module,
    x_test: torch.Tensor,
    y_test: torch.Tensor,
) -> dict[str, float | int | str]:
    model.eval()
    with torch.no_grad():
        logits = model(x_test)
        metrics = binary_classification_metrics(logits, y_test)

    return {
        "benchmark_label": benchmark_label,
        "device": device_name,
        "model": name,
        "accuracy": metrics.accuracy,
        "precision": metrics.precision,
        "recall": metrics.recall,
        "f1": metrics.f1,
        "parameters": parameter_count(model),
        "model_size_mb": model_size_mb(model),
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
    benchmark_label: str,
    device_name: str,
    x: torch.Tensor,
    y: torch.Tensor,
    epochs: int,
    learning_rate: float,
    seed: int,
) -> list[dict[str, float | int | str]]:
    torch.manual_seed(seed)
    x_train, y_train, x_test, y_test = split_train_test(x, y, seed=seed)
    x_train, x_test = normalize_from_training(x_train, x_test)

    x_train = x_train.to(device_name)
    y_train = y_train.to(device_name)
    x_test = x_test.to(device_name)
    y_test = y_test.to(device_name)

    rows = []
    for name, model in build_models(x_train.shape[1]):
        model = model.to(device_name)
        trained_model = train_model(model, x_train, y_train, epochs, learning_rate)
        rows.append(evaluate_model(benchmark_label, device_name, name, trained_model, x_test, y_test))
    return rows


def render_environment(summary: dict[str, str]) -> str:
    lines = ["## Environment"]
    lines.extend(f"{key}: {value}" for key, value in summary.items())
    lines.append("")
    return "\n".join(lines)


def render_markdown_table(rows: list[dict[str, float | int | str]]) -> str:
    lines = [
        "| Label | Device | Model | Accuracy | Precision | Recall | F1 | Parameters | Model size MB |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['benchmark_label']} | "
            f"{row['device']} | "
            f"{row['model']} | "
            f"{row['accuracy']:.4f} | "
            f"{row['precision']:.4f} | "
            f"{row['recall']:.4f} | "
            f"{row['f1']:.4f} | "
            f"{row['parameters']} | "
            f"{row['model_size_mb']:.6f} |"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark model quality and footprint across CPU and CUDA environments.")
    parser.add_argument("--csv", type=Path, default=Path("data/creditcard.csv"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42, help="Random seed used for split and model initialization.")
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda", "both"],
        default="both",
        help="Run on CPU, CUDA GPU, or both. Default runs CPU and GPU when CUDA is available.",
    )
    parser.add_argument(
        "--label",
        default="local-run",
        help="Benchmark label, for example cpu-baseline, cuda-12-old, or cuda-13-latest.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to save the rendered markdown benchmark output.",
    )
    args = parser.parse_args()

    x, y = load_dataset(args.csv)
    devices = resolve_devices(args.device)

    all_results = []
    for device_name in devices:
        all_results.extend(run_for_device(args.label, device_name, x, y, args.epochs, args.learning_rate, args.seed))

    sections = [
        render_environment(environment_summary(args.label)),
        f"Rows: {x.shape[0]}",
        f"Fraud labels: {int(y.sum().item())}",
        f"Devices evaluated: {', '.join(devices)}",
        "Split: stratified random holdout",
        "Normalization: fitted on training rows only",
    ]
    if "cuda" not in devices:
        sections.append("CUDA evaluated: No")
    sections.append(render_markdown_table(all_results))

    rendered_output = "\n".join(sections) + "\n"
    print(rendered_output)

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered_output, encoding="utf-8")


if __name__ == "__main__":
    main()
