"""Evaluation and memory-footprint helpers for model quality benchmarks."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class ClassificationMetrics:
    accuracy: float
    precision: float
    recall: float
    f1: float
    true_positive: int
    false_positive: int
    false_negative: int
    true_negative: int


def binary_classification_metrics(logits: torch.Tensor, labels: torch.Tensor, threshold: float = 0.5) -> ClassificationMetrics:
    """Compute binary classification metrics without requiring scikit-learn."""
    probabilities = torch.sigmoid(logits.detach())
    predictions = (probabilities >= threshold).float()
    labels = labels.detach().float()

    tp = int(((predictions == 1) & (labels == 1)).sum().item())
    fp = int(((predictions == 1) & (labels == 0)).sum().item())
    fn = int(((predictions == 0) & (labels == 1)).sum().item())
    tn = int(((predictions == 0) & (labels == 0)).sum().item())

    accuracy = (tp + tn) / max(tp + fp + fn + tn, 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = (2 * precision * recall) / max(precision + recall, 1e-12)

    return ClassificationMetrics(
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1=f1,
        true_positive=tp,
        false_positive=fp,
        false_negative=fn,
        true_negative=tn,
    )


def parameter_count(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def model_size_mb(model: torch.nn.Module) -> float:
    total_bytes = 0
    for parameter in model.parameters():
        total_bytes += parameter.numel() * parameter.element_size()
    for buffer in model.buffers():
        total_bytes += buffer.numel() * buffer.element_size()
    return total_bytes / (1024 * 1024)


def reset_peak_memory(device: str) -> None:
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def peak_memory_mb(device: str) -> float:
    if device == "cuda" and torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / (1024 * 1024)
    return 0.0
