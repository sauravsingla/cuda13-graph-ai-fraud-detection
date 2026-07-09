import torch
from torch import nn

from src.metrics import binary_classification_metrics, model_size_mb, parameter_count


def test_binary_classification_metrics():
    logits = torch.tensor([10.0, -10.0, 10.0, -10.0])
    labels = torch.tensor([1.0, 0.0, 0.0, 1.0])
    metrics = binary_classification_metrics(logits, labels)

    assert metrics.true_positive == 1
    assert metrics.true_negative == 1
    assert metrics.false_positive == 1
    assert metrics.false_negative == 1
    assert metrics.accuracy == 0.5
    assert metrics.precision == 0.5
    assert metrics.recall == 0.5
    assert metrics.f1 == 0.5


def test_model_memory_helpers():
    model = nn.Linear(10, 1)
    assert parameter_count(model) == 11
    assert model_size_mb(model) > 0
