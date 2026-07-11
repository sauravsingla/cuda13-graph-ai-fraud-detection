import pytest
import torch

from src.metrics import binary_classification_metrics, model_size_mb, parameter_count


def test_binary_classification_metrics_counts_and_rates():
    probabilities = torch.tensor([0.9, 0.8, 0.7, 0.1])
    logits = torch.logit(probabilities)
    labels = torch.tensor([1.0, 0.0, 1.0, 0.0])

    metrics = binary_classification_metrics(logits, labels)

    assert metrics.true_positive == 2
    assert metrics.false_positive == 1
    assert metrics.false_negative == 0
    assert metrics.true_negative == 1
    assert metrics.accuracy == pytest.approx(0.75)
    assert metrics.precision == pytest.approx(2 / 3)
    assert metrics.recall == pytest.approx(1.0)
    assert metrics.f1 == pytest.approx(0.8)


def test_binary_classification_metrics_handles_no_positive_predictions():
    logits = torch.tensor([-10.0, -10.0])
    labels = torch.tensor([1.0, 0.0])

    metrics = binary_classification_metrics(logits, labels)

    assert metrics.precision == 0.0
    assert metrics.recall == 0.0
    assert metrics.f1 == 0.0


def test_model_footprint_helpers_include_parameters():
    model = torch.nn.Linear(3, 1)

    assert parameter_count(model) == 4
    assert model_size_mb(model) == pytest.approx(16 / (1024 * 1024))
