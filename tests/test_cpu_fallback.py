import torch

from src.metrics import binary_classification_metrics, model_size_mb, parameter_count


def test_cpu_tensor_operation():
    x = torch.ones(10, 10)
    y = x @ x
    assert y.shape == (10, 10)
    assert torch.all(y == 10)


def test_binary_classification_metrics():
    logits = torch.tensor([4.0, 3.0, -2.0, -3.0])
    labels = torch.tensor([1.0, 0.0, 0.0, 1.0])

    metrics = binary_classification_metrics(logits, labels)

    assert metrics.true_positive == 1
    assert metrics.false_positive == 1
    assert metrics.false_negative == 1
    assert metrics.true_negative == 1
    assert metrics.precision == 0.5
    assert metrics.recall == 0.5
    assert metrics.f1 == 0.5


def test_model_footprint_helpers():
    model = torch.nn.Linear(3, 1)
    assert parameter_count(model) == 4
    assert model_size_mb(model) > 0
