import pytest
import torch

from benchmarks.model_quality_memory_benchmark import normalize_from_training, split_train_test


def test_split_train_test_preserves_both_classes():
    x = torch.arange(60, dtype=torch.float32).reshape(12, 5)
    y = torch.tensor([0.0] * 8 + [1.0] * 4)

    x_train, y_train, x_test, y_test = split_train_test(x, y, train_ratio=0.75, seed=7)

    assert x_train.shape == (9, 5)
    assert x_test.shape == (3, 5)
    assert torch.bincount(y_train.to(torch.int64), minlength=2).tolist() == [6, 3]
    assert torch.bincount(y_test.to(torch.int64), minlength=2).tolist() == [2, 1]


def test_split_train_test_is_repeatable():
    x = torch.arange(40, dtype=torch.float32).reshape(10, 4)
    y = torch.tensor([0.0] * 6 + [1.0] * 4)

    first = split_train_test(x, y, seed=19)
    second = split_train_test(x, y, seed=19)

    assert all(torch.equal(left, right) for left, right in zip(first, second))


def test_normalization_uses_training_statistics_only():
    x_train = torch.tensor([[0.0, 5.0], [2.0, 5.0], [4.0, 5.0]])
    x_test = torch.tensor([[100.0, 5.0]])

    normalized_train, normalized_test = normalize_from_training(x_train, x_test)

    expected_train = torch.tensor(
        [
            [-1.2247449, 0.0],
            [0.0, 0.0],
            [1.2247449, 0.0],
        ]
    )
    expected_test = torch.tensor([[60.012497, 0.0]])

    assert torch.allclose(normalized_train, expected_train, atol=1e-5)
    assert torch.allclose(normalized_test, expected_test, atol=1e-5)


def test_split_rejects_a_class_with_one_row():
    x = torch.arange(20, dtype=torch.float32).reshape(5, 4)
    y = torch.tensor([0.0, 0.0, 0.0, 0.0, 1.0])

    with pytest.raises(ValueError, match="requires at least two rows"):
        split_train_test(x, y)
