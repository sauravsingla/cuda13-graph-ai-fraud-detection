import torch

from benchmarks.model_quality_memory_benchmark import normalize_from_training, split_train_test


def test_split_train_test_preserves_both_classes():
    x = torch.arange(60, dtype=torch.float32).reshape(12, 5)
    y = torch.tensor([0.0] * 8 + [1.0] * 4)

    x_train, y_train, x_test, y_test = split_train_test(x, y, train_ratio=0.75, seed=7)

    assert x_train.shape[0] == 9
    assert x_test.shape[0] == 3
    assert set(y_train.tolist()) == {0.0, 1.0}
    assert set(y_test.tolist()) == {0.0, 1.0}


def test_split_train_test_is_repeatable():
    x = torch.arange(40, dtype=torch.float32).reshape(10, 4)
    y = torch.tensor([0.0] * 6 + [1.0] * 4)

    first = split_train_test(x, y, seed=19)
    second = split_train_test(x, y, seed=19)

    for first_part, second_part in zip(first, second, strict=True):
        assert torch.equal(first_part, second_part)


def test_normalization_uses_training_statistics_only():
    x_train = torch.tensor([[0.0, 5.0], [2.0, 5.0], [4.0, 5.0]])
    x_test = torch.tensor([[100.0, 5.0]])

    normalized_train, normalized_test = normalize_from_training(x_train, x_test)

    assert torch.allclose(normalized_train.mean(dim=0), torch.zeros(2), atol=1e-6)
    assert torch.allclose(normalized_train[:, 1], torch.zeros(3))
    assert normalized_test[0, 0] > 50
    assert normalized_test[0, 1] == 0


def test_split_rejects_a_class_with_one_row():
    x = torch.arange(20, dtype=torch.float32).reshape(5, 4)
    y = torch.tensor([0.0, 0.0, 0.0, 0.0, 1.0])

    try:
        split_train_test(x, y)
    except ValueError as exc:
        assert "requires at least two rows" in str(exc)
    else:
        raise AssertionError("Expected ValueError for a class with one row")
