import torch

from examples.public_creditcard_fraud_gpu import normalize_from_training, split_train_test


def test_creditcard_split_keeps_both_classes_in_each_partition():
    x = torch.arange(48, dtype=torch.float32).reshape(12, 4)
    y = torch.tensor([0.0] * 8 + [1.0] * 4)

    _, y_train, _, y_test = split_train_test(x, y, train_ratio=0.75, seed=11)

    assert set(y_train.tolist()) == {0.0, 1.0}
    assert set(y_test.tolist()) == {0.0, 1.0}


def test_creditcard_normalization_uses_training_statistics():
    x_train = torch.tensor([[0.0, 4.0], [2.0, 4.0], [4.0, 4.0]])
    x_test = torch.tensor([[20.0, 4.0]])

    normalized_train, normalized_test = normalize_from_training(x_train, x_test)

    assert torch.allclose(normalized_train.mean(dim=0), torch.zeros(2), atol=1e-6)
    assert torch.equal(normalized_train[:, 1], torch.zeros(3))
    assert normalized_test[0, 0] > 5
    assert normalized_test[0, 1] == 0


def test_creditcard_split_is_repeatable():
    x = torch.arange(40, dtype=torch.float32).reshape(10, 4)
    y = torch.tensor([0.0] * 6 + [1.0] * 4)

    first = split_train_test(x, y, seed=5)
    second = split_train_test(x, y, seed=5)

    for first_part, second_part in zip(first, second, strict=True):
        assert torch.equal(first_part, second_part)
