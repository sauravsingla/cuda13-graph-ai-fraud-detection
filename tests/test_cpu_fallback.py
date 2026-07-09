import torch


def test_cpu_tensor_operation():
    x = torch.ones(10, 10)
    y = x @ x
    assert y.shape == (10, 10)
    assert torch.all(y == 10)
