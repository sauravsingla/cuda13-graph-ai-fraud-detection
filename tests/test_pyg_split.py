import pytest
import torch

from examples.pyg_gnn_elliptic_baseline import split_labeled_nodes


def test_split_labeled_nodes_preserves_both_classes():
    y = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1, -1])

    train_mask, eval_mask = split_labeled_nodes(y, train_ratio=0.75, seed=7)

    assert set(y[train_mask].tolist()) == {0, 1}
    assert set(y[eval_mask].tolist()) == {0, 1}
    assert not torch.any(train_mask & eval_mask)
    assert not train_mask[-1]
    assert not eval_mask[-1]


def test_split_labeled_nodes_is_repeatable():
    y = torch.tensor([0, 0, 0, 1, 1, 1])

    first = split_labeled_nodes(y, seed=19)
    second = split_labeled_nodes(y, seed=19)

    assert torch.equal(first[0], second[0])
    assert torch.equal(first[1], second[1])


def test_split_labeled_nodes_rejects_small_class():
    y = torch.tensor([0, 0, 1, -1])

    with pytest.raises(ValueError, match="at least two labeled nodes"):
        split_labeled_nodes(y)
