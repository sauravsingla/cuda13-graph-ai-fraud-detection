import pytest
import torch

from examples.graph_feature_gpu import aggregate_node_features, generate_synthetic_edges


def test_aggregate_node_features_matches_manual_counts_and_amounts():
    src = torch.tensor([0, 0, 1, 2])
    dst = torch.tensor([1, 2, 2, 0])
    amount = torch.tensor([10.0, 20.0, 5.0, 7.0])

    features = aggregate_node_features(3, src, dst, amount)

    expected = torch.tensor(
        [
            [2.0, 1.0, 30.0, 7.0],
            [1.0, 1.0, 5.0, 10.0],
            [1.0, 2.0, 7.0, 25.0],
        ]
    )
    assert torch.equal(features, expected)


def test_aggregate_node_features_keeps_isolated_nodes_zeroed():
    src = torch.tensor([0])
    dst = torch.tensor([1])
    amount = torch.tensor([3.5])

    features = aggregate_node_features(4, src, dst, amount)

    assert torch.equal(features[2], torch.zeros(4))
    assert torch.equal(features[3], torch.zeros(4))


def test_aggregate_node_features_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match="same length"):
        aggregate_node_features(
            3,
            torch.tensor([0, 1]),
            torch.tensor([1]),
            torch.tensor([5.0, 7.0]),
        )


def test_aggregate_node_features_rejects_out_of_range_nodes():
    with pytest.raises(ValueError, match="node range"):
        aggregate_node_features(
            2,
            torch.tensor([0]),
            torch.tensor([2]),
            torch.tensor([5.0]),
        )


def test_generate_synthetic_edges_allows_empty_edge_set():
    src, dst, amount = generate_synthetic_edges(num_nodes=4, num_edges=0)

    assert src.numel() == 0
    assert dst.numel() == 0
    assert amount.numel() == 0
