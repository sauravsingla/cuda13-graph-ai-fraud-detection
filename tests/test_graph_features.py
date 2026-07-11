import torch

from examples.graph_feature_gpu import aggregate_node_features


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
