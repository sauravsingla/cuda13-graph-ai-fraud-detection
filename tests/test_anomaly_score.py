import torch

from examples.anomaly_score_gpu import robust_z_score, score_nodes


def test_robust_z_score_centers_the_median():
    features = torch.tensor(
        [
            [1.0, 10.0],
            [2.0, 10.0],
            [3.0, 10.0],
        ]
    )

    scores = robust_z_score(features)

    assert torch.equal(scores[1], torch.zeros(2))
    assert torch.equal(scores[:, 1], torch.zeros(3))


def test_score_nodes_ranks_large_outlier_first():
    features = torch.tensor(
        [
            [1.0, 1.0],
            [1.1, 0.9],
            [0.9, 1.1],
            [25.0, 30.0],
        ]
    )

    scores = score_nodes(features)

    assert int(torch.argmax(scores).item()) == 3
    assert scores[3] > scores[:3].max()
