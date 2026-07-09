"""Synthetic anomaly scoring example for transaction graph node features."""

import torch


def robust_z_score(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    median = x.median(dim=0).values
    mad = (x - median).abs().median(dim=0).values
    return (x - median) / (mad + eps)


def score_nodes(features: torch.Tensor) -> torch.Tensor:
    z = robust_z_score(features)
    score = z.abs().mean(dim=1)
    return score


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    features = torch.rand(100_000, 4, device=device)
    features[:50, 2:] *= 100  # inject synthetic high-value anomalies
    scores = score_nodes(features)
    top_scores, top_indices = torch.topk(scores, 10)

    print("Device:", device)
    print("Top suspicious node ids:", top_indices.detach().cpu().tolist())
    print("Top suspicious scores:", top_scores.detach().cpu().tolist())


if __name__ == "__main__":
    main()
