"""Synthetic anomaly scoring for transaction graph node features."""

import torch


def robust_z_score(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    if x.ndim != 2:
        raise ValueError("x must be a two-dimensional feature matrix")
    if x.shape[0] == 0:
        raise ValueError("x must contain at least one row")
    if eps <= 0:
        raise ValueError("eps must be positive")

    median = x.median(dim=0).values
    mad = (x - median).abs().median(dim=0).values
    return (x - median) / (mad + eps)


def score_nodes(features: torch.Tensor) -> torch.Tensor:
    return robust_z_score(features).abs().mean(dim=1)


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
