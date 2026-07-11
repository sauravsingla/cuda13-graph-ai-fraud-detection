"""Synthetic transaction graph feature extraction using torch."""

import torch


def generate_synthetic_edges(
    num_nodes: int = 100_000,
    num_edges: int = 1_000_000,
    device: str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if num_nodes <= 0:
        raise ValueError("num_nodes must be positive")
    if num_edges < 0:
        raise ValueError("num_edges cannot be negative")

    src = torch.randint(0, num_nodes, (num_edges,), device=device)
    dst = torch.randint(0, num_nodes, (num_edges,), device=device)
    amount = torch.rand(num_edges, device=device) * 10_000
    return src, dst, amount


def aggregate_node_features(
    num_nodes: int,
    src: torch.Tensor,
    dst: torch.Tensor,
    amount: torch.Tensor,
) -> torch.Tensor:
    if num_nodes <= 0:
        raise ValueError("num_nodes must be positive")
    if src.ndim != 1 or dst.ndim != 1 or amount.ndim != 1:
        raise ValueError("src, dst, and amount must be one-dimensional")
    if not (src.numel() == dst.numel() == amount.numel()):
        raise ValueError("src, dst, and amount must have the same length")
    if src.numel() and (src.min() < 0 or dst.min() < 0 or src.max() >= num_nodes or dst.max() >= num_nodes):
        raise ValueError("edge indices must be within the node range")

    out_degree = torch.zeros(num_nodes, device=amount.device)
    in_degree = torch.zeros(num_nodes, device=amount.device)
    out_amount = torch.zeros(num_nodes, device=amount.device)
    in_amount = torch.zeros(num_nodes, device=amount.device)

    out_degree.scatter_add_(0, src, torch.ones_like(amount))
    in_degree.scatter_add_(0, dst, torch.ones_like(amount))
    out_amount.scatter_add_(0, src, amount)
    in_amount.scatter_add_(0, dst, amount)

    return torch.stack([out_degree, in_degree, out_amount, in_amount], dim=1)


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_nodes = 100_000
    src, dst, amount = generate_synthetic_edges(num_nodes=num_nodes, device=device)
    features = aggregate_node_features(num_nodes, src, dst, amount)
    print("Device:", device)
    print("Feature matrix shape:", tuple(features.shape))
    print("Sample feature row:", features[0].detach().cpu().tolist())


if __name__ == "__main__":
    main()
