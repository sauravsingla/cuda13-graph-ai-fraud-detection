"""CPU vs GPU benchmark for synthetic graph feature aggregation."""

import time
import torch


def aggregate(num_nodes: int, num_edges: int, device: str):
    src = torch.randint(0, num_nodes, (num_edges,), device=device)
    dst = torch.randint(0, num_nodes, (num_edges,), device=device)
    amount = torch.rand(num_edges, device=device)

    out_amount = torch.zeros(num_nodes, device=device)
    in_amount = torch.zeros(num_nodes, device=device)
    out_amount.scatter_add_(0, src, amount)
    in_amount.scatter_add_(0, dst, amount)
    return out_amount, in_amount


def run_once(device: str, num_nodes: int = 100_000, num_edges: int = 1_000_000) -> float:
    start = time.perf_counter()
    aggregate(num_nodes, num_edges, device)
    if device == "cuda":
        torch.cuda.synchronize()
    return time.perf_counter() - start


def main() -> None:
    cpu_time = run_once("cpu")
    print(f"CPU time: {cpu_time:.4f}s")

    if torch.cuda.is_available():
        gpu_time = run_once("cuda")
        print(f"GPU time: {gpu_time:.4f}s")
        print(f"Speedup: {cpu_time / gpu_time:.2f}x")
    else:
        print("CUDA not available. GPU benchmark skipped.")


if __name__ == "__main__":
    main()
