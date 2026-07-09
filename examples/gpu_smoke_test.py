"""Simple CUDA availability and GPU/CPU compute smoke test.

Environment variables:
  SMOKE_MATRIX_SIZE: matrix size for the smoke test. Defaults to 1024.
  REQUIRE_CUDA: set to 1 to fail when CUDA is not available.
"""

from __future__ import annotations

import os

import torch


def _matrix_size() -> int:
    value = int(os.getenv("SMOKE_MATRIX_SIZE", "1024"))
    if value <= 0:
        raise ValueError("SMOKE_MATRIX_SIZE must be a positive integer.")
    return value


def main() -> None:
    require_cuda = os.getenv("REQUIRE_CUDA", "0") == "1"
    cuda_available = torch.cuda.is_available()
    print("CUDA available:", cuda_available)

    size = _matrix_size()
    if cuda_available:
        print("CUDA device count:", torch.cuda.device_count())
        print("CUDA device name:", torch.cuda.get_device_name(0))
        print("PyTorch CUDA version:", torch.version.cuda)

        x = torch.randn(size, size, device="cuda")
        y = x @ x
        torch.cuda.synchronize()
        print("GPU test passed:", tuple(y.shape))
        return

    if require_cuda:
        raise RuntimeError("REQUIRE_CUDA=1 but torch.cuda.is_available() is False.")

    print("CUDA GPU not detected. Running CPU fallback.")
    x = torch.randn(size, size)
    y = x @ x
    print("CPU fallback passed:", tuple(y.shape))


if __name__ == "__main__":
    main()
