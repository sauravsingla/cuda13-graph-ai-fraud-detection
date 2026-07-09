"""Simple CUDA availability and GPU compute smoke test."""

import torch


def main() -> None:
    print("CUDA available:", torch.cuda.is_available())

    if torch.cuda.is_available():
        print("CUDA device count:", torch.cuda.device_count())
        print("CUDA device name:", torch.cuda.get_device_name(0))
        print("PyTorch CUDA version:", torch.version.cuda)

        x = torch.randn(5000, 5000, device="cuda")
        y = x @ x
        print("GPU test passed:", tuple(y.shape))
    else:
        print("CUDA GPU not detected. Running CPU fallback.")
        x = torch.randn(1000, 1000)
        y = x @ x
        print("CPU fallback passed:", tuple(y.shape))


if __name__ == "__main__":
    main()
