# Changelog

## v0.3.0 - Model Quality and Memory Benchmarking

Added:

- Model quality and memory-footprint benchmark
- Shared metrics utilities for accuracy, precision, recall, F1, parameter count, model size, and peak CUDA memory
- Tests for model metrics and memory helpers
- README refocus on accuracy and memory footprint
- Dataset guide updates for model-quality benchmarking

Changed:

- `make benchmark` now runs the model quality and memory-footprint benchmark
- Public credit-card fraud example now reports F1, parameter count, model size, and peak CUDA memory
- Benchmark result templates now focus on quality and memory, not speed
- Architecture documentation now emphasizes memory-efficient model evaluation

Removed:

- Speed-focused CPU vs GPU benchmark script

## v0.2.0 - Public Dataset Support

Added:

- Public credit-card fraud dataset example
- Public Elliptic Bitcoin graph dataset loader
- Public dataset guide
- Benchmark results template for public datasets
- README badges and public dataset instructions
- Apache 2.0 license file

## v0.1.0 - Initial Release

Added:

- CUDA 13 compatible Dockerfile
- GPU smoke test using PyTorch
- Synthetic graph feature extraction example
- GPU-friendly scoring example
- CPU fallback test for CI
- GitHub Actions CI workflow
- Kubernetes GPU deployment reference
- CUDA 13 migration notes

Notes:

- Public datasets are not committed to the repository.
- GPU tests should be run locally or on self-hosted GPU runners.
