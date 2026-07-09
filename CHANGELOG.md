# Changelog

## v0.4.0 - CUDA 12 vs CUDA 13 Benchmark Readiness

Added:

- `Dockerfile.cuda12` using `nvidia/cuda:12.6.3-devel-ubuntu24.04` for older CUDA GPU comparison
- Docker benchmark targets for CUDA 12 and CUDA 13
- `benchmark-all-docker` Makefile target
- `--output` option in `benchmarks/model_quality_memory_benchmark.py`
- `--seed` option in `benchmarks/model_quality_memory_benchmark.py`
- `docs/self-hosted-gpu-runner.md`
- CUDA 12 vs CUDA 13 comparison table in `docs/cuda-version-support.md`
- Dataset source and license checklist in `docs/public-datasets.md`

Changed:

- Benchmark outputs are now saved under `benchmarks/measured/` when Makefile targets are used
- README documents CUDA 12 and CUDA 13 Docker benchmark paths
- Benchmark result docs now describe measured-output files and the `--output` workflow

Notes:

- Actual CPU, CUDA 12, and CUDA 13 measured benchmark numbers are still not committed until real runs are performed on the dataset and target hardware.

## v0.3.0 - Model Quality and Memory Benchmarking

Added:

- Model quality and model-footprint benchmark
- Shared metrics utilities for accuracy, precision, recall, F1, parameter count, and model size
- Tests for model metrics and memory helpers
- README refocus on common CPU/CUDA benchmark metrics
- Dataset guide updates for model-quality benchmarking

Changed:

- `make benchmark` now runs the model-quality and model-footprint benchmark
- Public credit-card fraud example now reports F1, parameter count, and model size
- Benchmark result templates now focus on common model-quality and model-footprint metrics, not speed
- Architecture documentation now emphasizes consistent model evaluation

Removed:

- Speed-focused CPU vs GPU benchmark script
- CUDA-only memory metrics from the primary benchmark table

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
