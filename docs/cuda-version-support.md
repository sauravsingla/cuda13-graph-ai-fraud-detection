# CUDA Version Support

This repository targets CUDA 13.x and includes an older CUDA 12 comparison environment for benchmark reproducibility.

## Current NVIDIA reference

NVIDIA's current CUDA release notes page is for CUDA Toolkit 13.3 Update 1.

Key CUDA 13.3 Update 1 component versions from NVIDIA release notes:

| Component | Version |
|---|---|
| CUDA Runtime | 13.3.29 |
| CUDA NVCC | 13.3.73 |
| CUDA Application Compiler | 13.3.73 |
| NVIDIA Linux Driver packaged with toolkit | 610.43.02 |

## Repository CUDA environments

| Benchmark role | CUDA line | Dockerfile | Docker base image | Label used in benchmark |
|---|---|---|---|---|
| Older CUDA GPU baseline | CUDA 12.6.3 | `Dockerfile.cuda12` | `nvidia/cuda:12.6.3-devel-ubuntu24.04` | `cuda-12-old` |
| Latest CUDA GPU target | CUDA 13.3 / 13.3 Update 1 aligned | `Dockerfile.cuda13` | `nvidia/cuda:13.3.0-devel-ubuntu24.04` | `cuda-13-latest` |

## Repository CUDA target

| Item | Repository value |
|---|---|
| Main CUDA target | CUDA 13.x |
| Older CUDA comparison | CUDA 12.6.3 |
| Latest CUDA Docker base image | `nvidia/cuda:13.3.0-devel-ubuntu24.04` |
| Older CUDA Docker base image | `nvidia/cuda:12.6.3-devel-ubuntu24.04` |
| Recommended operating system | Ubuntu 24.04 |
| Validation mode | CUDA availability, PyTorch CUDA detection, model-quality benchmark, model-footprint benchmark |

## Driver guidance

NVIDIA lists the minimum driver range for CUDA 13.x minor-version compatibility as:

```text
CUDA 13.x: >= 580
```

For CUDA Toolkit 13.3 Update 1, NVIDIA lists the packaged Linux driver version as:

```text
>= 610.43.02
```

Recommended interpretation for this repository:

| Scenario | Driver guidance |
|---|---|
| CUDA 13.x compatibility | Driver >= 580 |
| CUDA 13.3 / 13.3 Update 1 alignment | Driver >= 610.43.02 preferred |
| CUDA 12.6.3 comparison | Use a driver compatible with the selected CUDA 12.6.3 container runtime |

## Validation commands

Run these on a CUDA-enabled machine:

```bash
nvidia-smi
nvcc --version
python examples/gpu_smoke_test.py
python examples/public_creditcard_fraud_gpu.py --csv data/creditcard.csv
python benchmarks/model_quality_memory_benchmark.py --csv data/creditcard.csv --device cuda --label cuda-13-latest
```

Run Docker-based old-vs-latest CUDA validation:

```bash
make docker-build-cuda12
make docker-build-cuda13
make docker-benchmark-cuda12
make docker-benchmark-cuda13
```

Expected validation checks:

- `nvidia-smi` shows a CUDA-capable NVIDIA GPU.
- `nvcc --version` reports the CUDA toolkit version inside the target environment.
- PyTorch reports CUDA availability.
- Fraud model benchmark reports accuracy, precision, recall, F1, parameters, and model size.

## Notes

- This repository does not claim measured CUDA 12 or CUDA 13 benchmark results until a real CUDA-enabled machine is used.
- Benchmark results should report GPU model, driver version, CUDA toolkit version, PyTorch CUDA version, CPU model, RAM, operating system, dataset version, and training settings.
- RAPIDS and PyTorch Geometric installation depends on CUDA, Python, and platform versions, so their installation is documented separately from `requirements.txt`.
