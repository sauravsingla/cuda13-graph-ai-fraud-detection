# CUDA Version Support

This repository targets CUDA 13.x and is currently aligned with CUDA Toolkit 13.3.

## Current NVIDIA reference

NVIDIA's current CUDA release notes page is for CUDA Toolkit 13.3 Update 1.

Key CUDA 13.3 Update 1 component versions from NVIDIA release notes:

| Component | Version |
|---|---|
| CUDA Runtime | 13.3.29 |
| CUDA NVCC | 13.3.73 |
| CUDA Application Compiler | 13.3.73 |
| NVIDIA Linux Driver packaged with toolkit | 610.43.02 |

## Repository CUDA target

| Item | Repository value |
|---|---|
| CUDA target | CUDA 13.x |
| Current Docker base image | `nvidia/cuda:13.3.0-devel-ubuntu24.04` |
| Recommended operating system | Ubuntu 24.04 |
| Validation mode | CUDA availability, PyTorch CUDA detection, model accuracy benchmark, memory-footprint benchmark |

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

## Validation commands

Run these on a CUDA-enabled machine:

```bash
nvidia-smi
nvcc --version
python examples/gpu_smoke_test.py
python examples/public_creditcard_fraud_gpu.py --csv data/creditcard.csv
python benchmarks/model_quality_memory_benchmark.py --csv data/creditcard.csv
```

Expected validation checks:

- `nvidia-smi` shows a CUDA-capable NVIDIA GPU.
- `nvcc --version` shows CUDA 13.x.
- PyTorch reports CUDA availability.
- Fraud model benchmark reports accuracy, precision, recall, F1, parameters, model size, and peak CUDA memory.

## Notes

- This repository does not claim measured CUDA 13.3 Update 1 benchmark results until a real CUDA-enabled machine is used.
- Benchmark results should report GPU model, driver version, CUDA toolkit version, PyTorch CUDA version, CPU model, RAM, operating system, dataset version, and training settings.
- RAPIDS and PyTorch Geometric installation depends on CUDA, Python, and platform versions, so their installation is documented separately from `requirements.txt`.
