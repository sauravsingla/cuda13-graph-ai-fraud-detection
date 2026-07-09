# Self-Hosted GPU Runner Guide

GitHub-hosted runners usually do not provide NVIDIA GPUs. To run CUDA 12 and CUDA 13 benchmark jobs automatically, use a self-hosted runner on a CUDA-capable machine.

## Required machine

Recommended minimum setup:

| Component | Recommendation |
|---|---|
| GPU | NVIDIA CUDA-capable GPU |
| Driver | Compatible with CUDA 12.6.3 and CUDA 13.3 containers |
| OS | Ubuntu 24.04 or compatible Linux distribution |
| Docker | Installed |
| NVIDIA Container Toolkit | Installed and configured |
| GitHub runner | Self-hosted runner registered to this repository |

## Validate GPU runtime

Run:

```bash
nvidia-smi
docker run --rm --gpus all nvidia/cuda:13.3.0-devel-ubuntu24.04 nvidia-smi
docker run --rm --gpus all nvidia/cuda:12.6.3-devel-ubuntu24.04 nvidia-smi
```

Both Docker commands should detect the NVIDIA GPU.

## Runner labels

When registering the runner, add labels such as:

```text
self-hosted
linux
gpu
nvidia
cuda
```

These labels allow workflow jobs to target the GPU runner.

## Dataset placement

Place benchmark datasets on the runner machine before running GPU jobs:

```text
data/creditcard.csv
```

Do not commit raw datasets to the repository unless the dataset license explicitly allows redistribution.

## Manual benchmark workflow

From the repository root on the GPU runner:

```bash
make docker-build-cuda12
make docker-build-cuda13
make docker-benchmark-cuda12
make docker-benchmark-cuda13
```

The benchmark outputs are written to:

```text
benchmarks/measured/cuda-12-old.md
benchmarks/measured/cuda-13-latest.md
```

Run CPU baseline:

```bash
make benchmark-cpu
```

CPU output is written to:

```text
benchmarks/measured/cpu-baseline.md
```

## Suggested GitHub Actions workflow skeleton

Create a workflow such as `.github/workflows/gpu-benchmark.yml` only after a self-hosted GPU runner is available.

```yaml
name: GPU Benchmarks

on:
  workflow_dispatch:

jobs:
  cuda-benchmark:
    runs-on: [self-hosted, linux, gpu, nvidia, cuda]
    steps:
      - uses: actions/checkout@v4

      - name: Verify NVIDIA GPU
        run: nvidia-smi

      - name: Build CUDA 12 benchmark image
        run: make docker-build-cuda12

      - name: Build CUDA 13 benchmark image
        run: make docker-build-cuda13

      - name: Run CUDA 12 benchmark
        run: make docker-benchmark-cuda12

      - name: Run CUDA 13 benchmark
        run: make docker-benchmark-cuda13

      - name: Upload benchmark outputs
        uses: actions/upload-artifact@v4
        with:
          name: cuda-benchmark-results
          path: benchmarks/measured/*.md
```

## Reporting rule

Do not publish measured benchmark results unless:

- The dataset source and version are recorded.
- CPU/GPU hardware is recorded.
- NVIDIA driver version is recorded.
- CUDA toolkit/PyTorch CUDA version is recorded.
- The same model settings are used across CPU, CUDA 12, and CUDA 13 runs.
