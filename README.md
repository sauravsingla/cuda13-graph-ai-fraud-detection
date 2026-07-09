# CUDA 13 Graph AI Fraud Detection

CUDA 13 compatible GPU acceleration examples for graph-based fraud detection, anomaly scoring, and large-scale financial risk analytics.

## Why this project matters

Fraud and mule-risk detection systems often need to process large transaction graphs quickly. This project demonstrates how CUDA 13.x can be used to accelerate graph feature extraction, anomaly scoring, benchmarking, and production-oriented AI workflows.

## Features

- CUDA 13 Docker environment
- GPU smoke test
- CPU vs GPU benchmark
- Synthetic transaction graph generator
- GPU-accelerated anomaly scoring example
- CI workflow for CPU validation
- Kubernetes GPU deployment reference
- MLOps-friendly project structure

## Tested stack

| Component | Version |
|---|---|
| CUDA Toolkit | 13.x |
| NVIDIA CUDA Docker image | `nvidia/cuda:13.3.0-devel-ubuntu24.04` |
| NVIDIA Driver | Use CUDA 13 compatible driver |
| Python | 3.10+ |
| OS | Ubuntu 24.04 recommended |
| Docker | NVIDIA Container Toolkit required for GPU runtime |

## Quick start

### 1. Verify GPU and CUDA

```bash
nvidia-smi
nvcc --version
```

### 2. Build Docker image

```bash
docker build -f Dockerfile.cuda13 -t cuda13-graph-ai-fraud-detection:latest .
```

### 3. Run GPU smoke test

```bash
docker run --rm --gpus all cuda13-graph-ai-fraud-detection:latest
```

Expected output:

```text
CUDA available: True
GPU test passed
```

## Run locally without Docker

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
python examples/gpu_smoke_test.py
python examples/graph_feature_gpu.py
python benchmarks/cpu_vs_gpu_benchmark.py
```

## Project structure

```text
cuda13-graph-ai-fraud-detection/
  README.md
  Dockerfile.cuda13
  requirements.txt
  examples/
    gpu_smoke_test.py
    graph_feature_gpu.py
    anomaly_score_gpu.py
  benchmarks/
    cpu_vs_gpu_benchmark.py
    results.md
  docs/
    architecture.md
    cuda13-migration-notes.md
  k8s/
    gpu-deployment.yaml
  tests/
    test_cpu_fallback.py
  .github/workflows/
    ci.yml
```

## Use case

This repository uses a synthetic transaction graph to demonstrate GPU-friendly patterns for:

- graph feature extraction
- suspicious node scoring
- transaction anomaly scoring
- mule-risk style network analytics
- CPU vs GPU benchmarking

The data is synthetic and does not contain real payment data.

## Contribution roadmap

- [ ] Add RAPIDS cuGraph implementation
- [ ] Add PyTorch Geometric GNN baseline
- [ ] Add CUDA kernel example for edge aggregation
- [ ] Add self-hosted GitHub Actions GPU runner guide
- [ ] Add Kubernetes autoscaling pattern for GPU inference
- [ ] Add model monitoring dashboard

## Suggested GitHub topics

`cuda`, `cuda-13`, `gpu-computing`, `graph-ai`, `fraud-detection`, `anomaly-detection`, `financial-crime`, `mlops`, `pytorch`, `docker`, `kubernetes`, `responsible-ai`

## License

Apache-2.0
