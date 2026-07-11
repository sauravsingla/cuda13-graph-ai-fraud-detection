# CUDA 13 Graph Fraud Detection Examples

![CUDA](https://img.shields.io/badge/CUDA-13.x-green)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![CI](https://img.shields.io/badge/CI-GitHub%20Actions-black)
![License](https://img.shields.io/badge/License-Apache--2.0-lightgrey)

Small, reproducible examples for graph-based fraud analysis across CPU and CUDA environments.

The repository covers synthetic transaction graphs, anomaly scoring, a credit-card fraud baseline, Elliptic dataset loaders, cuGraph and PyTorch Geometric examples, and a simple CPU/CUDA benchmark path. It is an experimentation repository, not a production fraud platform.

## What is included

- CUDA 12 and CUDA 13 Dockerfiles
- CPU-compatible tests and smoke checks
- synthetic graph feature aggregation and anomaly scoring
- credit-card fraud classification example
- Elliptic Bitcoin transaction graph loaders
- cuGraph and GraphSAGE examples
- benchmark output in Markdown format
- self-hosted GPU runner notes
- Kubernetes GPU deployment reference

## Environment

| Component | Version or requirement |
|---|---|
| Python | 3.10+ |
| CUDA target | 13.x |
| CUDA 13 image | `nvidia/cuda:13.3.0-devel-ubuntu24.04` |
| CUDA 12 comparison image | `nvidia/cuda:12.6.3-devel-ubuntu24.04` |
| OS | Ubuntu 24.04 recommended |
| GPU runtime | NVIDIA Container Toolkit |

Driver and compatibility notes are in [`docs/cuda-version-support.md`](docs/cuda-version-support.md).

## Local setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e ".[dev]"
```

Run the CPU checks:

```bash
ruff check benchmarks/model_quality_memory_benchmark.py src tests
pytest
python examples/gpu_smoke_test.py
```

## Docker

Build the CUDA 13 image:

```bash
docker build -f Dockerfile.cuda13 -t graph-fraud:cuda13 .
docker run --rm --gpus all graph-fraud:cuda13
```

Build the CUDA 12 comparison image:

```bash
docker build -f Dockerfile.cuda12 -t graph-fraud:cuda12 .
docker run --rm --gpus all graph-fraud:cuda12
```

## Examples

Synthetic graph features:

```bash
python examples/graph_feature_gpu.py
```

Synthetic anomaly scoring:

```bash
python examples/anomaly_score_gpu.py
```

Credit-card fraud example:

```bash
python examples/public_creditcard_fraud_gpu.py --csv data/creditcard.csv
```

Elliptic dataset loader:

```bash
python examples/elliptic_graph_loader.py --data-dir data/elliptic_bitcoin_dataset
```

cuGraph and PyTorch Geometric examples require their respective environments:

```bash
python examples/rapids_cugraph_elliptic.py --data-dir data/elliptic_bitcoin_dataset
python examples/pyg_gnn_elliptic_baseline.py --data-dir data/elliptic_bitcoin_dataset
```

Dataset download and licensing notes are in [`docs/public-datasets.md`](docs/public-datasets.md).

## Benchmark

The benchmark compares a compact logistic model with a wider MLP and reports metrics that can be reproduced across CPU and CUDA environments:

- accuracy
- precision
- recall
- F1 score
- parameter count
- model size

It does not claim runtime, throughput, GPU utilization, or kernel-level comparisons.

CPU run:

```bash
python benchmarks/model_quality_memory_benchmark.py \
  --csv data/creditcard.csv \
  --device cpu \
  --label cpu-baseline \
  --output benchmarks/measured/cpu-baseline.md
```

CUDA run:

```bash
python benchmarks/model_quality_memory_benchmark.py \
  --csv data/creditcard.csv \
  --device cuda \
  --label cuda-13-latest \
  --output benchmarks/measured/cuda-13-latest.md
```

Use the same dataset, seed, epochs, and learning rate when comparing environments. The benchmark uses a deterministic stratified holdout and fits normalization statistics on training rows only. See [`docs/benchmark-execution-matrix.md`](docs/benchmark-execution-matrix.md).

## Repository layout

```text
benchmarks/   benchmark script and result templates
docs/         setup, dataset, architecture, and compatibility notes
examples/     runnable CPU, CUDA, cuGraph, and PyG examples
k8s/          GPU deployment reference
src/          shared metric helpers
tests/        CPU-side regression tests
```

## Data and results

No payment data or public dataset files are committed. Synthetic examples are generated locally. Public datasets must be downloaded from their original sources and used under their own terms.

Benchmark numbers should be added only after an actual run in the stated environment, together with the GPU model, driver, CUDA runtime, PyTorch version, and exact command.

## License

Apache-2.0
