# Benchmark Results

This file records benchmark output from synthetic and public dataset experiments.

## Synthetic graph benchmark

Run:

```bash
python benchmarks/cpu_vs_gpu_benchmark.py
```

### Current CPU baseline result

| Environment | Runtime | Torch build | CUDA available | Nodes | Edges | Mean time | Runs | Notes |
|---|---|---|---:|---:|---:|---:|---:|---|
| ChatGPT sandbox CPU baseline | CPU only | PyTorch 2.10.0+cpu | No | 100,000 | 1,000,000 | 0.3642s | 3 | CPU validation only; not a CUDA GPU benchmark |

Per-run timings:

| Run | Time |
|---:|---:|
| 1 | 0.3950s |
| 2 | 0.3953s |
| 3 | 0.3022s |

### Pending measured CUDA GPU result

| Environment | GPU | CUDA | CPU time | GPU time | Speedup | Notes |
|---|---|---|---:|---:|---:|---|
| Local CUDA 13 GPU | TBD | 13.x | TBD | TBD | TBD | Replace with measured result from a real NVIDIA GPU machine |

Run on a real CUDA-enabled machine:

```bash
nvidia-smi
nvcc --version
python benchmarks/cpu_vs_gpu_benchmark.py
```

Report these details with the measured result:

```text
GPU model:
NVIDIA driver version:
CUDA toolkit version:
PyTorch CUDA version:
CPU model:
RAM:
OS:
```

## Public credit-card fraud example

Run:

```bash
python examples/public_creditcard_fraud_gpu.py --csv data/creditcard.csv
```

Suggested reporting format:

| Environment | Rows | Fraud labels | Accuracy | Precision | Recall | Notes |
|---|---:|---:|---:|---:|---:|---|
| Local CUDA 13 GPU | TBD | TBD | TBD | TBD | TBD | Public credit-card dataset |

## Elliptic graph loader example

Run:

```bash
python examples/elliptic_graph_loader.py --data-dir data/elliptic_bitcoin_dataset
```

Suggested reporting format:

| Environment | Nodes | Edges | Label rows | GPU degree aggregation | Notes |
|---|---:|---:|---:|---:|---|
| Local CUDA 13 GPU | TBD | TBD | TBD | TBD | Public Elliptic graph dataset |

## RAPIDS cuGraph Elliptic example

Run in a RAPIDS environment:

```bash
python examples/rapids_cugraph_elliptic.py --data-dir data/elliptic_bitcoin_dataset
```

Suggested reporting format:

| Environment | GPU | Nodes | Edges | PageRank time | Degree time | Notes |
|---|---|---:|---:|---:|---:|---|
| Local RAPIDS CUDA environment | TBD | TBD | TBD | TBD | TBD | Public Elliptic graph dataset |

## PyTorch Geometric GNN baseline

Run in a PyTorch Geometric environment:

```bash
python examples/pyg_gnn_elliptic_baseline.py --data-dir data/elliptic_bitcoin_dataset
```

Suggested reporting format:

| Environment | GPU | Nodes | Edges | Known labels | Accuracy | Notes |
|---|---|---:|---:|---:|---:|---|
| Local CUDA 13 PyG environment | TBD | TBD | TBD | TBD | TBD | GraphSAGE baseline |

## Interpretation

The CPU numeric result is measured in a CPU-only environment. CUDA GPU, RAPIDS cuGraph, and PyTorch Geometric GNN results should be added only after running on a real GPU machine. Report GPU model, driver version, CUDA version, PyTorch CUDA version, and hardware details.

Do not compare numbers across different GPUs or CPUs without reporting hardware details.
