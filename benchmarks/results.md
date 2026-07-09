# Benchmark Results

This file records benchmark output focused on CPU vs GPU model quality and memory footprint.

## Benchmark philosophy

This repository does **not** use raw speed as the primary benchmark. The goal is to compare fraud models using production-relevant quality and efficiency metrics on CPU and GPU:

- accuracy
- precision
- recall
- F1 score
- parameter count
- model size in MB
- CPU model memory footprint through model size
- peak CUDA memory in MB when CUDA is available

## Static model footprint

These values are derived directly from the model definitions and do not require the dataset to be downloaded.

| Model | Input features | Parameters | Model size MB | Relative size |
|---|---:|---:|---:|---:|
| Compact Logistic | 30 | 31 | 0.000118 | 1.00x |
| Wider MLP | 30 | 12,289 | 0.046879 | 396.42x |

Interpretation:

- The compact logistic model is about **396x smaller** than the wider MLP by parameter count and serialized parameter footprint.
- Accuracy, precision, recall, and F1 require the public credit-card dataset and should be filled after a real local run.
- Peak CUDA memory requires a CUDA GPU run and should not be fabricated.

## Public credit-card fraud CPU vs GPU benchmark

Run CPU and GPU when CUDA is available:

```bash
python benchmarks/model_quality_memory_benchmark.py --csv data/creditcard.csv --device both
```

Run CPU only:

```bash
python benchmarks/model_quality_memory_benchmark.py --csv data/creditcard.csv --device cpu
```

Run CUDA GPU only:

```bash
python benchmarks/model_quality_memory_benchmark.py --csv data/creditcard.csv --device cuda
```

Or:

```bash
make benchmark
```

Reporting template with known static values pre-filled:

| Environment | Device | Rows | Fraud labels | Model | Accuracy | Precision | Recall | F1 | Parameters | Model size MB | Peak memory MB | Notes |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---|
| Local CUDA 13 machine | CPU | run-required | run-required | Compact Logistic | run-required | run-required | run-required | run-required | 31 | 0.000118 | 0.00 | CPU baseline |
| Local CUDA 13 machine | CPU | run-required | run-required | Wider MLP | run-required | run-required | run-required | run-required | 12,289 | 0.046879 | 0.00 | CPU baseline |
| Local CUDA 13 machine | CUDA GPU | run-required | run-required | Compact Logistic | run-required | run-required | run-required | run-required | 31 | 0.000118 | run-required | CUDA memory measured |
| Local CUDA 13 machine | CUDA GPU | run-required | run-required | Wider MLP | run-required | run-required | run-required | run-required | 12,289 | 0.046879 | run-required | CUDA memory measured |

## Compact model target

A compact model is preferred when it preserves strong recall and F1 while reducing:

- parameters
- serialized model size
- peak CUDA memory
- deployment complexity

Suggested acceptance criteria:

| Metric | Target |
|---|---|
| Recall | Keep high for fraud detection use cases |
| F1 | Improve or remain close to wider model |
| Model size | Lower than wider model |
| Peak CUDA memory | Lower than wider model on GPU |
| Parameters | Lower than wider model |

## Public credit-card single-model example

Run:

```bash
python examples/public_creditcard_fraud_gpu.py --csv data/creditcard.csv
```

Reporting template with known static values pre-filled:

| Environment | Device | Rows | Fraud labels | Accuracy | Precision | Recall | F1 | Parameters | Model size MB | Peak CUDA memory MB | Notes |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Local CUDA 13 machine | CUDA GPU or CPU | run-required | run-required | run-required | run-required | run-required | run-required | 31 | 0.000118 | run-required | Compact logistic model |

## Elliptic graph loader example

Run:

```bash
python examples/elliptic_graph_loader.py --data-dir data/elliptic_bitcoin_dataset
```

Suggested reporting format:

| Environment | Device | Nodes | Edges | Label rows | Added graph features | Peak CUDA memory MB | Notes |
|---|---|---:|---:|---:|---|---:|---|
| Local CUDA 13 machine | CUDA GPU or CPU | run-required | run-required | run-required | in_degree, out_degree | run-required | Public Elliptic graph dataset |

## RAPIDS cuGraph Elliptic example

Run in a RAPIDS environment:

```bash
python examples/rapids_cugraph_elliptic.py --data-dir data/elliptic_bitcoin_dataset
```

Suggested reporting format:

| Environment | GPU | Nodes | Edges | Graph features | Peak GPU memory MB | Notes |
|---|---|---:|---:|---|---:|---|
| Local RAPIDS CUDA environment | run-required | run-required | run-required | PageRank, in-degree, out-degree | run-required | Public Elliptic graph dataset |

## PyTorch Geometric GNN baseline

Run in a PyTorch Geometric environment:

```bash
python examples/pyg_gnn_elliptic_baseline.py --data-dir data/elliptic_bitcoin_dataset
```

Suggested reporting format:

| Environment | GPU | Nodes | Edges | Known labels | Model | Accuracy | Parameters | Model size MB | Peak CUDA memory MB | Notes |
|---|---|---:|---:|---:|---|---:|---:|---:|---:|---|
| Local CUDA 13 PyG environment | run-required | run-required | run-required | run-required | GraphSAGE | run-required | run-required | run-required | run-required | Public Elliptic graph dataset |

## Hardware details to report

```text
GPU model:
NVIDIA driver version:
CUDA toolkit version:
PyTorch CUDA version:
CPU model:
RAM:
OS:
Dataset version/source:
Epochs:
Learning rate:
```

## Interpretation

Add measured accuracy, precision, recall, F1, dataset row counts, and peak CUDA memory only after running on a real dataset and, when relevant, a real CUDA GPU machine. Static model footprint values are already pre-filled because they are derived from the model code. Accuracy should be similar across CPU and GPU for the same model and training settings, while memory reporting differs: CPU uses model-size and parameter footprint, and CUDA additionally reports peak GPU memory. Do not compare model accuracy or memory footprint without reporting dataset version, hardware, model configuration, and training settings.
