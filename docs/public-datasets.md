# Public Dataset Guide

This repository does not commit public datasets directly. Download datasets from their official public pages and place them under `data/`.

## Dataset source and license checklist

Before using any dataset:

- Download it from the official dataset host.
- Review the dataset license and terms of use on the host page.
- Do not redistribute raw data through this repository unless the dataset license explicitly allows it.
- Record the dataset source, download date, and version in any benchmark report.

## Option 1: ULB/Kaggle Credit Card Fraud Detection

Use this dataset when you want a quick tabular fraud-detection example and a model-quality / model-footprint benchmark.

Known public reference details:

| Item | Detail |
|---|---|
| Common host | Kaggle Credit Card Fraud Detection dataset |
| Academic source family | ULB / Worldline credit-card fraud data used in public studies |
| Commonly reported size | 284,807 transactions |
| Commonly reported fraud labels | 492 fraud cases |
| Required local file | `data/creditcard.csv` |
| License guidance | Check the active Kaggle dataset page before use or redistribution |

Expected path:

```text
data/creditcard.csv
```

Expected columns:

```text
Time, V1, V2, ..., V28, Amount, Class
```

Run the compact single-model example:

```bash
python examples/public_creditcard_fraud_gpu.py --csv data/creditcard.csv
```

Or:

```bash
make creditcard
```

Run the model-quality and model-footprint benchmark:

```bash
python benchmarks/model_quality_memory_benchmark.py --csv data/creditcard.csv --device cpu --label cpu-baseline
```

Or:

```bash
make benchmark-cpu
```

Run CUDA comparison after building CUDA containers:

```bash
make docker-build-cuda12
make docker-build-cuda13
make docker-benchmark-cuda12
make docker-benchmark-cuda13
```

Notes:

- The benchmark compares compact and wider PyTorch models.
- It uses positive-class weighting to handle fraud-class imbalance.
- It reports accuracy, precision, recall, F1, parameter count, and model size.
- It intentionally does not report speed, throughput, peak CUDA memory, GPU utilization, or CUDA kernel timing.

## Option 2: Elliptic Bitcoin Transaction Graph Dataset

Use this dataset when you want a graph-style AML/fraud analytics example.

Known public reference details:

| Item | Detail |
|---|---|
| Common host | Elliptic Bitcoin transaction graph dataset public releases / Kaggle mirrors |
| Task type | AML / illicit transaction node classification |
| Required local directory | `data/elliptic_bitcoin_dataset/` |
| Required files | features, classes, edgelist CSV files |
| License guidance | Check the active dataset host before use or redistribution |

Expected directory:

```text
data/elliptic_bitcoin_dataset/
```

Expected files:

```text
elliptic_txs_features.csv
elliptic_txs_classes.csv
elliptic_txs_edgelist.csv
```

Run the basic PyTorch loader:

```bash
python examples/elliptic_graph_loader.py --data-dir data/elliptic_bitcoin_dataset
```

Or:

```bash
make elliptic
```

Run RAPIDS cuGraph feature extraction in a RAPIDS environment:

```bash
python examples/rapids_cugraph_elliptic.py --data-dir data/elliptic_bitcoin_dataset
```

Run the PyTorch Geometric GraphSAGE baseline in a PyG environment:

```bash
python examples/pyg_gnn_elliptic_baseline.py --data-dir data/elliptic_bitcoin_dataset
```

Notes:

- The loader builds a transaction-id mapping.
- It loads node features, labels, and directed transaction edges.
- It computes GPU-friendly in-degree and out-degree features.
- The cuGraph example computes PageRank and degree features on GPU.
- The PyG example trains a simple GraphSAGE node-classification baseline on known labels.
- Future benchmark results should report the same common metrics used by the main benchmark where applicable: accuracy, precision, recall, F1, parameters, and model size.

## Why datasets are not committed

Fraud datasets can be large and may have separate licensing or usage terms. Keeping data outside the repo avoids accidental redistribution and keeps the repository lightweight.

The `.gitignore` file excludes `data/`, CSV files, model artifacts, logs, and other generated outputs.
