# Public Dataset Guide

This repository does not commit public datasets directly. Download datasets from their official public pages and place them under `data/`.

## Option 1: ULB/Kaggle Credit Card Fraud Detection

Use this dataset when you want a quick tabular fraud-detection example.

Expected path:

```text
data/creditcard.csv
```

Expected columns:

```text
Time, V1, V2, ..., V28, Amount, Class
```

Run:

```bash
python examples/public_creditcard_fraud_gpu.py --csv data/creditcard.csv
```

Or:

```bash
make creditcard
```

Notes:

- The example trains a small PyTorch logistic classifier.
- It uses GPU automatically when CUDA is available.
- It uses positive-class weighting to handle fraud-class imbalance.
- It prints accuracy, precision, and recall.

## Option 2: Elliptic Bitcoin Transaction Graph Dataset

Use this dataset when you want a graph-style AML/fraud analytics example.

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
- This is a starting point for graph neural networks, cuGraph, or temporal graph methods.

## Why datasets are not committed

Fraud datasets can be large and may have separate licensing or usage terms. Keeping data outside the repo avoids accidental redistribution and keeps the repository lightweight.

The `.gitignore` file excludes `data/`, CSV files, model artifacts, logs, and other generated outputs.
