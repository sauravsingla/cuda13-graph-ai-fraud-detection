# Benchmark Results

This file is a template for benchmark output from synthetic and public dataset experiments.

## Synthetic graph benchmark

Run:

```bash
python benchmarks/cpu_vs_gpu_benchmark.py
```

Suggested reporting format:

| Environment | CPU time | GPU time | Speedup | Notes |
|---|---:|---:|---:|---|
| Local GPU | TBD | TBD | TBD | CUDA 13.x |

## Public credit-card fraud example

Run:

```bash
python examples/public_creditcard_fraud_gpu.py --csv data/creditcard.csv
```

Suggested reporting format:

| Environment | Rows | Fraud labels | Accuracy | Precision | Recall | Notes |
|---|---:|---:|---:|---:|---:|---|
| Local GPU | TBD | TBD | TBD | TBD | TBD | Public credit-card dataset |

## Elliptic graph loader example

Run:

```bash
python examples/elliptic_graph_loader.py --data-dir data/elliptic_bitcoin_dataset
```

Suggested reporting format:

| Environment | Nodes | Edges | Label rows | GPU degree aggregation | Notes |
|---|---:|---:|---:|---:|---|
| Local GPU | TBD | TBD | TBD | TBD | Public Elliptic graph dataset |

Do not compare numbers across different GPUs or CPUs without reporting hardware details.
