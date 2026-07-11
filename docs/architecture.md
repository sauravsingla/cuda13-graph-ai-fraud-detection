# Architecture

This repository contains small CPU and CUDA examples for graph-based fraud experimentation. The components are intentionally independent so they can be run without treating the repository as a single deployment platform.

## Flow

```text
Public or synthetic dataset
        |
        v
Graph or tabular loader
        |
        +--> PyTorch tensor features
        |
        +--> RAPIDS cuGraph features
        |
        +--> PyTorch Geometric GraphSAGE baseline
        |
        v
Classification or anomaly score
        |
        v
Quality and model-footprint output
```

Dockerfiles provide CUDA 12 and CUDA 13 execution environments. The Kubernetes manifest is a minimal GPU scheduling reference for the smoke test; it is not a complete serving deployment.

## Benchmark scope

The cross-environment benchmark reports:

- accuracy
- precision
- recall
- F1 score
- parameter count
- model size in MB

Runtime, throughput, GPU utilization, and kernel timing are outside the current benchmark. The credit-card example separately reports peak allocated CUDA memory when it runs on a GPU.

## Design choices

- Use public datasets or clearly synthetic data.
- Keep datasets outside the repository.
- Fit tabular preprocessing on training rows only.
- Use deterministic splits for reproducible comparisons.
- Keep CPU tests available for shared logic.
- Record environment details with measured GPU results.

## Possible extensions

- time-ordered evaluation for fraud datasets
- compact GNN footprint comparisons
- model registry and model-card examples
- score-distribution monitoring
- GPU inference serving and autoscaling examples
