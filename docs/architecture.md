# Architecture

This project demonstrates a CUDA 13 compatible architecture for graph AI experimentation with a focus on model quality and memory footprint.

## Flow

```text
Public or synthetic dataset
        |
        v
Graph / tabular loader
        |
        +--> CUDA/PyTorch tensor features
        |
        +--> RAPIDS cuGraph graph features
        |
        +--> PyTorch Geometric GraphSAGE baseline
        |
        v
Fraud / anomaly model
        |
        v
Model quality and memory-footprint benchmark
        |
        v
Docker and Kubernetes deployment reference
```

## Benchmark focus

The benchmark layer focuses on:

- accuracy
- precision
- recall
- F1 score
- parameter count
- model size in MB
- peak CUDA memory in MB when CUDA is available

Raw speed is not the primary benchmark target for this repository.

## Design principles

- Use public datasets or clearly synthetic data.
- Do not commit datasets directly to the repository.
- Prefer compact models when they maintain strong recall and F1.
- Track model size and memory footprint alongside model quality.
- Support CPU fallback for CI.
- Use Docker for CUDA 13 runtime consistency.
- Keep GPU tests runnable locally or on self-hosted GPU runners.

## Production extension ideas

- Add model registry integration.
- Add model cards for benchmarked models.
- Add drift monitoring for score distributions.
- Add compact GNN memory comparisons.
- Add graph neural network inference.
- Add Kubernetes autoscaling patterns.
