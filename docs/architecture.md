# Architecture

This project demonstrates a lightweight CUDA 13 compatible architecture for GPU-accelerated graph AI experimentation.

## Flow

```text
Synthetic transaction graph
        |
        v
Graph feature extraction
        |
        v
GPU-friendly tensor aggregation
        |
        v
Anomaly scoring
        |
        v
Benchmark and deployment reference
```

## Design principles

- Use synthetic data only.
- Keep examples simple and reproducible.
- Support CPU fallback for CI.
- Use Docker for CUDA 13 runtime consistency.
- Keep GPU tests runnable locally or on self-hosted GPU runners.

## Production extension ideas

- Add model registry integration.
- Add streaming feature computation.
- Add graph neural network inference.
- Add monitoring for latency, drift, and score distribution.
- Add Kubernetes GPU autoscaling patterns.
