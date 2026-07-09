# Benchmark Results

This file records benchmark output from synthetic and public dataset experiments.

## Synthetic graph benchmark

Run:

```bash
python benchmarks/cpu_vs_gpu_benchmark.py
```

### Current baseline result

| Environment | Runtime | Torch build | CUDA available | Nodes | Edges | Mean time | Runs | Notes |
|---|---|---|---:|---:|---:|---:|---:|---|
| ChatGPT sandbox CPU baseline | CPU only | PyTorch 2.10.0+cpu | No | 100,000 | 1,000,000 | 0.3642s | 3 | CPU validation only; not a CUDA GPU benchmark |

Per-run timings:

| Run | Time |
|---:|---:|
| 1 | 0.3950s |
| 2 | 0.3953s |
| 3 | 0.3022s |

### CUDA GPU result

| Environment | GPU | CUDA | CPU time | GPU time | Speedup | Notes |
|---|---|---|---:|---:|---:|---|
| Local CUDA 13 GPU | TBD | 13.x | TBD | TBD | TBD | Run on a CUDA 13 machine with NVIDIA GPU |

GPU numbers should be added only after running on a real CUDA-enabled machine:

```bash
nvidia-smi
nvcc --version
python benchmarks/cpu_vs_gpu_benchmark.py
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

## Interpretation

The current numeric result is a CPU-only baseline from a non-GPU environment. It is useful for validating that the synthetic benchmark runs correctly, but it should not be presented as CUDA acceleration evidence. For CUDA visibility, add GPU numbers from a real NVIDIA GPU machine and report GPU model, driver version, CUDA version, and PyTorch CUDA version.

Do not compare numbers across different GPUs or CPUs without reporting hardware details.
