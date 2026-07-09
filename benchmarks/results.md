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

### Sample CUDA GPU result

Important: the GPU row below is a clearly marked sample result format for documentation and should be replaced with a measured result from your own CUDA 13 machine.

| Environment | GPU | CUDA | CPU time | GPU time | Speedup | Notes |
|---|---|---|---:|---:|---:|---|
| Sample CUDA 13 GPU benchmark | NVIDIA GPU, replace with exact model | 13.x | 0.3642s | 0.0450s | 8.09x | Illustrative sample only; not measured in this environment |

Replace the sample row after running on a real CUDA-enabled machine:

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

## Interpretation

The CPU numeric result is measured in a CPU-only environment. The sample GPU row is not measured here and should be replaced with a real CUDA 13 GPU run before using it as performance evidence. For CUDA visibility, report GPU model, driver version, CUDA version, PyTorch CUDA version, and hardware details.

Do not compare numbers across different GPUs or CPUs without reporting hardware details.
