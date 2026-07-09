# Benchmark Execution Matrix

This project reports benchmarks only after running the same dataset and model settings across three environments:

1. CPU baseline
2. GPU with an older CUDA environment
3. GPU with the latest CUDA 13.3 / 13.3 Update 1 aligned environment

The benchmark focus is **accuracy and memory footprint**, not raw runtime speed.

## Dataset requirement

Use the same dataset file for all runs:

```text
data/creditcard.csv
```

Expected columns:

```text
Time, V1, V2, ..., V28, Amount, Class
```

## Environment A: CPU baseline

Run on CPU only:

```bash
python benchmarks/model_quality_memory_benchmark.py \
  --csv data/creditcard.csv \
  --device cpu \
  --label cpu-baseline
```

This gives the reference model quality and static model footprint on CPU.

## Environment B: GPU with older CUDA

Recommended old-CUDA comparison target:

```text
CUDA 12.6.x GPU environment
```

Run in an older CUDA container or VM:

```bash
python benchmarks/model_quality_memory_benchmark.py \
  --csv data/creditcard.csv \
  --device cuda \
  --label cuda-12-old
```

Record:

```bash
nvidia-smi
nvcc --version
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.get_device_name(0))"
```

## Environment C: GPU with latest CUDA 13.3 alignment

Current latest CUDA documentation reference:

```text
CUDA Toolkit 13.3 Update 1
```

Repository Docker target:

```text
nvidia/cuda:13.3.0-devel-ubuntu24.04
```

Run:

```bash
python benchmarks/model_quality_memory_benchmark.py \
  --csv data/creditcard.csv \
  --device cuda \
  --label cuda-13-latest
```

Record:

```bash
nvidia-smi
nvcc --version
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.get_device_name(0))"
```

## Result table

Paste the output from the benchmark script into `benchmarks/results.md`.

Expected final comparison shape:

| Label | Device | Model | Accuracy | Precision | Recall | F1 | Parameters | Model size MB | Peak memory MB |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| cpu-baseline | cpu | Compact Logistic | run-required | run-required | run-required | run-required | 31 | 0.000118 | 0.00 |
| cpu-baseline | cpu | Wider MLP | run-required | run-required | run-required | run-required | 12,289 | 0.046879 | 0.00 |
| cuda-12-old | cuda | Compact Logistic | run-required | run-required | run-required | run-required | 31 | 0.000118 | run-required |
| cuda-12-old | cuda | Wider MLP | run-required | run-required | run-required | run-required | 12,289 | 0.046879 | run-required |
| cuda-13-latest | cuda | Compact Logistic | run-required | run-required | run-required | run-required | 31 | 0.000118 | run-required |
| cuda-13-latest | cuda | Wider MLP | run-required | run-required | run-required | run-required | 12,289 | 0.046879 | run-required |

## Interpretation guidance

- Accuracy, precision, recall, and F1 should be broadly comparable across CPU and GPU for the same seed, dataset, model, and training settings.
- Peak memory is expected to differ between CUDA 12 and CUDA 13 environments because framework, CUDA runtime, and driver behavior may differ.
- The compact logistic model has 31 parameters and a 0.000118 MB parameter footprint.
- The wider MLP has 12,289 parameters and a 0.046879 MB parameter footprint.
- The wider MLP is about 396x larger than the compact logistic model by parameter count and serialized parameter footprint.

## Required reporting metadata

Always report:

```text
Dataset source/version:
Rows:
Fraud labels:
Train/test split:
Epochs:
Learning rate:
CPU model:
RAM:
GPU model:
NVIDIA driver version:
CUDA toolkit version:
PyTorch version:
PyTorch CUDA version:
Operating system:
Docker image or VM image:
```

## Important honesty rule

Do not publish benchmark numbers until the benchmark has been run in the stated environment. Static model footprint numbers can be pre-filled because they are derived directly from the model definitions.
