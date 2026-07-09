# Benchmark Execution Matrix

This project reports measured benchmarks only after running the same dataset and model settings across three environments:

1. CPU baseline
2. GPU with an older CUDA environment
3. GPU with the latest CUDA 13.3 / 13.3 Update 1 aligned environment

The benchmark includes only metrics that can be produced consistently across all three environments:

- accuracy
- precision
- recall
- F1 score
- parameter count
- model size MB

It excludes speed, peak CUDA memory, GPU utilization, and CUDA kernel timings.

## Dataset requirement

Use the same dataset file for all runs:

```text
data/creditcard.csv
```

Expected columns:

```text
Time, V1, V2, ..., V28, Amount, Class
```

## Static model footprint values

These values are already known because they come directly from the model definitions:

| Model | Input features | Parameters | Model size MB | Relative size |
|---|---:|---:|---:|---:|
| Compact Logistic | 30 | 31 | 0.000118 | 1.00x |
| Wider MLP | 30 | 12,289 | 0.046879 | 396.42x |

These values apply to CPU, older CUDA GPU, and latest CUDA GPU runs because the model architecture is unchanged.

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

## Measured result table shape

After actual runs, paste the measured output from the benchmark script into `benchmarks/results.md` using this shape:

| Label | Device | Model | Accuracy | Precision | Recall | F1 | Parameters | Model size MB |
|---|---|---|---:|---:|---:|---:|---:|---:|

Do not add rows until actual measured output is available.

## Interpretation guidance

- Accuracy, precision, recall, and F1 should be broadly comparable across CPU and GPU for the same seed, dataset, model, and training settings.
- Parameter count and model size are static and should remain identical across CPU, older CUDA GPU, and latest CUDA GPU environments.
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

Do not publish accuracy, precision, recall, or F1 numbers until the benchmark has been run in the stated environment. Static model footprint numbers can be published because they are derived directly from the model definitions.
