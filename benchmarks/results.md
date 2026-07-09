# Benchmark Results

This file contains only benchmark numbers that are known and reproducible from the committed code.

No measured CPU, older CUDA GPU, or latest CUDA GPU quality results are published here until the benchmark has been run in those environments on the same dataset.

## Benchmark scope

The benchmark compares the same fraud-detection models across three execution environments:

1. CPU baseline
2. GPU with older CUDA environment
3. GPU with latest CUDA 13.3 / 13.3 Update 1 aligned environment

The benchmark includes only metrics that can be produced consistently across all three environments.

## Included benchmark metrics

| Metric | Included | Reason |
|---|---|---|
| Accuracy | Yes | Available on CPU, older CUDA GPU, and latest CUDA GPU |
| Precision | Yes | Available on CPU, older CUDA GPU, and latest CUDA GPU |
| Recall | Yes | Available on CPU, older CUDA GPU, and latest CUDA GPU |
| F1 | Yes | Available on CPU, older CUDA GPU, and latest CUDA GPU |
| Parameters | Yes | Static model footprint, same across all environments |
| Model size MB | Yes | Static model footprint, same across all environments |

## Excluded benchmark metrics

| Metric | Excluded | Reason |
|---|---|---|
| Runtime speed | Yes | Not the benchmark objective |
| Peak CUDA memory MB | Yes | CUDA-only metric; not available on CPU in the same form |
| GPU utilization | Yes | CUDA-only metric |
| CUDA kernel timing | Yes | CUDA-only metric |
| Throughput | Yes | Speed-oriented metric |

## Static model footprint benchmark

These values are derived directly from the model definitions in `benchmarks/model_quality_memory_benchmark.py` and do not require the dataset to be downloaded.

| Model | Input features | Parameters | Model size MB | Relative size |
|---|---:|---:|---:|---:|
| Compact Logistic | 30 | 31 | 0.000118 | 1.00x |
| Wider MLP | 30 | 12,289 | 0.046879 | 396.42x |

## Static benchmark interpretation

- The compact logistic model has **31 parameters**.
- The wider MLP has **12,289 parameters**.
- The wider MLP is about **396x larger** than the compact logistic model by parameter count and serialized parameter footprint.
- These numbers are valid for CPU, older CUDA GPU, and latest CUDA GPU runs because the model architecture is unchanged across environments.

## Metrics intentionally not filled here

The following metrics are intentionally not published in this file until the benchmark is run on the real dataset in the target environment:

| Metric | Why it is not pre-filled |
|---|---|
| Accuracy | Requires actual dataset run |
| Precision | Requires actual dataset run |
| Recall | Requires actual dataset run |
| F1 | Requires actual dataset run |
| Rows | Requires exact dataset file/version |
| Fraud labels | Requires exact dataset file/version |

## Required benchmark runs

Run CPU baseline and save output:

```bash
python benchmarks/model_quality_memory_benchmark.py \
  --csv data/creditcard.csv \
  --device cpu \
  --label cpu-baseline \
  --output benchmarks/measured/cpu-baseline.md
```

Run older CUDA GPU benchmark and save output:

```bash
python benchmarks/model_quality_memory_benchmark.py \
  --csv data/creditcard.csv \
  --device cuda \
  --label cuda-12-old \
  --output benchmarks/measured/cuda-12-old.md
```

Run latest CUDA GPU benchmark and save output:

```bash
python benchmarks/model_quality_memory_benchmark.py \
  --csv data/creditcard.csv \
  --device cuda \
  --label cuda-13-latest \
  --output benchmarks/measured/cuda-13-latest.md
```

Run Docker-based CUDA comparison:

```bash
make benchmark-all-docker
```

## Final result format after real runs

After the benchmark is executed in the three target environments, paste the measured output below this section using this shape:

| Label | Device | Model | Accuracy | Precision | Recall | F1 | Parameters | Model size MB |
|---|---|---|---:|---:|---:|---:|---:|---:|

Do not add rows to this table until actual measured output is available.

## Measured output files

The benchmark script supports `--output` and the Makefile writes benchmark artifacts under:

```text
benchmarks/measured/
```

Expected files after real runs:

```text
benchmarks/measured/cpu-baseline.md
benchmarks/measured/cuda-12-old.md
benchmarks/measured/cuda-13-latest.md
```

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

## Honesty rule

Static model footprint numbers can be published because they are derived directly from code. Accuracy, precision, recall, and F1 must not be published until they are produced by a real benchmark run on the stated dataset and hardware.
