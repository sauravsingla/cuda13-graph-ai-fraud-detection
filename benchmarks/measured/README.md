# Measured benchmark outputs

Save benchmark outputs in this directory so CPU, older CUDA, and latest CUDA-aligned runs can be compared from version control.

Recommended files:

```text
cpu-baseline.md
cuda-12-old.md
cuda-13-latest.md
comparison-summary.md
```

Generate them with:

```bash
python benchmarks/model_quality_memory_benchmark.py \
  --csv data/creditcard.csv \
  --device cpu \
  --label cpu-baseline \
  --output benchmarks/measured/cpu-baseline.md

python benchmarks/model_quality_memory_benchmark.py \
  --csv data/creditcard.csv \
  --device cuda \
  --require-cuda \
  --label cuda-13-latest \
  --output benchmarks/measured/cuda-13-latest.md
```

Do not commit private datasets. Commit only benchmark summaries that can be reproduced from public or synthetic datasets.
