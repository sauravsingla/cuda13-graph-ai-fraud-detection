# Pull Request Description Template

## Summary

This PR adds CUDA 13 compatible GPU execution examples and validation notes.

## Changes

- Added CUDA 13 Docker environment
- Added GPU smoke test
- Added graph feature extraction example
- Added scoring example
- Added CPU vs GPU benchmark
- Added CI workflow for CPU validation
- Added Kubernetes GPU deployment reference

## Validation

```bash
nvidia-smi
nvcc --version
python examples/gpu_smoke_test.py
python benchmarks/cpu_vs_gpu_benchmark.py
```

## Notes

GPU validation should be performed locally or on a self-hosted GPU runner.
