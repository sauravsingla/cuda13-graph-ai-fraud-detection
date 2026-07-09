# CUDA 13 Migration Notes

These notes help validate CUDA 13.x setup for AI and graph analytics workloads.

## Validation checklist

```bash
nvidia-smi
nvcc --version
docker run --rm --gpus all nvidia/cuda:13.3.0-devel-ubuntu24.04 nvidia-smi
```

## Common issues

### `nvidia-smi` works but `nvcc` is missing

The NVIDIA driver is installed, but the CUDA Toolkit may not be installed or is not in `PATH`.

### Docker cannot see GPU

Install and configure NVIDIA Container Toolkit, then retry:

```bash
docker run --rm --gpus all nvidia/cuda:13.3.0-devel-ubuntu24.04 nvidia-smi
```

### PyTorch CUDA version differs from system CUDA

PyTorch packages bundle their own CUDA runtime compatibility. Check:

```python
import torch
print(torch.version.cuda)
print(torch.cuda.is_available())
```

## Recommended contribution style

For upstream CUDA ecosystem contributions, start with focused issues or PRs:

- build validation notes
- Docker runtime validation
- compiler warning fixes
- sample documentation improvements
- small compatibility examples
