# Contributing

Thanks for taking the time to improve the project.

## Local setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e ".[dev]"
```

## Before opening a pull request

Run the same checks used by CI:

```bash
ruff check .
pytest
```

For changes that affect CUDA behavior, include the GPU model, driver version, CUDA runtime, PyTorch version, and the command used to reproduce the result. Do not describe a benchmark as measured unless the output came from an actual run.

## Pull request scope

Keep changes focused. A useful pull request should explain:

- the problem being addressed;
- the implementation choice;
- how the change was tested;
- any CPU, CUDA, dataset, or compatibility limitations.

Public datasets must not be committed to the repository. Add download instructions and cite the original source instead.

## Coding conventions

- Support Python 3.10 and later.
- Add type hints to new public functions.
- Keep examples runnable from the repository root.
- Prefer deterministic tests with explicit random seeds.
- Add or update tests when behavior changes.
