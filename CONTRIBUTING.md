# Contributing

Thank you for your interest in contributing.

## Good first contributions

- Improve CUDA 13 setup notes
- Add benchmark results from different GPU environments
- Improve Docker documentation
- Add tests for CPU fallback paths
- Add small graph analytics examples

## Development

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pytest tests/
```

## Pull request checklist

- Keep examples synthetic and reproducible
- Include clear validation steps
- Avoid committing large generated files
- Update README or docs when behavior changes
