import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest
import torch

from benchmarks.model_quality_memory_benchmark import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    load_dataset,
    render_markdown_table,
    resolve_devices,
)
from examples.gpu_smoke_test import _matrix_size


def _write_tiny_creditcard_csv(csv_path: Path, rows: int = 20) -> None:
    records = []
    for index in range(rows):
        row = {column: float(index) for column in FEATURE_COLUMNS}
        row[TARGET_COLUMN] = 1.0 if index % 10 == 0 else 0.0
        records.append(row)
    pd.DataFrame(records).to_csv(csv_path, index=False)


def test_load_dataset_validates_expected_columns(tmp_path: Path):
    csv_path = tmp_path / "sample.csv"
    pd.DataFrame({"Time": [0], "Class": [0]}).to_csv(csv_path, index=False)

    with pytest.raises(ValueError):
        load_dataset(csv_path)


def test_load_dataset_accepts_creditcard_schema(tmp_path: Path):
    csv_path = tmp_path / "creditcard.csv"
    _write_tiny_creditcard_csv(csv_path, rows=2)

    x, y = load_dataset(csv_path)

    assert x.shape == (2, len(FEATURE_COLUMNS))
    assert y.shape == (2,)


def test_render_markdown_table_contains_runtime_columns():
    rendered = render_markdown_table(
        [
            {
                "benchmark_label": "unit",
                "device": "cpu",
                "model": "Compact Logistic",
                "accuracy": 1.0,
                "precision": 1.0,
                "recall": 1.0,
                "f1": 1.0,
                "parameters": 31,
                "model_size_mb": 0.0001,
                "train_runtime_seconds": 0.01,
                "inference_runtime_ms": 0.02,
                "examples_per_second": 1000.0,
                "peak_cuda_memory_mb": 0.0,
            }
        ]
    )

    assert "Train seconds" in rendered
    assert "Inference ms" in rendered
    assert "Examples/sec" in rendered
    assert "Peak CUDA memory MB" in rendered


def test_benchmark_script_runs_from_repo_root(tmp_path: Path):
    csv_path = tmp_path / "creditcard.csv"
    output_path = tmp_path / "benchmark.md"
    _write_tiny_creditcard_csv(csv_path, rows=20)

    result = subprocess.run(
        [
            sys.executable,
            "benchmarks/model_quality_memory_benchmark.py",
            "--csv",
            str(csv_path),
            "--device",
            "cpu",
            "--epochs",
            "1",
            "--output",
            str(output_path),
        ],
        check=False,
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert output_path.exists()
    assert "Examples/sec" in output_path.read_text(encoding="utf-8")


def test_resolve_devices_require_cuda_fails_without_cuda():
    if torch.cuda.is_available():
        pytest.skip("CUDA is available in this environment.")

    with pytest.raises(RuntimeError):
        resolve_devices("both", require_cuda=True)


def test_smoke_matrix_size_env(monkeypatch):
    monkeypatch.setenv("SMOKE_MATRIX_SIZE", "16")
    assert _matrix_size() == 16

    monkeypatch.setenv("SMOKE_MATRIX_SIZE", "0")
    with pytest.raises(ValueError):
        _matrix_size()
