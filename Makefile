PYTHON ?= python
RESULT_DIR ?= benchmarks/measured

.PHONY: install lint test smoke synthetic-graph score benchmark benchmark-cpu benchmark-gpu \
	benchmark-cuda-old benchmark-cuda-latest quality-memory creditcard elliptic \
	rapids-elliptic pyg-elliptic docker-build docker-build-cuda13 docker-build-cuda12 \
	docker-smoke docker-smoke-cuda13 docker-smoke-cuda12 docker-benchmark-cuda13 \
	docker-benchmark-cuda12 benchmark-all-docker

install:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -e ".[dev]"

lint:
	ruff check benchmarks/model_quality_memory_benchmark.py examples src tests

test:
	pytest

smoke:
	$(PYTHON) examples/gpu_smoke_test.py

synthetic-graph:
	$(PYTHON) examples/graph_feature_gpu.py

score:
	$(PYTHON) examples/anomaly_score_gpu.py

benchmark:
	mkdir -p $(RESULT_DIR)
	$(PYTHON) benchmarks/model_quality_memory_benchmark.py --csv data/creditcard.csv --device both --label local-run --output $(RESULT_DIR)/local-run.md

benchmark-cpu:
	mkdir -p $(RESULT_DIR)
	$(PYTHON) benchmarks/model_quality_memory_benchmark.py --csv data/creditcard.csv --device cpu --label cpu-baseline --output $(RESULT_DIR)/cpu-baseline.md

benchmark-gpu:
	mkdir -p $(RESULT_DIR)
	$(PYTHON) benchmarks/model_quality_memory_benchmark.py --csv data/creditcard.csv --device cuda --label cuda-gpu --output $(RESULT_DIR)/cuda-gpu.md

benchmark-cuda-old:
	mkdir -p $(RESULT_DIR)
	$(PYTHON) benchmarks/model_quality_memory_benchmark.py --csv data/creditcard.csv --device cuda --label cuda-12-old --output $(RESULT_DIR)/cuda-12-old.md

benchmark-cuda-latest:
	mkdir -p $(RESULT_DIR)
	$(PYTHON) benchmarks/model_quality_memory_benchmark.py --csv data/creditcard.csv --device cuda --label cuda-13-latest --output $(RESULT_DIR)/cuda-13-latest.md

quality-memory:
	mkdir -p $(RESULT_DIR)
	$(PYTHON) benchmarks/model_quality_memory_benchmark.py --csv data/creditcard.csv --device both --label local-run --output $(RESULT_DIR)/local-run.md

creditcard:
	$(PYTHON) examples/public_creditcard_fraud_gpu.py --csv data/creditcard.csv

elliptic:
	$(PYTHON) examples/elliptic_graph_loader.py --data-dir data/elliptic_bitcoin_dataset

rapids-elliptic:
	$(PYTHON) examples/rapids_cugraph_elliptic.py --data-dir data/elliptic_bitcoin_dataset

pyg-elliptic:
	$(PYTHON) examples/pyg_gnn_elliptic_baseline.py --data-dir data/elliptic_bitcoin_dataset

docker-build:
	docker build -f Dockerfile.cuda13 -t cuda13-graph-ai-fraud-detection:latest .

docker-build-cuda13:
	docker build -f Dockerfile.cuda13 -t cuda13-graph-ai-fraud-detection:cuda13 .

docker-build-cuda12:
	docker build -f Dockerfile.cuda12 -t cuda13-graph-ai-fraud-detection:cuda12 .

docker-smoke:
	docker run --rm --gpus all cuda13-graph-ai-fraud-detection:latest

docker-smoke-cuda13:
	docker run --rm --gpus all cuda13-graph-ai-fraud-detection:cuda13

docker-smoke-cuda12:
	docker run --rm --gpus all cuda13-graph-ai-fraud-detection:cuda12

docker-benchmark-cuda13:
	mkdir -p $(RESULT_DIR)
	docker run --rm --gpus all -v $(PWD)/data:/app/data -v $(PWD)/$(RESULT_DIR):/app/$(RESULT_DIR) cuda13-graph-ai-fraud-detection:cuda13 python3 benchmarks/model_quality_memory_benchmark.py --csv data/creditcard.csv --device cuda --label cuda-13-latest --output $(RESULT_DIR)/cuda-13-latest.md

docker-benchmark-cuda12:
	mkdir -p $(RESULT_DIR)
	docker run --rm --gpus all -v $(PWD)/data:/app/data -v $(PWD)/$(RESULT_DIR):/app/$(RESULT_DIR) cuda13-graph-ai-fraud-detection:cuda12 python3 benchmarks/model_quality_memory_benchmark.py --csv data/creditcard.csv --device cuda --label cuda-12-old --output $(RESULT_DIR)/cuda-12-old.md

benchmark-all-docker:
	$(MAKE) docker-build-cuda12
	$(MAKE) docker-build-cuda13
	$(MAKE) docker-benchmark-cuda12
	$(MAKE) docker-benchmark-cuda13
