PYTHON ?= python

install:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt

test:
	pytest tests/

smoke:
	$(PYTHON) examples/gpu_smoke_test.py

synthetic-graph:
	$(PYTHON) examples/graph_feature_gpu.py

score:
	$(PYTHON) examples/anomaly_score_gpu.py

benchmark:
	$(PYTHON) benchmarks/model_quality_memory_benchmark.py --csv data/creditcard.csv --device both --label local-run

benchmark-cpu:
	$(PYTHON) benchmarks/model_quality_memory_benchmark.py --csv data/creditcard.csv --device cpu --label cpu-baseline

benchmark-gpu:
	$(PYTHON) benchmarks/model_quality_memory_benchmark.py --csv data/creditcard.csv --device cuda --label cuda-gpu

benchmark-cuda-old:
	$(PYTHON) benchmarks/model_quality_memory_benchmark.py --csv data/creditcard.csv --device cuda --label cuda-12-old

benchmark-cuda-latest:
	$(PYTHON) benchmarks/model_quality_memory_benchmark.py --csv data/creditcard.csv --device cuda --label cuda-13-latest

quality-memory:
	$(PYTHON) benchmarks/model_quality_memory_benchmark.py --csv data/creditcard.csv --device both --label local-run

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
	docker run --rm --gpus all -v $(PWD)/data:/app/data cuda13-graph-ai-fraud-detection:cuda13 python3 benchmarks/model_quality_memory_benchmark.py --csv data/creditcard.csv --device cuda --label cuda-13-latest

docker-benchmark-cuda12:
	docker run --rm --gpus all -v $(PWD)/data:/app/data cuda13-graph-ai-fraud-detection:cuda12 python3 benchmarks/model_quality_memory_benchmark.py --csv data/creditcard.csv --device cuda --label cuda-12-old
