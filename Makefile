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
	$(PYTHON) benchmarks/cpu_vs_gpu_benchmark.py

creditcard:
	$(PYTHON) examples/public_creditcard_fraud_gpu.py --csv data/creditcard.csv

elliptic:
	$(PYTHON) examples/elliptic_graph_loader.py --data-dir data/elliptic_bitcoin_dataset

docker-build:
	docker build -f Dockerfile.cuda13 -t cuda13-graph-ai-fraud-detection:latest .

docker-smoke:
	docker run --rm --gpus all cuda13-graph-ai-fraud-detection:latest
