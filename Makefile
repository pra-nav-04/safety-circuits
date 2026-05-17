.PHONY: help install dev build shell test lint format clean mvp

PY ?= python
IMG ?= safety-circuits:dev

help:
	@grep -E '^[a-zA-Z_-]+:.*?##' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?##"}{printf "  %-12s %s\n", $$1, $$2}'

install: ## pip install (editable) for runtime use
	$(PY) -m pip install -e .

dev: ## pip install (editable) + dev extras
	$(PY) -m pip install -e ".[dev]"

build: ## build CUDA docker image
	docker build -f docker/Dockerfile -t $(IMG) .

shell: ## drop into a shell inside the docker image (GPU passthrough)
	docker run --rm -it --gpus all \
		-v $(PWD):/workspace -w /workspace \
		-v $(HOME)/.cache/huggingface:/root/.cache/huggingface \
		$(IMG) bash

test: ## fast CPU smoke tests
	pytest -q

lint:
	ruff check src tests

format:
	ruff format src tests

mvp: ## run end-to-end MVP on TinyLlama (small batch)
	$(PY) -m safety_circuits.cli run-mvp --model tinyllama --n_pairs 8

clean:
	rm -rf build dist *.egg-info .pytest_cache .ruff_cache
	find . -type d -name __pycache__ -exec rm -rf {} +
