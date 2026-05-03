.PHONY: install dev test lint format services-up download-base-model ingest-corpus train-sft train-dpo eval

PYTHON ?= python3.11
VENV   ?= .venv
PORT   ?= 8003

install:
	$(PYTHON) -m venv $(VENV)
	$(VENV)/bin/pip install -U pip
	$(VENV)/bin/pip install -e ".[dev,training]"

dev:
	$(VENV)/bin/uvicorn pharmagpt_vn.api.main:app --reload --host 0.0.0.0 --port $(PORT)

test:
	$(VENV)/bin/pytest

lint:
	$(VENV)/bin/ruff check src tests
	$(VENV)/bin/mypy src

format:
	$(VENV)/bin/ruff format src tests
	$(VENV)/bin/ruff check --fix src tests

services-up:
	docker compose up -d qdrant redis

download-base-model:
	$(VENV)/bin/python scripts/download_base_model.py

ingest-corpus:
	$(VENV)/bin/python scripts/ingest_corpus.py --source data/corpus_vn_pharma/

train-sft:
	$(VENV)/bin/python scripts/train_sft.py

train-dpo:
	$(VENV)/bin/python scripts/train_dpo.py

eval:
	$(VENV)/bin/python scripts/eval_bench.py
