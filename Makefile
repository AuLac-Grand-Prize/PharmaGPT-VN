.PHONY: install dev test lint format services-up ingest-corpus ingest-demo eval

PYTHON ?= python3.11
VENV   ?= .venv
PORT   ?= 8003

install:
	$(PYTHON) -m venv $(VENV)
	$(VENV)/bin/pip install -U pip
	$(VENV)/bin/pip install -e ".[dev]"

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

ingest-corpus:
	$(VENV)/bin/python scripts/ingest_corpus.py --source data/corpus_vn_pharma/

ingest-demo:
	$(VENV)/bin/python scripts/ingest_corpus.py \
		--source data/demo_corpus.json \
		--embedder lexical \
		--qdrant-url :memory: \
		--collection pharmagpt_demo \
		--dry-run

eval:
	$(VENV)/bin/python scripts/eval_bench.py
