# PharmaGPT-VN — Vietnamese pharma-domain LLM assistant

> **Engine 3 / 4 of the PharmLink AI platform.**
> Vietnamese pharma-domain RAG + 3-branch query understanding + Corrective RAG → a 24/7 AI pharmacist assistant with cited sources.

---

## 1. Problem it solves

GPT-4 and other foreign LLMs:
- Don't understand Vietnamese medical/pharma terminology (local brand drugs, Ministry of Health guidelines).
- Require sending patient conversation data to overseas clouds → violating health-data sovereignty.
- Are expensive and high-latency.

PharmaGPT-VN is a **Vietnamese pharma-domain RAG system**: it grounds answers on a vetted corpus (VN pharmacopoeia, MoH guidelines), validates with Corrective RAG to **refuse when evidence is insufficient**, redacts PII, and appends a medical disclaimer.

## 2. Core technology

- **LLM**: called via an **OpenAI-compatible API** (`OPENAI_BASE_URL` + `LLM_MODEL_MAIN`). The app does not self-host the model.
- **Hybrid retrieval**: **Qdrant** vector store + **BGE-M3** (dense + sparse) → RRF fusion.
- **3-branch query understanding** (run in parallel for clinical questions):
  - **MultiQueryRetriever** — `HeuristicVNRewriter` (VN synonym pairs) + `LLMQueryRewriter` (n=3).
  - **HeuristicVNDecomposer** — splits multi-drug / drug × clinical-context questions into sub-queries.
  - **HyDEGenerator** — generates a "hypothetical document" to boost recall (only enabled when `is_clinical_query`).
- **Reranker**: **OpenRouterReranker** (Cohere `rerank-v3.5` by default, configurable via `RERANKER_MODEL`); a `Reranker` Protocol lets chat/disambiguation bind to the interface.
- **Corrective RAG (CRAG)**: `TieredGrader` (heuristic first, LLM when ambiguous) classifies sufficient/ambiguous/insufficient → refuses when evidence is insufficient.
- **Guardrails**: PII redaction, citation enforcement for clinical answers, medical disclaimer.
- **Caching/tracing**: Redis cache keyed by (query, filters); a `Tracer` protocol for observability.

## 3. Target KPIs

| Metric | Target |
|--------|--------|
| VN-PharmBench (internal) — accuracy | Beat GPT-4 |
| Hallucination rate (RAGAS faithfulness) | ≤ 3% |
| p95 latency (end-to-end RAG + LLM API) | ≤ 2 s |
| % of answers with a citation | 100% (clinical) |

## 4. Tech stack & versions

| Layer | Component | Version |
|-------|-----------|---------|
| Runtime | Python | ≥ 3.11 |
| API | FastAPI / Uvicorn / Pydantic | ≥ 0.115 / ≥ 0.32 / ≥ 2.9 |
| Embedding | sentence-transformers / FlagEmbedding (BGE-M3) | ≥ 3.2 / ≥ 1.3 |
| ML | PyTorch / Transformers | ≥ 2.4 / ≥ 4.45 |
| Vector DB | qdrant-client | ≥ 1.12 |
| HTTP | httpx (OpenRouter) | ≥ 0.27 |
| Cache | redis | ≥ 5.1 |
| Other | tiktoken / structlog | ≥ 0.8 / ≥ 24.4 |
| Dev | pytest / pytest-asyncio / pytest-cov / ruff / mypy | ≥ 8.3 / — / ≥ 5.0 / ≥ 0.7 / ≥ 1.13 |

**Docker services**: Qdrant v1.12.0 (dense + sparse), Redis 7-alpine.

## 5. Directory structure

```
pharmagpt-vn/
├── src/pharmagpt_vn/
│   ├── api/
│   │   ├── main.py            # FastAPI app + router
│   │   ├── dependencies.py    # DI (get_chat_service, ...)
│   │   └── routes/
│   │       ├── chat.py        # POST /v1/chat/completions
│   │       ├── embed.py       # POST /v1/embed
│   │       ├── disambiguate.py# POST /v1/disambiguate
│   │       └── health.py      # GET /health
│   ├── core/
│   │   ├── config.py          # Pydantic Settings (env)
│   │   ├── guardrails.py      # redact_pii, is_clinical_query, MEDICAL_DISCLAIMER
│   │   ├── refusal.py         # RefusalClassifier
│   │   ├── validators.py      # dosage/drug-name/citation/tone checks
│   │   └── tracing.py         # Tracer protocol
│   ├── models/
│   │   └── llm_client.py      # LLMClient (OpenAI-compatible)
│   ├── services/
│   │   ├── chat_service.py    # Orchestrator: refuse→redact→retrieve→rerank→CRAG→gen→validate
│   │   ├── disambiguation_service.py
│   │   └── prompt.py          # Prompt builders
│   ├── rag/
│   │   ├── retriever.py       # HybridRetriever (dense+sparse, RRF)
│   │   ├── query_rewrite.py   # MultiQueryRetriever, HeuristicVNRewriter
│   │   ├── query_decompose.py # HeuristicVNDecomposer
│   │   ├── hyde.py            # HyDEGenerator
│   │   ├── crag.py            # TieredGrader, LLMRelevanceGrader
│   │   ├── reranker.py        # Reranker protocol, CrossEncoderReranker
│   │   ├── openrouter_reranker.py # OpenRouterReranker (Cohere rerank-v3.5)
│   │   ├── embeddings.py      # BGE-M3 (dense + sparse)
│   │   ├── ingest.py          # Load → chunk → embed → upsert Qdrant
│   │   ├── chunker.py
│   │   ├── cache.py           # Redis cache
│   │   └── qdrant_store.py    # QdrantBackend (async)
│   └── evaluation/            # VN-PharmBench (benchmark, judges, ragas, metrics)
├── tests/{unit,integration,fixtures}
├── scripts/                   # ingest_corpus, eval_bench, demo_backend
├── docker-compose.yml         # qdrant + redis
├── Dockerfile
├── Makefile
├── pyproject.toml
└── .env.example
```

## 6. Processing pipeline (ChatService.complete)

1. **Refusal check** — classify OOD/harmful → refuse if needed.
2. **PII redaction** — `redact_pii` removes phone numbers (including +84 prefix), national ID (CCCD), insurance ID (BHYT).
3. **Retrieve** — for clinical questions: run the 3 branches in parallel (rewrite + decompose + HyDE) → RRF (k=60) → top-50 chunks; otherwise plain hybrid retrieval.
4. **Rerank** — OpenRouterReranker (top-10); falls back to retrieval order if the API fails.
5. **CRAG grade** — `TieredGrader`; insufficient → refuse + refer to a real pharmacist.
6. **Generate** — build context + prompt (role "clinical pharmacist") → LLM API.
7. **Validate** — citations, dosage sanity, drug names, tone.

## 7. API contract

### `POST /v1/chat/completions` (OpenAI-compatible)
Body:
```json
{
  "model": "pharmagpt-vn-8b-instruct",
  "messages": [
    {"role": "system", "content": "You are an AI pharmacist assistant."},
    {"role": "user", "content": "Can Metformin be used in a patient with eGFR 35?"}
  ],
  "temperature": 0.2,
  "max_tokens": 512,
  "rag": {"enabled": true, "top_k": 5}
}
```
Response:
```json
{
  "id": "chatcmpl-...",
  "model": "pharmagpt-vn-8b-instruct",
  "choices": [
    {
      "message": {
        "role": "assistant",
        "content": "According to MoH guidelines (2023)... [1][2]",
        "citations": [
          {"id": 1, "source": "HDDT-BYT-2023-0142", "page": 18},
          {"id": 2, "source": "duocdien-vn-v5", "monograph": "Metformin"}
        ]
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {"prompt_tokens": 142, "completion_tokens": 198},
  "guardrails": {"refused": false, "rag_used": true, "medical_advice_disclaimer": true}
}
```

Other endpoints: `POST /v1/embed` (BGE-M3), `POST /v1/disambiguate` (clarify ambiguous questions), `GET /health`.

## 8. Getting started

### Requirements
- Python 3.11+
- An OpenAI-compatible LLM endpoint (URL + API key)
- An OpenRouter API key (reranker)
- Qdrant + Redis (via `docker compose`)

### Setup
```bash
cp .env.example .env
# Fill in OPENAI_BASE_URL, OPENAI_API_KEY, LLM_MODEL_MAIN, OPENROUTER_API_KEY
make install
make services-up             # qdrant + redis
make ingest-demo             # sample corpus (no GPU) — or:
make ingest-corpus           # load the real pharma corpus into Qdrant
make dev                     # FastAPI at http://localhost:8003
```

Custom ingest:
```bash
python scripts/ingest_corpus.py --source corpus.jsonl \
  --qdrant-url http://localhost:6333 --collection vn_pharma_corpus \
  --embedder bge-m3 [--contextual]
```

### Test & evaluate
```bash
make test                    # pytest + coverage (80% threshold)
make lint                    # ruff + mypy
make eval                    # run VN-PharmBench
```

## 9. Configuration (.env)

| Variable | Purpose | Default |
|----------|---------|---------|
| `APP_ENV` / `APP_PORT` / `LOG_LEVEL` | Runtime | development / 8003 / INFO |
| `OPENAI_BASE_URL` | LLM endpoint (OpenAI-compatible) | — |
| `OPENAI_API_KEY` | LLM API key | — |
| `LLM_MODEL_MAIN` | Main generation model | — |
| `EMBEDDING_MODEL` / `EMBEDDER_DEVICE` | Embedding model / device | BAAI/bge-m3 / cpu |
| `OPENROUTER_API_KEY` / `OPENROUTER_BASE_URL` | Reranker | — / https://openrouter.ai/api/v1 |
| `RERANKER_MODEL` | Rerank model | cohere/rerank-v3.5 |
| `QDRANT_URL` / `QDRANT_API_KEY` / `QDRANT_COLLECTION` | Vector DB | http://localhost:6333 / — / vn_pharma_corpus |
| `REDIS_URL` | Cache / rate-limit | redis://localhost:6379/0 |
| `INTERNAL_API_TOKEN` | API auth secret | — |
| `ENFORCE_CITATIONS_FOR_CLINICAL` | Require citations for clinical answers | true |
| `PII_REDACTION_ENABLED` | Enable PII redaction | true |
| `MEDICAL_DISCLAIMER_ENABLED` | Append disclaimer | true |
| `DEFAULT_TEMPERATURE` / `DEFAULT_TOP_P` / `DEFAULT_MAX_TOKENS` | LLM params | 0.2 / 0.9 / 512 |
| `QU_REWRITE_N` | Number of rewrite variants (branch A) | 3 |
| `QU_DECOMPOSE_MAX` | Max sub-queries (branch B) | 4 |
| `QU_HYDE_ENABLED` / `QU_HYDE_MAX_TOKENS` | HyDE (branch C) | true / 256 |
| `QU_PER_BRANCH_TOP_K` | Retrieval pool per branch | 30 |
| `QU_RRF_K` | RRF fusion parameter | 60 |

## 10. Medical guardrails

1. **PII redaction** — strip phone numbers (incl. +84), national ID (`\b\d{12}\b`), insurance ID before logging.
2. **Scope check** — non-pharma questions are redirected to a real pharmacist.
3. **Citation enforcement** — clinical answers must include ≥ 1 citation from Qdrant.
4. **Disclaimer** — appends "AI is for reference only, not a substitute for a pharmacist's diagnosis".

## 11. VN-PharmBench

A JSONL benchmark (`evaluation/benchmark.py`) across categories: `drug_info_basic`, `drug_info_advanced`, `dosage_adjustment`, `interactions`, `contraindications`, `otc_counseling`, `refusal`. Reports: overall accuracy, citation quality, correct-refusal rate, and a per-category breakdown.

## 12. Docker

```bash
docker compose up -d           # qdrant + redis
docker build -t pharmagpt-vn . # build API image
```

## 13. Roadmap

- **v0.1** (MVP): basic RAG via LLM API, 3s latency.
- **v0.2**: full pipeline (hybrid retrieve → 3-branch QU → rerank → CRAG → generate), 2s latency.
- **v1.0**: tool use (call VietDrug AI as a tool), multi-turn dialogue.

---

## About PharmLink AI

This repository is **Engine 3 / 4** of [**PharmLink AI**](https://github.com/AuLac-Grand-Prize) — Vietnam's *Made-in-Vietnam* pharmaceutical AI platform serving 60,000+ pharmacies and up to 100 million citizens, in service of medication safety and national health-data sovereignty.

**The platform:**
- 💊 [VietDrug AI](https://github.com/AuLac-Grand-Prize/Pharma-VietDrugAI) — drug-interaction checks
- 📝 [PrescriptionVision](https://github.com/AuLac-Grand-Prize/Pharma-PrescriptionVision) — handwritten-prescription OCR
- 🤖 **PharmaGPT-VN** — Vietnamese pharma assistant *(this repo)*
- 📈 [DemandForecast AI](https://github.com/AuLac-Grand-Prize/Pharma-DemandForecast) — demand forecasting
- 🖥️ [Pharma Portal](https://github.com/AuLac-Grand-Prize/Pharma-Portal) — the pharmacist workspace

**Technology ownership:** the 10M+ token Vietnamese pharma corpus and the fine-tuned model weights are proprietary IP — deployable on-premise for institutions requiring maximum confidentiality; patient conversations never leave Vietnam.

> **Disclaimer:** PharmLink AI augments pharmacists — it does not replace them. Every clinical decision rests with a licensed pharmacist; outputs are validated by the Vietnamese Clinical Pharmacist Scientific Council.
