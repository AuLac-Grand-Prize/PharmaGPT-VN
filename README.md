# PharmaGPT-VN — LLM tiếng Việt chuyên ngành dược

> **Engine 3 / 4 của nền tảng PharmLink AI.**
> RAG tiếng Việt chuyên ngành dược + 3-branch query understanding + Corrective RAG → trợ lý dược sĩ AI 24/7, có trích dẫn nguồn.

---

## 1. Vấn đề giải quyết

GPT-4 và các LLM nước ngoài:
- Không hiểu thuật ngữ y dược tiếng Việt (biệt dược VN, hướng dẫn của Bộ Y tế).
- Yêu cầu gửi dữ liệu hội thoại bệnh nhân ra cloud nước ngoài → vi phạm chủ quyền dữ liệu y tế.
- Chi phí cao, latency cao.

PharmaGPT-VN là hệ **RAG chuyên ngành dược tiếng Việt**: grounding câu trả lời trên corpus đã kiểm duyệt (dược điển VN, hướng dẫn BYT), validate bằng Corrective RAG để **từ chối khi thiếu bằng chứng**, redact PII và đính kèm disclaimer y khoa.

## 2. Công nghệ lõi

- **LLM**: gọi qua **API tương thích OpenAI** (`OPENAI_BASE_URL` + `LLM_MODEL_MAIN`). App không tự host model.
- **Hybrid retrieval**: **Qdrant** vector store + **BGE-M3** (dense + sparse) → RRF fusion.
- **3-branch query understanding** (chạy song song cho câu hỏi lâm sàng):
  - **MultiQueryRetriever** — `HeuristicVNRewriter` (cặp đồng nghĩa VN) + `LLMQueryRewriter` (n=3).
  - **HeuristicVNDecomposer** — tách câu hỏi đa thuốc / thuốc × bối cảnh lâm sàng thành sub-query.
  - **HyDEGenerator** — sinh "tài liệu giả định" để tăng recall (chỉ bật khi `is_clinical_query`).
- **Reranker**: **OpenRouterReranker** (Cohere `rerank-v3.5` mặc định, cấu hình qua `RERANKER_MODEL`); có `Reranker` Protocol để chat/disambiguation bind theo interface.
- **Corrective RAG (CRAG)**: `TieredGrader` (heuristic trước, LLM khi mơ hồ) phân loại sufficient/ambiguous/insufficient → từ chối khi không đủ bằng chứng.
- **Guardrails**: redact PII, citation enforcement cho câu trả lời clinical, medical disclaimer.
- **Caching/tracing**: Redis cache theo (query, filters); `Tracer` protocol cho observability.

## 3. KPIs mục tiêu

| Chỉ số | Mục tiêu |
|--------|----------|
| VN-PharmBench (nội bộ) — accuracy | Vượt GPT-4 |
| Hallucination rate (RAGAS faithfulness) | ≤ 3% |
| Latency p95 (end-to-end RAG + LLM API) | ≤ 2 giây |
| % câu trả lời có citation | 100% (clinical) |

## 4. Tech stack & phiên bản

| Lớp | Thành phần | Phiên bản |
|-----|-----------|-----------|
| Runtime | Python | ≥ 3.11 |
| API | FastAPI / Uvicorn / Pydantic | ≥ 0.115 / ≥ 0.32 / ≥ 2.9 |
| Embedding | sentence-transformers / FlagEmbedding (BGE-M3) | ≥ 3.2 / ≥ 1.3 |
| ML | PyTorch / Transformers | ≥ 2.4 / ≥ 4.45 |
| Vector DB | qdrant-client | ≥ 1.12 |
| HTTP | httpx (OpenRouter) | ≥ 0.27 |
| Cache | redis | ≥ 5.1 |
| Khác | tiktoken / structlog | ≥ 0.8 / ≥ 24.4 |
| Dev | pytest / pytest-asyncio / pytest-cov / ruff / mypy | ≥ 8.3 / — / ≥ 5.0 / ≥ 0.7 / ≥ 1.13 |

**Docker services**: Qdrant v1.12.0 (dense + sparse), Redis 7-alpine.

## 5. Cấu trúc thư mục

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

## 6. Pipeline xử lý (ChatService.complete)

1. **Refusal check** — phân loại OOD/harmful → từ chối nếu cần.
2. **PII redaction** — `redact_pii` loại SĐT (kể cả tiền tố +84), CCCD, BHYT.
3. **Retrieve** — nếu là câu hỏi lâm sàng: chạy song song 3 branch (rewrite + decompose + HyDE) → RRF (k=60) → top-50 chunks; nếu không: hybrid retrieve thường.
4. **Rerank** — OpenRouterReranker (top-10); fallback giữ thứ tự nếu API lỗi.
5. **CRAG grade** — `TieredGrader`; insufficient → từ chối + chuyển dược sĩ thật.
6. **Generate** — build context + prompt (role "dược sĩ lâm sàng") → LLM API.
7. **Validate** — citation, dosage sanity, drug name, tone.

## 7. API contract

### `POST /v1/chat/completions` (OpenAI-compatible)
Body:
```json
{
  "model": "pharmagpt-vn-8b-instruct",
  "messages": [
    {"role": "system", "content": "Bạn là trợ lý dược sĩ AI."},
    {"role": "user", "content": "Metformin có dùng được cho người suy thận eGFR 35 không?"}
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
        "content": "Theo hướng dẫn của Bộ Y tế (2023)... [1][2]",
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

Các endpoint khác: `POST /v1/embed` (BGE-M3), `POST /v1/disambiguate` (làm rõ câu hỏi mơ hồ), `GET /health`.

## 8. Khởi chạy

### Yêu cầu
- Python 3.11+
- Endpoint LLM tương thích OpenAI (URL + API key)
- API key OpenRouter (reranker)
- Qdrant + Redis (qua `docker compose`)

### Setup
```bash
cp .env.example .env
# Điền OPENAI_BASE_URL, OPENAI_API_KEY, LLM_MODEL_MAIN, OPENROUTER_API_KEY
make install
make services-up             # qdrant + redis
make ingest-demo             # corpus mẫu (không cần GPU) — hoặc:
make ingest-corpus           # nạp corpus y dược thật vào Qdrant
make dev                     # FastAPI tại http://localhost:8003
```

Ingest tùy biến:
```bash
python scripts/ingest_corpus.py --source corpus.jsonl \
  --qdrant-url http://localhost:6333 --collection vn_pharma_corpus \
  --embedder bge-m3 [--contextual]
```

### Test & evaluate
```bash
make test                    # pytest + coverage (ngưỡng 80%)
make lint                    # ruff + mypy
make eval                    # chạy VN-PharmBench
```

## 9. Cấu hình (.env)

| Biến | Mục đích | Mặc định |
|------|----------|----------|
| `APP_ENV` / `APP_PORT` / `LOG_LEVEL` | Runtime | development / 8003 / INFO |
| `OPENAI_BASE_URL` | Endpoint LLM (OpenAI-compatible) | — |
| `OPENAI_API_KEY` | API key LLM | — |
| `LLM_MODEL_MAIN` | Model sinh chính | — |
| `EMBEDDING_MODEL` / `EMBEDDER_DEVICE` | Model embedding / thiết bị | BAAI/bge-m3 / cpu |
| `OPENROUTER_API_KEY` / `OPENROUTER_BASE_URL` | Reranker | — / https://openrouter.ai/api/v1 |
| `RERANKER_MODEL` | Model rerank | cohere/rerank-v3.5 |
| `QDRANT_URL` / `QDRANT_API_KEY` / `QDRANT_COLLECTION` | Vector DB | http://localhost:6333 / — / vn_pharma_corpus |
| `REDIS_URL` | Cache / rate-limit | redis://localhost:6379/0 |
| `INTERNAL_API_TOKEN` | Secret xác thực API | — |
| `ENFORCE_CITATIONS_FOR_CLINICAL` | Bắt buộc citation cho clinical | true |
| `PII_REDACTION_ENABLED` | Bật redact PII | true |
| `MEDICAL_DISCLAIMER_ENABLED` | Đính kèm disclaimer | true |
| `DEFAULT_TEMPERATURE` / `DEFAULT_TOP_P` / `DEFAULT_MAX_TOKENS` | Tham số LLM | 0.2 / 0.9 / 512 |
| `QU_REWRITE_N` | Số biến thể rewrite (branch A) | 3 |
| `QU_DECOMPOSE_MAX` | Max sub-query (branch B) | 4 |
| `QU_HYDE_ENABLED` / `QU_HYDE_MAX_TOKENS` | HyDE (branch C) | true / 256 |
| `QU_PER_BRANCH_TOP_K` | Pool retrieval mỗi branch | 30 |
| `QU_RRF_K` | Tham số RRF fusion | 60 |

## 10. Guardrails y khoa

1. **PII redaction** — loại SĐT (gồm +84), CCCD (`\b\d{12}\b`), BHYT trước khi log.
2. **Scope check** — câu hỏi ngoài dược chuyển hướng về dược sĩ thật.
3. **Citation enforcement** — câu trả lời clinical bắt buộc ≥ 1 citation từ Qdrant.
4. **Disclaimer** — đính kèm "AI hỗ trợ tham khảo, không thay thế chẩn đoán dược sĩ".

## 11. VN-PharmBench

Bộ benchmark JSONL (`evaluation/benchmark.py`) theo các nhóm: `drug_info_basic`, `drug_info_advanced`, `dosage_adjustment`, `interactions`, `contraindications`, `otc_counseling`, `refusal`. Báo cáo: accuracy tổng, chất lượng citation, tỷ lệ refusal đúng, breakdown theo nhóm.

## 12. Docker

```bash
docker compose up -d           # qdrant + redis
docker build -t pharmagpt-vn . # build API image
```

## 13. Roadmap

- **v0.1** (MVP): RAG cơ bản qua LLM API, latency 3s.
- **v0.2**: Pipeline hoàn chỉnh (hybrid retrieve → 3-branch QU → rerank → CRAG → generate), latency 2s.
- **v1.0**: Tool-use (gọi VietDrug AI làm tool), multi-turn dialogue.
