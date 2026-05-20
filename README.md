# PharmaGPT-VN — LLM tiếng Việt chuyên ngành dược

> **Engine 3 / 4 của nền tảng PharmLink AI.**
> LLM tiếng Việt chuyên ngành dược + RAG → trợ lý dược sĩ AI 24/7, vận hành trên GPU tại Việt Nam.

---

## 1. Vấn đề giải quyết

GPT-4 và các LLM nước ngoài:
- Không hiểu thuật ngữ y dược tiếng Việt (biệt dược VN, hướng dẫn của Bộ Y tế).
- Yêu cầu gửi dữ liệu hội thoại bệnh nhân ra cloud nước ngoài → vi phạm chủ quyền dữ liệu y tế.
- Chi phí cao, latency cao.

PharmaGPT-VN là LLM **chuyên ngành dược tiếng Việt**, fine-tune từ mô hình mở, vận hành 100% on-premise tại Việt Nam.

## 2. Công nghệ lõi

- **LLM**: gọi qua **API tương thích OpenAI** (endpoint cấu hình qua `OPENAI_BASE_URL` + `LLM_MODEL_MAIN`). App không tự host model — toàn bộ trọng số nằm ở phía gateway/provider.
- **RAG pipeline** với **Qdrant** vector store + **bge-m3** Vietnamese embedding → augment câu trả lời từ database dược đã kiểm duyệt, tránh hallucination.
- **Reranker**: BGE-reranker-v2-m3 (cross-encoder) — chạy local (CPU/GPU tùy môi trường).
- **Guardrails**: Đầu ra clinical phải có citation; câu hỏi ngoài scope dược chuyển hướng về dược sĩ thật.

## 3. KPIs mục tiêu

| Chỉ số | Mục tiêu |
|--------|----------|
| VietPharmaBench (nội bộ) — accuracy | Vượt GPT-4 |
| Hallucination rate (RAGAS faithfulness) | ≤ 3% |
| Latency p95 (end-to-end RAG + LLM API) | ≤ 2 giây |
| % câu trả lời có citation | 100% (clinical) |

## 4. Cấu trúc thư mục

```
pharmagpt-vn/
├── src/pharmagpt_vn/
│   ├── api/              # FastAPI: /chat, /completions, /embed
│   │   └── routes/
│   ├── core/             # Config, logging, guardrails
│   ├── models/           # LLM client (OpenAI-compatible)
│   ├── services/         # Chat orchestrator
│   ├── rag/              # Retrieval pipeline
│   │   ├── retriever.py  # Qdrant + bge-m3
│   │   ├── reranker.py   # Cross-encoder
│   │   └── chunker.py
│   └── evaluation/       # VietPharmaBench harness
├── tests/
├── scripts/              # ingest_corpus, eval_bench, demo_backend
└── data/                 # (gitignored)
```

## 5. API contract

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
  "guardrails": {"medical_advice_disclaimer": true}
}
```

## 6. Khởi chạy

### Yêu cầu
- Python 3.11+
- Endpoint LLM tương thích OpenAI (URL + API key)
- Qdrant + Redis (qua `docker compose`)

### Setup
```bash
cp .env.example .env
# Điền OPENAI_BASE_URL, OPENAI_API_KEY, LLM_MODEL_MAIN vào .env
make install
make services-up             # qdrant + redis
make ingest-corpus           # nạp corpus y dược vào Qdrant
make dev                     # FastAPI tại http://localhost:8003
```

### Test & evaluate
```bash
make test
make eval                    # chạy VietPharmaBench
```

## 7. Guardrails y khoa

Mỗi response đi qua 4 lớp guardrail:
1. **PII redaction** — loại số CCCD, số thẻ BHYT, SĐT trước khi log.
2. **Scope check** — câu hỏi ngoài dược chuyển hướng về dược sĩ thật.
3. **Citation enforcement** — câu trả lời clinical bắt buộc có ít nhất 1 citation từ Qdrant.
4. **Disclaimer** — mỗi response đính kèm cảnh báo "AI hỗ trợ tham khảo, không thay thế chẩn đoán dược sĩ".

## 8. Roadmap

- **v0.1** (MVP): RAG cơ bản qua LLM API, latency 3s.
- **v0.2**: Pipeline hoàn chỉnh (hybrid retrieve → rerank → CRAG → generate), latency 2s.
- **v1.0**: Tool-use (gọi VietDrug AI làm tool), multi-turn dialogue.
