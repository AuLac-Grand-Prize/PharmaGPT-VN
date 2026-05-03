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

- **Base model**: Llama 3.1 8B / Qwen 2.5 7B / Vistral 7B / PhoGPT — chọn theo benchmark nội bộ.
- **Domain adaptation**: Tiếp tục pre-train trên **corpus 10 triệu token** y dược tiếng Việt (Dược điển VN, hướng dẫn điều trị Bộ Y tế, tờ rơi thuốc, tạp chí chuyên ngành).
- **Instruction tuning**: 50.000+ cặp Q&A do dược sĩ lâm sàng VN biên soạn.
- **RAG pipeline** với **Qdrant** vector store + **bge-m3** Vietnamese embedding → augment câu trả lời từ database dược đã kiểm duyệt, tránh hallucination.
- **Inference**: **vLLM** với continuous batching trên A100/H100 tại data center FPT/Viettel.
- **Guardrails**: Đầu ra clinical phải có citation; câu trả lời ngoài scope dược chuyển hướng về dược sĩ thật.

## 3. KPIs mục tiêu

| Chỉ số | Mục tiêu |
|--------|----------|
| VietPharmaBench (nội bộ) — accuracy | Vượt GPT-4 |
| Hallucination rate (RAGAS faithfulness) | ≤ 3% |
| Latency p95 (8B model, vLLM) | ≤ 2 giây |
| Chi phí / 1M token (so với GPT-4 API) | ≤ 1/10 |
| % câu trả lời có citation | 100% (clinical) |

## 4. Cấu trúc thư mục

```
pharmagpt-vn/
├── src/pharmagpt_vn/
│   ├── api/              # FastAPI: /chat, /completions, /embed
│   │   └── routes/
│   ├── core/             # Config, logging, guardrails
│   ├── models/           # Model loader (vLLM, transformers)
│   ├── services/         # Chat orchestrator
│   ├── rag/              # Retrieval pipeline
│   │   ├── retriever.py  # Qdrant + bge-m3
│   │   ├── reranker.py   # Cross-encoder
│   │   └── chunker.py
│   ├── training/         # SFT, DPO, continued pre-training
│   └── evaluation/       # VietPharmaBench harness
├── tests/
├── notebooks/
├── scripts/              # build_corpus, train_sft, eval_bench
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
- (Inference) GPU NVIDIA A100/H100 ≥ 40GB hoặc 2× RTX 4090
- (Training) 4× A100 80GB tối thiểu

### Setup
```bash
cp .env.example .env
make install
make download-base-model     # tải base model từ HF mirror nội bộ
make services-up             # qdrant + redis
make ingest-corpus           # nạp corpus y dược vào Qdrant
make dev                     # FastAPI tại http://localhost:8003
```

### Test & evaluate
```bash
make test
make eval                    # chạy VietPharmaBench
```

## 7. Training pipeline

```bash
# 1. Tiếp tục pre-train trên corpus dược tiếng Việt (10M token)
python scripts/continued_pretrain.py --corpus data/corpus_vn_pharma/ --epochs 1

# 2. SFT trên 50K instruction pairs do dược sĩ biên soạn
python scripts/train_sft.py --data data/sft_pairs.jsonl --epochs 3

# 3. DPO với preference data
python scripts/train_dpo.py --data data/dpo_pairs.jsonl

# 4. Eval
python scripts/eval_bench.py --benchmark vietpharma_bench_v1
```

## 8. Guardrails y khoa

Mỗi response đi qua 4 lớp guardrail:
1. **PII redaction** — loại số CCCD, số thẻ BHYT, SĐT trước khi log.
2. **Scope check** — câu hỏi ngoài dược chuyển hướng về dược sĩ thật.
3. **Citation enforcement** — câu trả lời clinical bắt buộc có ít nhất 1 citation từ Qdrant.
4. **Disclaimer** — mỗi response đính kèm cảnh báo "AI hỗ trợ tham khảo, không thay thế chẩn đoán dược sĩ".

## 9. Cam kết mở

Một phần checkpoint v1 sẽ được mở cho **cộng đồng nghiên cứu AI Việt Nam** dưới giấy phép research-only — đóng góp cho hệ sinh thái AI tiếng Việt:
- HuggingFace Hub: `pharmlink/pharmagpt-vn-8b-research`
- Corpus đã làm sạch một phần: `pharmlink/vn-pharma-corpus-mini`

## 10. Roadmap

- **v0.1** (MVP): 7B model, RAG cơ bản, latency 3s.
- **v0.2**: Domain-adapted, instruction-tuned, latency 2s.
- **v1.0**: DPO + tool-use (gọi VietDrug AI làm tool), multi-turn dialogue.
- **v2.0**: 70B variant cho enterprise hospitals, on-prem deploy.
