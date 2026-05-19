"""Standalone demo backend for PharmaGPT-VN v0.2.

Wires the v0.2 pipeline (Hybrid retrieve → Rerank → CRAG → Generate) without
requiring Qdrant, GPU, or vLLM. Everything runs in-memory off `data/demo_corpus.json`.

Run:
    PYTHONPATH=src .venv/bin/python scripts/demo_backend.py
    # → http://localhost:8003/v1/chat/completions

Pieces that are real (production code paths exercised):
  - chat_service.ChatService orchestration (refusal → PII → retrieve → rerank → CRAG → generate)
  - guardrails (PII redaction, clinical detection, citation enforcement)
  - validators (citation coverage, dosage sanity, tone)
  - rag.crag.HeuristicGrader (with rerank-score thresholds)
  - rag.retriever.reciprocal_rank_fusion (real RRF k=60)

Pieces that are demo-only stubs (clearly marked):
  - InMemoryBM25Backend     — replaces Qdrant for retrieval
  - LexicalEmbedder         — replaces BGE-M3 (no model download)
  - PassThroughReranker     — uses BM25 score (no cross-encoder load)
  - TemplateLLMClient       — composes grounded answer from chunks (no vLLM)
"""

from __future__ import annotations

import json
import math
import re
import sys
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from pharmagpt_vn.api.routes import chat as chat_route  # noqa: E402
from pharmagpt_vn.api.routes import health  # noqa: E402
from pharmagpt_vn.core.refusal import Classification, RefusalClassifier  # noqa: E402
from pharmagpt_vn.models.llm_client import GenerationRequest, GenerationResult  # noqa: E402
from pharmagpt_vn.rag.crag import HeuristicGrader  # noqa: E402
from pharmagpt_vn.rag.embeddings import EmbeddingPair  # noqa: E402
from pharmagpt_vn.rag.reranker import CrossEncoderReranker, RerankedChunk  # noqa: E402
from pharmagpt_vn.rag.retriever import HybridRetriever, RetrievedChunk  # noqa: E402
from pharmagpt_vn.services.chat_service import ChatService  # noqa: E402

CORPUS_PATH = ROOT / "data" / "demo_corpus.json"
TOKEN_RE = re.compile(r"\w+", re.UNICODE)


# ---------------------------------------------------------------------------
# Corpus loading — flattens monographs into RetrievedChunk-ready records.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CorpusChunk:
    text: str
    source: str
    drug: str
    section: str

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "drug": self.drug,
            "section": self.section,
            "parent_path": (self.drug, self.section),
            "text": self.text,
            "source": self.source,
        }


def load_corpus(path: Path = CORPUS_PATH) -> list[CorpusChunk]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    out: list[CorpusChunk] = []
    for doc in raw["documents"]:
        for sec in doc["sections"]:
            out.append(
                CorpusChunk(
                    text=sec["text"],
                    source=doc["source"],
                    drug=doc["drug"],
                    section=sec["title"],
                )
            )
    return out


# ---------------------------------------------------------------------------
# Lexical embedder — produces sparse weights (BM25-style) and a deterministic
# pseudo-dense vector from term-frequency hashing. Avoids any model download.
# ---------------------------------------------------------------------------

class LexicalEmbedder:
    """Demo-only: stand-in for BGEM3Embedder. Same `encode_query` signature."""

    def __init__(self, dim: int = 64) -> None:
        self._dim = dim

    def encode_query(self, text: str) -> EmbeddingPair:
        tokens = TOKEN_RE.findall(text.lower())
        counts = Counter(tokens)
        dense = [0.0] * self._dim
        for tok, c in counts.items():
            dense[hash(tok) % self._dim] += float(c)
        norm = math.sqrt(sum(x * x for x in dense)) or 1.0
        dense = [x / norm for x in dense]
        sparse = {hash(tok) & 0xFFFFFFFF: float(c) for tok, c in counts.items()}
        return EmbeddingPair(dense=dense, sparse=sparse)


# ---------------------------------------------------------------------------
# In-memory BM25 backend — replaces Qdrant for the demo. Same Protocol as
# QdrantBackend so HybridRetriever doesn't notice.
# ---------------------------------------------------------------------------

class InMemoryBM25Backend:
    """Tiny BM25 index over the demo corpus. Production swaps to Qdrant."""

    K1 = 1.5
    B = 0.75

    def __init__(self, chunks: Sequence[CorpusChunk]) -> None:
        self._chunks = list(chunks)
        self._tokens: list[list[str]] = [TOKEN_RE.findall(c.text.lower()) for c in self._chunks]
        self._lengths = [len(t) for t in self._tokens]
        self._avgdl = sum(self._lengths) / max(len(self._lengths), 1)
        self._df: Counter[str] = Counter()
        for toks in self._tokens:
            for tok in set(toks):
                self._df[tok] += 1
        self._n_docs = len(self._chunks)

    def _bm25_score(self, query_tokens: list[str], doc_idx: int) -> float:
        score = 0.0
        doc_toks = self._tokens[doc_idx]
        doc_len = self._lengths[doc_idx] or 1
        tf = Counter(doc_toks)
        for q in query_tokens:
            df = self._df.get(q, 0)
            if df == 0:
                continue
            idf = math.log((self._n_docs - df + 0.5) / (df + 0.5) + 1.0)
            f = tf.get(q, 0)
            denom = f + self.K1 * (1 - self.B + self.B * doc_len / self._avgdl)
            score += idf * (f * (self.K1 + 1)) / (denom or 1.0)
        return score

    def _score_query(self, query: str, limit: int) -> list[RetrievedChunk]:
        q_tokens = TOKEN_RE.findall(query.lower())
        scored = []
        for i, chunk in enumerate(self._chunks):
            s = self._bm25_score(q_tokens, i)
            if s > 0:
                scored.append((s, chunk))
        scored.sort(key=lambda x: x[0], reverse=True)
        # Normalize to 0..1 so HeuristicGrader's thresholds (tuned for normalized
        # cross-encoder scores) behave sensibly in demo mode too.
        top_score = scored[0][0] if scored else 1.0
        return [
            RetrievedChunk(
                text=c.text,
                source=c.source,
                score=float(s / top_score),
                metadata=c.metadata,
            )
            for s, c in scored[:limit]
        ]

    async def dense_search(
        self,
        vector: list[float],
        limit: int,
        filters: dict[str, object] | None = None,
    ) -> list[RetrievedChunk]:
        # Demo: reuse BM25 — query string passed via `_last_query` set by retriever.
        return self._filter(self._score_query(self._last_query, limit), filters)

    async def sparse_search(
        self,
        weights: dict[int, float],
        limit: int,
        filters: dict[str, object] | None = None,
    ) -> list[RetrievedChunk]:
        return self._filter(self._score_query(self._last_query, limit), filters)

    @staticmethod
    def _filter(
        hits: list[RetrievedChunk], filters: dict[str, object] | None
    ) -> list[RetrievedChunk]:
        if not filters:
            return hits
        def ok(h: RetrievedChunk) -> bool:
            for k, v in filters.items():
                meta_val = h.metadata.get(k)
                if isinstance(v, (list, tuple, set)):
                    if meta_val not in v:
                        return False
                else:
                    if meta_val != v:
                        return False
            return True
        return [h for h in hits if ok(h)]

    _last_query: str = ""


class DemoHybridRetriever(HybridRetriever):
    """Thin wrapper that captures the raw query so the BM25 stub can use it."""

    def __init__(self, backend: InMemoryBM25Backend, embedder: LexicalEmbedder) -> None:
        super().__init__(embedder=embedder, backend=backend, rrf_k=60)
        self._mem = backend

    async def retrieve(self, query: str, top_k: int = 5) -> list[RetrievedChunk]:
        self._mem._last_query = query
        return await super().retrieve(query, top_k=top_k)


# ---------------------------------------------------------------------------
# Pass-through reranker — keeps BM25 order. Real cross-encoder is dropped to
# keep the demo CPU-friendly.
# ---------------------------------------------------------------------------

class PassThroughReranker(CrossEncoderReranker):
    """Demo: normalize candidate scores to 0..1 so HeuristicGrader thresholds
    (calibrated for cross-encoder outputs) behave as expected."""

    def __init__(self) -> None:
        pass

    def rerank(  # type: ignore[override]
        self, query: str, candidates, top_k: int = 10
    ) -> list[RerankedChunk]:
        cand = list(candidates)[:top_k]
        if not cand:
            return []
        top = max(c.score for c in cand) or 1.0
        return [RerankedChunk(chunk=c, rerank_score=c.score / top) for c in cand]


# ---------------------------------------------------------------------------
# Template LLM — composes a grounded answer using only retrieved chunk text.
# Guarantees [REF:n] citations so the validator passes.
# ---------------------------------------------------------------------------

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


class TemplateLLMClient:
    """Demo-only LLMClient. Stitches chunk sentences with per-sentence [REF:n]
    so the citation-coverage validator (min 85%) passes deterministically. No
    hallucination — every clause comes from a retrieved chunk."""

    def __init__(self) -> None:
        pass

    async def generate(self, req: GenerationRequest) -> GenerationResult:
        ref_blocks = _extract_chunks_from_prompt(req.prompt)
        if not ref_blocks:
            return GenerationResult(
                text="Chưa đủ căn cứ trong dược thư để trả lời câu hỏi này.",
                prompt_tokens=len(req.prompt) // 4,
                completion_tokens=20,
            )
        sentences: list[str] = []
        for ref_id, _source, body in ref_blocks[:3]:
            for sent in _SENT_SPLIT.split(body.strip()):
                sent = sent.strip()
                if not sent:
                    continue
                if not sent.endswith((".", "!", "?")):
                    sent += "."
                sentences.append(f"{sent} [REF:{ref_id}]")
        text = " ".join(sentences)
        return GenerationResult(
            text=text,
            prompt_tokens=len(req.prompt) // 4,
            completion_tokens=len(text) // 4,
        )


_REF_RE = re.compile(r"\[REF:(\d+)\]\s+([^\n]+)\n([^\n]+(?:\n(?!\[REF:)[^\n]+)*)")
_USER_RE = re.compile(r"<\|user\|>\n(.+?)\n<\|assistant\|>", re.DOTALL)


def _extract_chunks_from_prompt(prompt: str) -> list[tuple[int, str, str]]:
    out: list[tuple[int, str, str]] = []
    for m in _REF_RE.finditer(prompt):
        ref_id = int(m.group(1))
        source = m.group(2).strip()
        body = m.group(3).strip()
        out.append((ref_id, source, body))
    return out


def _extract_user_query(prompt: str) -> str:
    m = _USER_RE.search(prompt)
    return m.group(1).strip() if m else ""


# ---------------------------------------------------------------------------
# Demo classifier — defaults clinical_safe so RAG always runs.
# ---------------------------------------------------------------------------

class DemoClassifier(RefusalClassifier):
    def classify(self, query: str) -> Classification:
        return Classification(label="clinical_safe", confidence=0.9)


# ---------------------------------------------------------------------------
# App assembly
# ---------------------------------------------------------------------------

DEMO_KNOWN_TERMS = (
    # Drugs in corpus
    "Metformin", "Paracetamol", "Aspirin",
    # Co-mentioned drugs / drug classes the chunk text quotes verbatim
    "Insulin", "Warfarin", "Ibuprofen", "Naproxen", "NSAID", "ACEi",
    "N-acetylcysteine", "B12",
    # Medical acronyms
    "PCOS", "Reye", "INR",
    # Vietnamese sentence-starter words wrongly flagged as drug-like by the
    # validator's caps-only regex (a real classifier would handle this; for
    # demo we just whitelist the ones present in our corpus quotations).
    "Nghi", "Nguy", "Thận", "Liều", "Các", "Chống", "Không", "Người",
    "Hiếm", "Trẻ", "Với", "Hệ", "Theo", "Định",
)


def build_chat_service() -> ChatService:
    corpus = load_corpus()
    backend = InMemoryBM25Backend(corpus)
    embedder = LexicalEmbedder()
    retriever = DemoHybridRetriever(backend, embedder)
    reranker = PassThroughReranker()
    grader = HeuristicGrader(sufficient_threshold=0.55, ambiguous_threshold=0.25)
    known = list({*[c.drug for c in corpus], *DEMO_KNOWN_TERMS})
    return ChatService(
        retriever=retriever,
        reranker=reranker,
        refusal_classifier=DemoClassifier(),
        llm=TemplateLLMClient(),
        known_drugs=known,
        enforce_citations=True,
        crag_grader=grader,
    )


def build_app() -> FastAPI:
    app = FastAPI(title="PharmaGPT-VN v0.2 demo")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    service = build_chat_service()

    # Override the FastAPI dependency so the route uses our in-memory service.
    from pharmagpt_vn.api.dependencies import get_chat_service

    app.include_router(health.router, tags=["health"])
    app.include_router(chat_route.router, prefix="/v1", tags=["chat"])
    app.dependency_overrides[get_chat_service] = lambda: service
    return app


if __name__ == "__main__":
    uvicorn.run(build_app(), host="0.0.0.0", port=8003, log_level="info")
