"""Contextual Retrieval enricher (Anthropic 9/2024).

Pattern: for each chunk, ask a cheap LLM to write 50-100 tokens of context that
situates the chunk inside its parent document. Prepend that context to the chunk
text before embedding (and before BM25 indexing). Per Anthropic's published
results this lifts top-20 retrieval recall by 35-67% when combined with hybrid +
rerank — the highest-ROI single intervention for production RAG.

Indexing-time only. Runtime latency is unaffected.

Design:
  - `ContextualEnricher` orchestrates: hash document → check cache → call LLM
    for each chunk that misses → return list of `ContextualChunk`.
  - LLM access is injected (Protocol). Tests use a deterministic stub; production
    wires a small local model (e.g. Qwen 7B via vLLM) — cost is one-time at
    indexing.
  - The cache is also injected — production uses Redis or an on-disk SQLite.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Protocol

from pharmagpt_vn.rag.chunker import Chunk

CONTEXT_PROMPT_TEMPLATE = (
    "Bạn là chuyên gia dược lâm sàng. Đọc tài liệu dưới đây và viết 1-2 câu "
    "(tối đa 100 từ) ngữ cảnh giúp đoạn trích đứng độc lập khi tìm kiếm. "
    "Nêu rõ: thuốc/chủ đề, mục/phần, loại tài liệu. KHÔNG diễn giải lại nội dung.\n\n"
    "<document>\n{document}\n</document>\n\n"
    "<chunk>\n{chunk}\n</chunk>\n\n"
    "Ngữ cảnh:"
)

CONTEXT_MAX_CHARS = 600  # ~100 Vietnamese tokens


@dataclass(frozen=True)
class ContextualChunk:
    """A chunk whose text is prefixed with LLM-generated context."""

    chunk: Chunk
    context: str

    @property
    def enriched_text(self) -> str:
        return f"{self.context.strip()}\n\n{self.chunk.text}"


class ContextLLM(Protocol):
    """Minimal LLM interface for context generation. Production wires vLLM."""

    def generate_context(self, document: str, chunk: str) -> str: ...


class ContextCache(Protocol):
    """Cache keyed by (document_hash, chunk_hash) → context string."""

    def get(self, key: str) -> str | None: ...
    def put(self, key: str, value: str) -> None: ...


class InMemoryContextCache:
    """Default in-memory cache. Production swaps to Redis/SQLite."""

    def __init__(self) -> None:
        self._store: dict[str, str] = {}

    def get(self, key: str) -> str | None:
        return self._store.get(key)

    def put(self, key: str, value: str) -> None:
        self._store[key] = value


class ContextualEnricher:
    def __init__(
        self,
        llm: ContextLLM,
        cache: ContextCache | None = None,
        max_chars: int = CONTEXT_MAX_CHARS,
    ) -> None:
        self._llm = llm
        self._cache = cache if cache is not None else InMemoryContextCache()
        self._max_chars = max_chars

    def enrich(self, document: str, chunks: Sequence[Chunk]) -> list[ContextualChunk]:
        """Generate (or retrieve from cache) context for each chunk."""
        doc_hash = _hash(document)
        out: list[ContextualChunk] = []
        for chunk in chunks:
            key = f"{doc_hash}:{_hash(chunk.text)}"
            cached = self._cache.get(key)
            if cached is not None:
                out.append(ContextualChunk(chunk=chunk, context=cached))
                continue
            raw = self._llm.generate_context(document, chunk.text)
            context = _truncate(raw, self._max_chars)
            self._cache.put(key, context)
            out.append(ContextualChunk(chunk=chunk, context=context))
        return out

    def enrich_chunks_only(
        self, document: str, chunks: Sequence[Chunk]
    ) -> list[Chunk]:
        """Return chunks with `text` replaced by enriched text — drop-in for embedders."""
        enriched = self.enrich(document, chunks)
        return [replace(ec.chunk, text=ec.enriched_text) for ec in enriched]


def build_context_prompt(document: str, chunk: str) -> str:
    """Public helper so callers (e.g. ingestion scripts) can reuse the prompt."""
    return CONTEXT_PROMPT_TEMPLATE.format(document=document, chunk=chunk)


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _truncate(text: str, max_chars: int) -> str:
    text = text.strip()
    if len(text) <= max_chars:
        return text
    cut = text[:max_chars]
    last_period = cut.rfind(".")
    if last_period > max_chars * 0.6:
        return cut[: last_period + 1]
    return cut
