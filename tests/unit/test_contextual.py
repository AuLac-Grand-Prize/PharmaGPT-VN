"""Tests for ContextualEnricher (Anthropic Contextual Retrieval pattern)."""

from __future__ import annotations

from pharmagpt_vn.rag.chunker import Chunk
from pharmagpt_vn.rag.contextual import (
    ContextualEnricher,
    InMemoryContextCache,
    build_context_prompt,
)


class _StubLLM:
    """Deterministic stub: records calls, returns canned context."""

    def __init__(self, response: str = "Ngữ cảnh stub.") -> None:
        self.response = response
        self.calls: list[tuple[str, str]] = []

    def generate_context(self, document: str, chunk: str) -> str:
        self.calls.append((document, chunk))
        return self.response


def _chunk(text: str, source: str = "doc1") -> Chunk:
    return Chunk(text=text, source=source, parent_path=("Drug", "Metformin"))


def test_enrich_prepends_context_to_chunk() -> None:
    llm = _StubLLM(response="Phần Liều dùng của Metformin trong Dược điển VN v5.")
    enricher = ContextualEnricher(llm=llm)

    chunks = [_chunk("Liều khởi đầu 500mg/ngày, tăng dần theo dung nạp.")]
    enriched = enricher.enrich(document="Full monograph...", chunks=chunks)

    assert len(enriched) == 1
    assert enriched[0].context.startswith("Phần Liều dùng")
    assert "Liều khởi đầu 500mg" in enriched[0].enriched_text
    assert enriched[0].enriched_text.startswith(enriched[0].context)


def test_enrich_uses_cache_to_skip_duplicate_llm_calls() -> None:
    llm = _StubLLM()
    enricher = ContextualEnricher(llm=llm, cache=InMemoryContextCache())

    chunks = [_chunk("Đoạn A."), _chunk("Đoạn A.")]  # duplicate text → same hash
    enricher.enrich(document="Doc X", chunks=chunks)
    enricher.enrich(document="Doc X", chunks=chunks)  # second pass — should hit cache

    # First pass: 1 unique chunk → 1 call. Second pass: 0 calls. Total = 1.
    assert len(llm.calls) == 1


def test_enrich_chunks_only_returns_drop_in_replacement() -> None:
    llm = _StubLLM(response="Ngữ cảnh.")
    enricher = ContextualEnricher(llm=llm)

    original = _chunk("Nội dung gốc.")
    out = enricher.enrich_chunks_only("doc", [original])

    assert len(out) == 1
    assert isinstance(out[0], Chunk)
    assert out[0].text.startswith("Ngữ cảnh.")
    assert "Nội dung gốc." in out[0].text
    # Other fields preserved
    assert out[0].source == original.source
    assert out[0].parent_path == original.parent_path


def test_truncation_keeps_context_within_max_chars() -> None:
    long_response = "Câu một. " + ("Câu dài. " * 200)
    llm = _StubLLM(response=long_response)
    enricher = ContextualEnricher(llm=llm, max_chars=80)

    enriched = enricher.enrich("doc", [_chunk("x")])
    assert len(enriched[0].context) <= 80


def test_build_context_prompt_includes_both_document_and_chunk() -> None:
    prompt = build_context_prompt("DOC_BODY", "CHUNK_BODY")
    assert "DOC_BODY" in prompt
    assert "CHUNK_BODY" in prompt
    assert "Ngữ cảnh:" in prompt
