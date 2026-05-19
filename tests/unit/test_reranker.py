from __future__ import annotations

import logging

import pytest

from pharmagpt_vn.rag.reranker import CrossEncoderReranker
from pharmagpt_vn.rag.retriever import RetrievedChunk


def _chunk(source: str, text: str, score: float) -> RetrievedChunk:
    return RetrievedChunk(text=text, source=source, score=score, metadata={})


def test_rerank_fallback_logs_warning_on_failure(caplog: pytest.LogCaptureFixture) -> None:
    """Silent fallback was the bug — must warn loud when the cross-encoder dies."""
    reranker = CrossEncoderReranker()  # no model loaded → _load() raises ImportError
    candidates = [_chunk("a", "alpha", 0.9), _chunk("b", "beta", 0.4)]
    with caplog.at_level(logging.WARNING, logger="pharmagpt_vn.rag.reranker"):
        result = reranker.rerank("q", candidates, top_k=2)
    assert len(result) == 2
    # Fallback preserves the retrieval ordering by score.
    assert result[0].chunk.source == "a"
    assert any("reranker fallback" in rec.message for rec in caplog.records)


@pytest.mark.asyncio
async def test_arerank_runs_sync_rerank_in_thread() -> None:
    reranker = CrossEncoderReranker()
    candidates = [_chunk("a", "alpha", 0.7), _chunk("b", "beta", 0.3)]
    result = await reranker.arerank("q", candidates, top_k=2)
    assert [r.chunk.source for r in result] == ["a", "b"]


def test_rerank_handles_empty_candidates() -> None:
    assert CrossEncoderReranker().rerank("q", [], top_k=5) == []
