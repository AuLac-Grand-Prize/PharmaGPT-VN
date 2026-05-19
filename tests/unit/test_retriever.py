from __future__ import annotations

import pytest

from pharmagpt_vn.rag.embeddings import EmbeddingPair
from pharmagpt_vn.rag.retriever import (
    HybridRetriever,
    RetrievedChunk,
    reciprocal_rank_fusion,
)


def _chunk(source: str, text: str, score: float = 0.0) -> RetrievedChunk:
    return RetrievedChunk(text=text, source=source, score=score, metadata={})


def test_rrf_boosts_chunks_appearing_in_both_lists() -> None:
    a = _chunk("metformin", "first line for T2DM")
    b = _chunk("aspirin", "antiplatelet")
    fused = reciprocal_rank_fusion([a, b], [b, a], k=10)
    assert fused[0].source in {"metformin", "aspirin"}
    # Both appear in both lists, so they tie or differ by RRF; key check:
    # fused score must equal sum of two RRF contributions = 1/11 + 1/12 ≈ 0.174.
    expected = (1 / 11) + (1 / 12)
    assert abs(fused[0].score - expected) < 1e-9


def test_rrf_dedups_on_source_and_text_prefix() -> None:
    a = _chunk("src", "Hello world this is some text content")
    fused = reciprocal_rank_fusion([a], [a], k=60)
    assert len(fused) == 1


class _StubEmbedder:
    def encode_query(self, text: str) -> EmbeddingPair:
        return EmbeddingPair(dense=[0.1, 0.2], sparse={1: 0.5})


class _StubBackend:
    def __init__(self, dense: list[RetrievedChunk], sparse: list[RetrievedChunk]) -> None:
        self.dense = dense
        self.sparse = sparse
        self.dense_called_with: tuple | None = None

    async def dense_search(
        self,
        vector: list[float],
        limit: int,
        filters: dict[str, object] | None = None,
    ) -> list[RetrievedChunk]:
        self.dense_called_with = (tuple(vector), limit, filters)
        return self.dense

    async def sparse_search(
        self,
        weights: dict[int, float],
        limit: int,
        filters: dict[str, object] | None = None,
    ) -> list[RetrievedChunk]:
        self.sparse_filters = filters
        return self.sparse


@pytest.mark.asyncio
async def test_hybrid_retrieve_uses_both_backends_and_returns_top_k() -> None:
    backend = _StubBackend(
        dense=[_chunk("A", "alpha"), _chunk("B", "beta")],
        sparse=[_chunk("C", "gamma"), _chunk("A", "alpha")],
    )
    retriever = HybridRetriever(embedder=_StubEmbedder(), backend=backend)
    result = await retriever.retrieve("metformin", top_k=2)
    assert len(result) == 2
    assert result[0].source == "A"  # appears in both lists → highest fused score
    assert backend.dense_called_with is not None


@pytest.mark.asyncio
async def test_returns_empty_when_dependencies_missing() -> None:
    assert await HybridRetriever().retrieve("x", top_k=5) == []


@pytest.mark.asyncio
async def test_filters_are_passed_to_both_backends() -> None:
    backend = _StubBackend(dense=[], sparse=[])
    retriever = HybridRetriever(embedder=_StubEmbedder(), backend=backend)
    filters = {"drug": "Metformin", "section": ["Liều", "CCĐ"]}
    await retriever.retrieve("metformin", top_k=3, filters=filters)
    assert backend.dense_called_with is not None
    assert backend.dense_called_with[2] == filters
    assert backend.sparse_filters == filters  # type: ignore[attr-defined]
