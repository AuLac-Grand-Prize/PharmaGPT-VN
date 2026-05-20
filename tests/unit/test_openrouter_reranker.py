"""OpenRouterReranker — parse, sort, fallback paths."""

from __future__ import annotations

import httpx
import pytest

from pharmagpt_vn.rag.openrouter_reranker import OpenRouterReranker
from pharmagpt_vn.rag.retriever import RetrievedChunk


def _chunks() -> list[RetrievedChunk]:
    return [
        RetrievedChunk(text="metformin liều cao", source="s1", score=0.7, metadata={}),
        RetrievedChunk(text="paracetamol giảm đau", source="s2", score=0.6, metadata={}),
        RetrievedChunk(text="metformin suy thận eGFR<30", source="s3", score=0.5, metadata={}),
    ]


def _mock_transport(handler):
    return httpx.MockTransport(handler)


@pytest.mark.asyncio
async def test_arerank_parses_results_and_sorts_descending() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path.endswith("/rerank")
        assert request.headers["Authorization"] == "Bearer test-key"
        body = request.read().decode()
        assert "metformin liều cao" in body
        return httpx.Response(
            200,
            json={
                "id": "or-1",
                "model": "cohere/rerank-v3.5",
                "results": [
                    {"index": 2, "relevance_score": 0.95, "document": {"text": "..."}},
                    {"index": 0, "relevance_score": 0.30, "document": {"text": "..."}},
                    {"index": 1, "relevance_score": 0.10, "document": {"text": "..."}},
                ],
                "usage": {"cost": 0.0, "search_units": 1, "total_tokens": 100},
            },
        )

    client = httpx.AsyncClient(transport=_mock_transport(handler))
    reranker = OpenRouterReranker(api_key="test-key", async_client=client)
    out = await reranker.arerank("metformin suy thận", _chunks(), top_k=2)
    await client.aclose()

    assert [r.chunk.source for r in out] == ["s3", "s1"]
    assert out[0].rerank_score == 0.95
    assert out[1].rerank_score == 0.30


@pytest.mark.asyncio
async def test_arerank_fallback_on_http_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"error": "boom"})

    client = httpx.AsyncClient(transport=_mock_transport(handler))
    reranker = OpenRouterReranker(api_key="k", async_client=client)
    out = await reranker.arerank("q", _chunks(), top_k=2)
    await client.aclose()

    # Fallback preserves retrieval order with original scores.
    assert [r.chunk.source for r in out] == ["s1", "s2"]
    assert out[0].rerank_score == 0.7


@pytest.mark.asyncio
async def test_arerank_empty_candidates_returns_empty() -> None:
    reranker = OpenRouterReranker(api_key="k")
    assert await reranker.arerank("q", [], top_k=5) == []


def test_rerank_sync_path() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "results": [
                    {"index": 1, "relevance_score": 0.9, "document": {"text": "..."}},
                    {"index": 0, "relevance_score": 0.4, "document": {"text": "..."}},
                ]
            },
        )

    client = httpx.Client(transport=_mock_transport(handler))
    reranker = OpenRouterReranker(api_key="k", sync_client=client)
    out = reranker.rerank("q", _chunks()[:2], top_k=5)
    client.close()

    assert [r.chunk.source for r in out] == ["s2", "s1"]


@pytest.mark.asyncio
async def test_arerank_drops_invalid_indices() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "results": [
                    {"index": 99, "relevance_score": 0.9},  # out of range
                    {"index": 0, "relevance_score": 0.5},
                ]
            },
        )

    client = httpx.AsyncClient(transport=_mock_transport(handler))
    reranker = OpenRouterReranker(api_key="k", async_client=client)
    out = await reranker.arerank("q", _chunks(), top_k=5)
    await client.aclose()

    assert len(out) == 1
    assert out[0].chunk.source == "s1"
