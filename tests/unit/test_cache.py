from __future__ import annotations

import pytest

from pharmagpt_vn.rag.cache import (
    CachedEmbedder,
    CachedRetriever,
    InMemoryLRUCache,
)
from pharmagpt_vn.rag.embeddings import EmbeddingPair
from pharmagpt_vn.rag.retriever import RetrievedChunk


# ---------------------------------------------------------------------------
# LRU
# ---------------------------------------------------------------------------


def test_lru_returns_none_on_miss_then_stores() -> None:
    cache: InMemoryLRUCache[str, int] = InMemoryLRUCache(max_size=2)
    assert cache.get("a") is None
    cache.put("a", 1)
    assert cache.get("a") == 1
    assert cache.stats.hits == 1
    assert cache.stats.misses == 1


def test_lru_evicts_oldest_when_full() -> None:
    cache: InMemoryLRUCache[str, int] = InMemoryLRUCache(max_size=2)
    cache.put("a", 1)
    cache.put("b", 2)
    cache.put("c", 3)  # evicts "a"
    assert cache.get("a") is None
    assert cache.get("b") == 2
    assert cache.get("c") == 3
    assert cache.stats.evictions == 1


def test_lru_recently_used_survives_eviction() -> None:
    cache: InMemoryLRUCache[str, int] = InMemoryLRUCache(max_size=2)
    cache.put("a", 1)
    cache.put("b", 2)
    _ = cache.get("a")  # touch "a"
    cache.put("c", 3)  # should evict "b", not "a"
    assert cache.get("a") == 1
    assert cache.get("b") is None


def test_lru_rejects_zero_size() -> None:
    with pytest.raises(ValueError):
        InMemoryLRUCache(max_size=0)


def test_lru_hit_rate_handles_zero() -> None:
    cache: InMemoryLRUCache[str, int] = InMemoryLRUCache(max_size=4)
    assert cache.stats.hit_rate == 0.0


# ---------------------------------------------------------------------------
# CachedEmbedder
# ---------------------------------------------------------------------------


class _CountingEmbedder:
    def __init__(self) -> None:
        self.calls = 0

    def encode_query(self, text: str) -> EmbeddingPair:
        self.calls += 1
        return EmbeddingPair(dense=[float(len(text))], sparse={1: 1.0})


def test_cached_embedder_does_not_re_encode_same_query() -> None:
    inner = _CountingEmbedder()
    emb = CachedEmbedder(inner=inner)
    a = emb.encode_query("metformin liều")
    b = emb.encode_query("metformin liều")
    assert inner.calls == 1
    assert a == b


def test_cached_embedder_normalizes_whitespace_and_case() -> None:
    inner = _CountingEmbedder()
    emb = CachedEmbedder(inner=inner)
    emb.encode_query("Metformin   liều")
    emb.encode_query("metformin liều")
    # Same after lowercasing + collapsing whitespace.
    assert inner.calls == 1


def test_cached_embedder_distinguishes_different_queries() -> None:
    inner = _CountingEmbedder()
    emb = CachedEmbedder(inner=inner)
    emb.encode_query("metformin liều")
    emb.encode_query("paracetamol liều")
    assert inner.calls == 2


# ---------------------------------------------------------------------------
# CachedRetriever
# ---------------------------------------------------------------------------


class _CountingRetriever:
    def __init__(self) -> None:
        self.calls = 0

    async def retrieve(
        self,
        query: str,
        top_k: int = 5,
        filters: dict[str, object] | None = None,
    ) -> list[RetrievedChunk]:
        self.calls += 1
        return [
            RetrievedChunk(
                text=f"hit {self.calls}", source="src", score=1.0, metadata={"q": query}
            )
        ]


@pytest.mark.asyncio
async def test_cached_retriever_serves_cache_on_second_call() -> None:
    inner = _CountingRetriever()
    cr = CachedRetriever(inner=inner)  # type: ignore[arg-type]
    a = await cr.retrieve("metformin", top_k=3)
    b = await cr.retrieve("metformin", top_k=3)
    assert inner.calls == 1
    assert a == b


@pytest.mark.asyncio
async def test_cached_retriever_keys_on_filters() -> None:
    inner = _CountingRetriever()
    cr = CachedRetriever(inner=inner)  # type: ignore[arg-type]
    await cr.retrieve("liều", filters={"drug": "Metformin"})
    await cr.retrieve("liều", filters={"drug": "Paracetamol"})
    assert inner.calls == 2


@pytest.mark.asyncio
async def test_cached_retriever_filter_order_doesnt_matter() -> None:
    inner = _CountingRetriever()
    cr = CachedRetriever(inner=inner)  # type: ignore[arg-type]
    await cr.retrieve("liều", filters={"drug": "Metformin", "section": "Liều"})
    await cr.retrieve("liều", filters={"section": "Liều", "drug": "Metformin"})
    assert inner.calls == 1


@pytest.mark.asyncio
async def test_cached_retriever_keys_on_top_k() -> None:
    inner = _CountingRetriever()
    cr = CachedRetriever(inner=inner)  # type: ignore[arg-type]
    await cr.retrieve("liều", top_k=3)
    await cr.retrieve("liều", top_k=5)
    assert inner.calls == 2
