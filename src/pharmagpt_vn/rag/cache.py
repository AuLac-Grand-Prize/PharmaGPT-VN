"""Embedding + retrieval cache layer.

Cuts the obvious wins: identical user queries within a session re-encode
through BGE-M3 (~200ms on H100, much more on CPU) and re-hit Qdrant. A tiny
LRU in front of both is essentially free latency.

Design:
  - `EmbeddingCache` / `RetrievalCache` are Protocols → swap with Redis without
    touching the wrappers.
  - `CachedEmbedder` / `CachedRetriever` are drop-in: they implement the same
    `encode_query` / `retrieve` interface as the wrapped collaborator.
  - SHA-256 keys keep this safe across processes (vs. Python hash() which is
    randomized per interpreter).
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections import OrderedDict
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Generic, Protocol, TypeVar

from pharmagpt_vn.rag.embeddings import EmbeddingPair
from pharmagpt_vn.rag.retriever import HybridRetriever, RetrievedChunk

logger = logging.getLogger(__name__)

K = TypeVar("K")
V = TypeVar("V")


# ---------------------------------------------------------------------------
# Cache protocols + a single in-memory LRU implementation that backs both.
# ---------------------------------------------------------------------------


class Cache(Protocol[K, V]):
    def get(self, key: K) -> V | None: ...
    def put(self, key: K, value: V) -> None: ...


@dataclass
class CacheStats:
    hits: int = 0
    misses: int = 0
    evictions: int = 0

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0


class InMemoryLRUCache(Generic[K, V]):
    """Bounded LRU. Thread-unsafe by design — wrap with a lock for multi-thread."""

    def __init__(self, max_size: int = 1024) -> None:
        if max_size < 1:
            raise ValueError("max_size must be >= 1")
        self._store: OrderedDict[K, V] = OrderedDict()
        self._max_size = max_size
        self.stats = CacheStats()

    def get(self, key: K) -> V | None:
        if key in self._store:
            self._store.move_to_end(key)
            self.stats.hits += 1
            return self._store[key]
        self.stats.misses += 1
        return None

    def put(self, key: K, value: V) -> None:
        if key in self._store:
            self._store.move_to_end(key)
            self._store[key] = value
            return
        self._store[key] = value
        if len(self._store) > self._max_size:
            self._store.popitem(last=False)
            self.stats.evictions += 1

    def __len__(self) -> int:
        return len(self._store)


# ---------------------------------------------------------------------------
# CachedEmbedder
# ---------------------------------------------------------------------------


class _QueryEmbedder(Protocol):
    def encode_query(self, text: str) -> EmbeddingPair: ...


@dataclass
class CachedEmbedder:
    """Wraps any `encode_query` embedder with an LRU keyed by normalized query."""

    inner: _QueryEmbedder
    cache: Cache[str, EmbeddingPair] = field(default_factory=lambda: InMemoryLRUCache(max_size=512))

    def encode_query(self, text: str) -> EmbeddingPair:
        key = _hash_text(text)
        cached = self.cache.get(key)
        if cached is not None:
            return cached
        pair = self.inner.encode_query(text)
        self.cache.put(key, pair)
        return pair


# ---------------------------------------------------------------------------
# CachedRetriever
# ---------------------------------------------------------------------------


@dataclass
class CachedRetriever:
    """Wraps a HybridRetriever-shaped object with an LRU keyed by query + filters + top_k.

    `inner` is duck-typed: anything with an async `retrieve(query, top_k, filters=...)`
    method. The Hybrid retriever, MultiQueryRetriever, etc. all qualify.
    """

    inner: HybridRetriever
    cache: Cache[str, tuple[RetrievedChunk, ...]] = field(
        default_factory=lambda: InMemoryLRUCache(max_size=256)
    )

    async def retrieve(
        self,
        query: str,
        top_k: int = 5,
        filters: dict[str, object] | None = None,
    ) -> list[RetrievedChunk]:
        key = _retrieval_key(query, top_k, filters)
        cached = self.cache.get(key)
        if cached is not None:
            return list(cached)
        chunks = await self.inner.retrieve(query, top_k=top_k, filters=filters)
        self.cache.put(key, tuple(chunks))
        return chunks


# ---------------------------------------------------------------------------
# Key helpers
# ---------------------------------------------------------------------------


def _hash_text(text: str) -> str:
    norm = " ".join(text.lower().split())
    return hashlib.sha256(norm.encode("utf-8")).hexdigest()


def _retrieval_key(query: str, top_k: int, filters: dict[str, object] | None) -> str:
    payload = {
        "q": " ".join(query.lower().split()),
        "k": top_k,
        "f": _canonical_filters(filters),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def _canonical_filters(filters: dict[str, object] | None) -> dict[str, object]:
    if not filters:
        return {}
    out: dict[str, object] = {}
    for k in sorted(filters.keys()):
        v = filters[k]
        if isinstance(v, (list, tuple, set)):
            out[k] = sorted(str(x) for x in v)
        else:
            out[k] = v
    return out


__all__ = [
    "Cache",
    "CacheStats",
    "CachedEmbedder",
    "CachedRetriever",
    "InMemoryLRUCache",
]


# `Iterable` import retained for downstream callers that compose cache + others.
_ = Iterable
