"""Hybrid retriever — BM25 lexical + BGE-M3 dense → merged candidates (Plan §3.4.4).

Two retrieval calls run in parallel:
  - dense: Qdrant cosine-search over BGE-M3 dense vectors
  - sparse: Qdrant sparse-vector search using BGE-M3 lexical weights (BM25-like)

Results are merged with reciprocal-rank-fusion before the cross-encoder reranks.
The Qdrant client and embedder are injected so tests can swap them out.
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True)
class RetrievedChunk:
    text: str
    source: str
    score: float
    metadata: dict


class _Embedder(Protocol):
    def encode_query(self, text: str): ...  # type: ignore[no-untyped-def]


class _QdrantSearch(Protocol):
    async def dense_search(
        self, vector: list[float], limit: int
    ) -> list[RetrievedChunk]: ...
    async def sparse_search(
        self, weights: dict[int, float], limit: int
    ) -> list[RetrievedChunk]: ...


class HybridRetriever:
    def __init__(
        self,
        qdrant_url: str = "",
        collection: str = "",
        embedding_model: str = "BAAI/bge-m3",
        embedder: _Embedder | None = None,
        backend: _QdrantSearch | None = None,
        rrf_k: int = 60,
    ) -> None:
        self.qdrant_url = qdrant_url
        self.collection = collection
        self.embedding_model = embedding_model
        self._embedder = embedder
        self._backend = backend
        self._rrf_k = rrf_k

    async def retrieve(self, query: str, top_k: int = 5) -> list[RetrievedChunk]:
        if self._embedder is None or self._backend is None:
            return []  # placeholder until production wires both
        pair = self._embedder.encode_query(query)
        pool = max(top_k * 4, 20)
        dense_task = asyncio.create_task(self._backend.dense_search(pair.dense, pool))
        sparse_task = asyncio.create_task(self._backend.sparse_search(pair.sparse, pool))
        dense, sparse = await asyncio.gather(dense_task, sparse_task)
        merged = reciprocal_rank_fusion(dense, sparse, k=self._rrf_k)
        return merged[:top_k]


def reciprocal_rank_fusion(
    *result_lists: Iterable[RetrievedChunk], k: int = 60
) -> list[RetrievedChunk]:
    scored: dict[tuple[str, str], tuple[float, RetrievedChunk]] = {}
    for results in result_lists:
        for rank, item in enumerate(results, start=1):
            key = (item.source, item.text[:120])
            inc = 1.0 / (k + rank)
            current = scored.get(key)
            if current is None:
                scored[key] = (inc, item)
            else:
                scored[key] = (current[0] + inc, current[1])
    fused = [
        RetrievedChunk(
            text=chunk.text,
            source=chunk.source,
            score=score,
            metadata=chunk.metadata,
        )
        for score, chunk in scored.values()
    ]
    fused.sort(key=lambda c: c.score, reverse=True)
    return fused


class QdrantBackend:
    """Thin async wrapper around qdrant-client. Lazy-loaded for testability."""

    def __init__(self, url: str, collection: str, api_key: str | None = None) -> None:
        self._url = url
        self._collection = collection
        self._api_key = api_key
        self._client: Any | None = None

    def _connect(self):  # type: ignore[no-untyped-def]
        if self._client is not None:
            return self._client
        from qdrant_client import AsyncQdrantClient  # type: ignore[import-not-found]

        self._client = AsyncQdrantClient(url=self._url, api_key=self._api_key)
        return self._client

    async def dense_search(
        self, vector: list[float], limit: int
    ) -> list[RetrievedChunk]:
        client = self._connect()
        hits = await client.search(
            collection_name=self._collection,
            query_vector=("dense", vector),
            limit=limit,
            with_payload=True,
        )
        return [_to_chunk(h) for h in hits]

    async def sparse_search(
        self, weights: dict[int, float], limit: int
    ) -> list[RetrievedChunk]:
        from qdrant_client.http.models import (  # type: ignore[import-not-found]
            NamedSparseVector,
            SparseVector,
        )

        client = self._connect()
        indices = list(weights.keys())
        values = [weights[i] for i in indices]
        hits = await client.search(
            collection_name=self._collection,
            query_vector=NamedSparseVector(
                name="sparse",
                vector=SparseVector(indices=indices, values=values),
            ),
            limit=limit,
            with_payload=True,
        )
        return [_to_chunk(h) for h in hits]


def _to_chunk(hit: Any) -> RetrievedChunk:
    payload = getattr(hit, "payload", None) or {}
    return RetrievedChunk(
        text=payload.get("text", ""),
        source=payload.get("source", ""),
        score=float(getattr(hit, "score", 0.0)),
        metadata=payload,
    )
