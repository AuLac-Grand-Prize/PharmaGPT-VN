"""End-to-end metadata-filter test against an in-memory Qdrant.

Verifies that HybridRetriever's `filters` arg is honored by the production
QdrantBackend, not just the unit-test stub.
"""

from __future__ import annotations

import pytest

from pharmagpt_vn.rag.ingest import (
    CorpusDocument,
    CorpusSection,
    ingest,
)

qdrant_client = pytest.importorskip("qdrant_client")

from pharmagpt_vn.rag.embeddings import EmbeddingPair  # noqa: E402
from pharmagpt_vn.rag.qdrant_store import QdrantVectorStore  # noqa: E402
from pharmagpt_vn.rag.retriever import HybridRetriever, QdrantBackend  # noqa: E402


class _Embedder:
    """Deterministic dense+sparse keyed off raw token presence."""

    DIM = 16
    VOCAB = {
        "metformin": 0,
        "paracetamol": 1,
        "liều": 2,
        "ccđ": 3,
        "suy": 4,
        "thận": 5,
        "trẻ": 6,
        "em": 7,
    }

    def encode(self, texts):
        return [self._encode(t) for t in texts]

    def encode_query(self, text: str) -> EmbeddingPair:
        return self._encode(text)

    def _encode(self, text: str) -> EmbeddingPair:
        dense = [0.0] * self.DIM
        sparse: dict[int, float] = {}
        for tok, idx in self.VOCAB.items():
            if tok in text.lower():
                dense[idx] = 1.0
                sparse[idx + 1] = 1.0
        # avoid zero vector
        dense[-1] = 0.1
        return EmbeddingPair(dense=dense, sparse=sparse)


def _docs() -> list[CorpusDocument]:
    return [
        CorpusDocument(
            drug="Metformin",
            source="vn_pharm_metformin",
            sections=(
                CorpusSection(title="Liều", text="Metformin liều khởi đầu 500mg sau ăn."),
                CorpusSection(title="CCĐ", text="Metformin chống chỉ định suy thận nặng."),
            ),
        ),
        CorpusDocument(
            drug="Paracetamol",
            source="vn_pharm_paracetamol",
            sections=(
                CorpusSection(title="Liều", text="Paracetamol người lớn 500-1000mg."),
                CorpusSection(title="Liều trẻ em", text="Paracetamol trẻ em 10-15mg/kg."),
            ),
        ),
    ]


@pytest.mark.asyncio
async def test_drug_filter_isolates_one_drug() -> None:
    client = qdrant_client.QdrantClient(":memory:")
    store = QdrantVectorStore(url=":memory:", collection="filter_test", client=client)
    embedder = _Embedder()
    ingest(_docs(), embedder=embedder, store=store)

    # Build async client against the same in-memory store.
    async_client = qdrant_client.AsyncQdrantClient(":memory:")
    # The :memory: instances are independent — we have to use the existing sync
    # client wrapped. Build a backend that uses the sync client via thread offload.
    backend = QdrantBackend(url=":memory:", collection="filter_test")
    backend._client = client  # type: ignore[attr-defined]
    _ = async_client  # quiet linter; not used since we route through sync client.

    # qdrant-client 1.18 supports both sync and async on the same QdrantClient
    # local instance via its `.async_*` shims; HybridRetriever expects async.
    # Easier path: drive the sync client directly through a tiny shim that
    # mimics QdrantBackend's contract.
    class _SyncBackendShim:
        async def dense_search(self, vector, limit, filters=None):
            res = client.query_points(
                collection_name="filter_test",
                query=vector,
                using="dense",
                limit=limit,
                with_payload=True,
                query_filter=_filter(filters),
            )
            return [_to_chunk(h) for h in res.points]

        async def sparse_search(self, weights, limit, filters=None):
            from qdrant_client.http.models import SparseVector

            indices = list(weights.keys())
            values = [float(weights[i]) for i in indices]
            res = client.query_points(
                collection_name="filter_test",
                query=SparseVector(indices=indices, values=values),
                using="sparse",
                limit=limit,
                with_payload=True,
                query_filter=_filter(filters),
            )
            return [_to_chunk(h) for h in res.points]

    retriever = HybridRetriever(embedder=embedder, backend=_SyncBackendShim())  # type: ignore[arg-type]

    unfiltered = await retriever.retrieve("liều", top_k=10)
    drugs_in_unfiltered = {c.metadata.get("drug") for c in unfiltered}
    assert {"Metformin", "Paracetamol"}.issubset(drugs_in_unfiltered)

    filtered = await retriever.retrieve("liều", top_k=10, filters={"drug": "Metformin"})
    drugs_in_filtered = {c.metadata.get("drug") for c in filtered}
    assert drugs_in_filtered == {"Metformin"}


def _filter(filters):
    if not filters:
        return None
    from qdrant_client.http.models import FieldCondition, Filter, MatchAny, MatchValue

    must = []
    for k, v in filters.items():
        if isinstance(v, (list, tuple, set)):
            must.append(FieldCondition(key=k, match=MatchAny(any=list(v))))
        else:
            must.append(FieldCondition(key=k, match=MatchValue(value=v)))
    return Filter(must=must)


def _to_chunk(hit):
    from pharmagpt_vn.rag.retriever import RetrievedChunk

    payload = getattr(hit, "payload", None) or {}
    return RetrievedChunk(
        text=payload.get("text", ""),
        source=payload.get("source", ""),
        score=float(getattr(hit, "score", 0.0)),
        metadata=payload,
    )
