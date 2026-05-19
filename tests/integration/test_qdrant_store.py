"""Integration test for QdrantVectorStore using qdrant-client's `:memory:` mode.

Skipped automatically if qdrant-client isn't installed (e.g. on CI runners that
only need the unit suite).
"""

from __future__ import annotations

import pytest

from pharmagpt_vn.rag.ingest import (
    CorpusDocument,
    CorpusSection,
    IngestStats,
    ingest,
)

qdrant_client = pytest.importorskip("qdrant_client")

from pharmagpt_vn.rag.qdrant_store import QdrantVectorStore  # noqa: E402


class _DeterministicEmbedder:
    def __init__(self, dim: int = 8) -> None:
        self.dim = dim

    def encode(self, texts):
        from pharmagpt_vn.rag.embeddings import EmbeddingPair

        out = []
        for i, t in enumerate(texts):
            dense = [float(((i + j + len(t)) % 7) / 7.0) or 0.01 for j in range(self.dim)]
            sparse = {hash(tok) & 0xFFFF: 1.0 for tok in t.split()[:4]}
            out.append(EmbeddingPair(dense=dense, sparse=sparse))
        return out


def _docs() -> list[CorpusDocument]:
    return [
        CorpusDocument(
            drug="Metformin",
            source="vn_pharm",
            sections=(
                CorpusSection(title="Liều", text="Khởi đầu 500mg sau ăn."),
                CorpusSection(title="CCĐ", text="Suy thận nặng eGFR < 30."),
            ),
        )
    ]


def _make_store(name: str = "pharmagpt_test") -> QdrantVectorStore:
    client = qdrant_client.QdrantClient(":memory:")
    return QdrantVectorStore(url=":memory:", collection=name, client=client)


def test_ingest_creates_collection_and_upserts() -> None:
    store = _make_store()
    stats: IngestStats = ingest(_docs(), embedder=_DeterministicEmbedder(dim=8), store=store)

    assert stats.upserted == 2
    client = store._connect()  # type: ignore[attr-defined]
    assert client.count("pharmagpt_test", exact=True).count == 2


def test_collection_has_named_dense_and_sparse() -> None:
    store = _make_store("pharmagpt_schema")
    ingest(_docs(), embedder=_DeterministicEmbedder(dim=8), store=store)

    client = store._connect()  # type: ignore[attr-defined]
    info = client.get_collection("pharmagpt_schema")
    vectors = info.config.params.vectors
    sparse = info.config.params.sparse_vectors

    assert "dense" in vectors
    assert vectors["dense"].size == 8
    assert "sparse" in sparse


def test_ingest_is_idempotent_on_rerun() -> None:
    store = _make_store("pharmagpt_idem")
    ingest(_docs(), embedder=_DeterministicEmbedder(dim=8), store=store)
    ingest(_docs(), embedder=_DeterministicEmbedder(dim=8), store=store)

    client = store._connect()  # type: ignore[attr-defined]
    # Same uuid5 IDs → upsert overwrites instead of duplicating.
    assert client.count("pharmagpt_idem", exact=True).count == 2


def test_dense_query_returns_payload_with_drug_and_section() -> None:
    store = _make_store("pharmagpt_q")
    embedder = _DeterministicEmbedder(dim=8)
    ingest(_docs(), embedder=embedder, store=store)

    # Use the first chunk's own vector as the probe; it must come back top-1.
    client = store._connect()  # type: ignore[attr-defined]
    probe = embedder.encode(["Khởi đầu 500mg sau ăn."])[0]
    res = client.query_points(
        collection_name="pharmagpt_q",
        query=probe.dense,
        using="dense",
        limit=2,
        with_payload=True,
    )
    payload = res.points[0].payload
    assert payload["drug"] == "Metformin"
    assert payload["section"] in {"Liều", "CCĐ"}
    assert "text" in payload
