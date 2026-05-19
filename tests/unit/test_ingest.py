from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from pharmagpt_vn.rag.chunker import Chunk
from pharmagpt_vn.rag.embeddings import EmbeddingPair
from pharmagpt_vn.rag.ingest import (
    CorpusDocument,
    CorpusSection,
    IngestPoint,
    document_to_sections,
    ingest,
    load_corpus,
)

# ---------------------------------------------------------------------------
# Fixtures: doubles for the heavy collaborators (embedder + vector store).
# ---------------------------------------------------------------------------


@dataclass
class _RecordingEmbedder:
    """Emit deterministic EmbeddingPair per text. Records every batch call."""

    dim: int = 4
    batches: list[list[str]] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        self.batches = []

    def encode(self, texts: Sequence[str]) -> list[EmbeddingPair]:
        self.batches.append(list(texts))
        out: list[EmbeddingPair] = []
        for i, t in enumerate(texts):
            base = (len(t) % 7) + 1
            dense = [float((i + j + base) % 5) / 5.0 for j in range(self.dim)]
            sparse = {hash(tok) & 0xFFFF: 1.0 for tok in t.split()[:3]}
            out.append(EmbeddingPair(dense=dense, sparse=sparse))
        return out


@dataclass
class _RecordingStore:
    ensure_called_with: int | None = None
    upserted: list[IngestPoint] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        self.upserted = []

    def ensure_collection(self, dim: int) -> None:
        self.ensure_called_with = dim

    def upsert_batch(self, points: Sequence[IngestPoint]) -> None:
        self.upserted.extend(points)


@dataclass
class _StubEnricher:
    """Prepends a deterministic context string to each chunk's text."""

    prefix: str = "CTX:"

    def enrich(self, document: str, chunks: Sequence[Chunk]) -> list[Chunk]:
        from dataclasses import replace

        return [replace(c, text=f"{self.prefix}{c.text}") for c in chunks]


# ---------------------------------------------------------------------------
# Sample corpus fixtures.
# ---------------------------------------------------------------------------


SAMPLE_JSON = {
    "documents": [
        {
            "drug": "Metformin",
            "source": "wikipedia_vi_metformin",
            "sections": [
                {"title": "Chỉ định", "text": "Dùng cho đái tháo đường týp 2."},
                {
                    "title": "Chống chỉ định",
                    "text": "Không dùng cho suy thận nặng eGFR < 30.",
                },
            ],
        },
        {
            "drug": "Paracetamol",
            "source": "wikipedia_vi_paracetamol",
            "sections": [{"title": "Liều", "text": "Người lớn 500-1000mg mỗi 4-6 giờ."}],
        },
    ]
}


@pytest.fixture
def json_corpus(tmp_path: Path) -> Path:
    p = tmp_path / "corpus.json"
    p.write_text(json.dumps(SAMPLE_JSON, ensure_ascii=False), encoding="utf-8")
    return p


@pytest.fixture
def jsonl_corpus(tmp_path: Path) -> Path:
    p = tmp_path / "corpus.jsonl"
    lines = [json.dumps(doc, ensure_ascii=False) for doc in SAMPLE_JSON["documents"]]
    p.write_text("\n".join(lines), encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# load_corpus
# ---------------------------------------------------------------------------


def test_load_corpus_reads_json_envelope(json_corpus: Path) -> None:
    docs = load_corpus(json_corpus)
    assert [d.drug for d in docs] == ["Metformin", "Paracetamol"]
    assert docs[0].sections[0].title == "Chỉ định"


def test_load_corpus_reads_jsonl(jsonl_corpus: Path) -> None:
    docs = load_corpus(jsonl_corpus)
    assert len(docs) == 2
    assert docs[1].drug == "Paracetamol"


def test_load_corpus_skips_blank_lines_in_jsonl(tmp_path: Path) -> None:
    p = tmp_path / "c.jsonl"
    p.write_text(
        "\n"
        + json.dumps({"drug": "X", "source": "s", "sections": [{"title": "t", "text": "abc"}]})
        + "\n\n",
        encoding="utf-8",
    )
    docs = load_corpus(p)
    assert len(docs) == 1


def test_load_corpus_raises_for_unknown_suffix(tmp_path: Path) -> None:
    p = tmp_path / "c.txt"
    p.write_text("nope", encoding="utf-8")
    with pytest.raises(ValueError, match="Unsupported"):
        load_corpus(p)


# ---------------------------------------------------------------------------
# document_to_sections
# ---------------------------------------------------------------------------


def test_document_to_sections_carries_drug_and_path() -> None:
    doc = CorpusDocument(
        drug="Metformin",
        source="src",
        sections=(
            CorpusSection(title="Liều", text="500mg"),
            CorpusSection(title="CCĐ", text="Suy thận"),
        ),
    )
    sections = document_to_sections(doc)
    assert len(sections) == 2
    assert sections[0].parent_path == ("Metformin", "Liều")
    assert sections[0].drug_names == ("Metformin",)
    assert sections[1].source == "src"


# ---------------------------------------------------------------------------
# ingest()
# ---------------------------------------------------------------------------


def _docs() -> list[CorpusDocument]:
    return [
        CorpusDocument(
            drug=d["drug"],
            source=d["source"],
            sections=tuple(
                CorpusSection(title=s["title"], text=s["text"]) for s in d["sections"]
            ),
        )
        for d in SAMPLE_JSON["documents"]
    ]


def test_ingest_creates_collection_with_embedder_dim() -> None:
    emb = _RecordingEmbedder(dim=8)
    store = _RecordingStore()
    ingest(_docs(), embedder=emb, store=store, batch_size=16)
    assert store.ensure_called_with == 8


def test_ingest_embeds_in_batches() -> None:
    emb = _RecordingEmbedder(dim=4)
    store = _RecordingStore()
    ingest(_docs(), embedder=emb, store=store, batch_size=2)
    # 3 sections total → batches of size 2,1
    assert [len(b) for b in emb.batches] == [2, 1]


def test_ingest_upserts_one_point_per_chunk() -> None:
    emb = _RecordingEmbedder(dim=4)
    store = _RecordingStore()
    stats = ingest(_docs(), embedder=emb, store=store)
    # demo sections are short → 1 chunk each → 3 points
    assert stats.documents == 2
    assert stats.chunks == 3
    assert stats.upserted == 3
    assert len(store.upserted) == 3


def test_ingest_payload_carries_drug_section_source_and_text() -> None:
    emb = _RecordingEmbedder(dim=4)
    store = _RecordingStore()
    ingest(_docs(), embedder=emb, store=store)
    p = store.upserted[0]
    assert p.payload["drug"] == "Metformin"
    assert p.payload["section"] == "Chỉ định"
    assert p.payload["source"] == "wikipedia_vi_metformin"
    assert "đái tháo đường" in p.payload["text"]
    assert p.payload["parent_path"] == ["Metformin", "Chỉ định"]


def test_ingest_point_id_is_deterministic() -> None:
    emb = _RecordingEmbedder(dim=4)
    store_a, store_b = _RecordingStore(), _RecordingStore()
    ingest(_docs(), embedder=emb, store=store_a)
    ingest(_docs(), embedder=_RecordingEmbedder(dim=4), store=store_b)
    ids_a = [p.id for p in store_a.upserted]
    ids_b = [p.id for p in store_b.upserted]
    assert ids_a == ids_b
    assert len(set(ids_a)) == len(ids_a)  # unique


def test_ingest_runs_enricher_and_embeds_enriched_text() -> None:
    emb = _RecordingEmbedder(dim=4)
    store = _RecordingStore()
    ingest(_docs(), embedder=emb, store=store, enricher=_StubEnricher(prefix="CTX:"))
    embedded_texts = [t for batch in emb.batches for t in batch]
    assert all(t.startswith("CTX:") for t in embedded_texts)
    # Payload keeps the *original* text so the LLM sees clean retrieval output.
    assert not store.upserted[0].payload["text"].startswith("CTX:")


def test_ingest_dry_run_skips_store_calls() -> None:
    emb = _RecordingEmbedder(dim=4)
    store = _RecordingStore()
    stats = ingest(_docs(), embedder=emb, store=store, dry_run=True)
    assert stats.chunks == 3
    assert stats.upserted == 0
    assert store.upserted == []
    assert store.ensure_called_with is None


def test_ingest_point_sparse_payload_round_trip() -> None:
    emb = _RecordingEmbedder(dim=4)
    store = _RecordingStore()
    ingest(_docs(), embedder=emb, store=store)
    p: IngestPoint = store.upserted[0]
    assert isinstance(p.dense, list) and len(p.dense) == 4
    assert isinstance(p.sparse, dict) and len(p.sparse) >= 1


def test_ingest_skips_empty_documents() -> None:
    emb = _RecordingEmbedder(dim=4)
    store = _RecordingStore()
    docs: list[CorpusDocument] = [
        CorpusDocument(drug="Empty", source="s", sections=()),
        *_docs(),
    ]
    stats = ingest(docs, embedder=emb, store=store)
    assert stats.documents == 3
    assert stats.chunks == 3  # empty doc contributes nothing
