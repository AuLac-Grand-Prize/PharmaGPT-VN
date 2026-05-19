"""Corpus ingest pipeline: load → chunk → (optional) contextual enrich → embed → upsert.

Pure orchestration logic — heavy collaborators (embedder, vector store, enricher)
are injected via Protocol so the pipeline is testable without GPU or Qdrant.

`scripts/ingest_corpus.py` is the production wiring; `tests/unit/test_ingest.py`
exercises this module with stubs.
"""

from __future__ import annotations

import json
import logging
import uuid
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Protocol

from pharmagpt_vn.rag.chunker import Chunk, Section, chunk_section
from pharmagpt_vn.rag.embeddings import EmbeddingPair

logger = logging.getLogger(__name__)

_NAMESPACE = uuid.UUID("8b1f7cf6-2f8b-4d7a-9c1e-1f2a5e9d4c10")


# ---------------------------------------------------------------------------
# Domain types — the corpus-on-disk shape.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CorpusSection:
    title: str
    text: str


@dataclass(frozen=True)
class CorpusDocument:
    drug: str
    source: str
    sections: tuple[CorpusSection, ...] = field(default_factory=tuple)


# ---------------------------------------------------------------------------
# Output of the pipeline — one point per chunk, ready for Qdrant upsert.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class IngestPoint:
    id: str
    dense: list[float]
    sparse: dict[int, float]
    payload: dict[str, Any]


@dataclass(frozen=True)
class IngestStats:
    documents: int
    chunks: int
    embedded: int
    upserted: int


# ---------------------------------------------------------------------------
# Collaborator protocols.
# ---------------------------------------------------------------------------


class Embedder(Protocol):
    def encode(self, texts: Sequence[str]) -> list[EmbeddingPair]: ...


class VectorStore(Protocol):
    def ensure_collection(self, dim: int) -> None: ...
    def upsert_batch(self, points: Sequence[IngestPoint]) -> None: ...


class Enricher(Protocol):
    """Contextual Retrieval enricher — prefixes chunks with LLM-generated context.

    The wider `rag.contextual.ContextualEnricher` already conforms; this Protocol
    keeps the ingest module agnostic.
    """

    def enrich(self, document: str, chunks: Sequence[Chunk]) -> list[Chunk]: ...


# ---------------------------------------------------------------------------
# Corpus loading — accepts the envelope JSON (`{"documents": [...]}`) or JSONL.
# ---------------------------------------------------------------------------


def load_corpus(path: Path) -> list[CorpusDocument]:
    """Load corpus from a JSON envelope or a JSONL file."""
    suffix = path.suffix.lower()
    if suffix == ".json":
        raw = json.loads(path.read_text(encoding="utf-8"))
        docs_raw = raw["documents"] if isinstance(raw, dict) and "documents" in raw else raw
        return [_parse_document(d) for d in docs_raw]
    if suffix in {".jsonl", ".ndjson"}:
        out: list[CorpusDocument] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            out.append(_parse_document(json.loads(line)))
        return out
    raise ValueError(f"Unsupported corpus format: {path.suffix}")


def _parse_document(raw: dict[str, Any]) -> CorpusDocument:
    return CorpusDocument(
        drug=raw["drug"],
        source=raw["source"],
        sections=tuple(
            CorpusSection(title=s["title"], text=s["text"]) for s in raw.get("sections", [])
        ),
    )


# ---------------------------------------------------------------------------
# Corpus → chunker.Section transform. Keeps parent_path = (drug, section_title)
# so retrieval payloads can show "Metformin / Chống chỉ định" context.
# ---------------------------------------------------------------------------


def document_to_sections(doc: CorpusDocument) -> list[Section]:
    return [
        Section(
            text=s.text,
            source=doc.source,
            parent_path=(doc.drug, s.title),
            drug_names=(doc.drug,),
        )
        for s in doc.sections
    ]


# ---------------------------------------------------------------------------
# The pipeline.
# ---------------------------------------------------------------------------


def ingest(
    documents: Sequence[CorpusDocument],
    *,
    embedder: Embedder,
    store: VectorStore,
    enricher: Enricher | None = None,
    chunker_kwargs: dict[str, Any] | None = None,
    batch_size: int = 32,
    dry_run: bool = False,
) -> IngestStats:
    """Chunk → (enrich) → embed → upsert. Returns aggregate counts.

    `enricher` is optional Contextual Retrieval (Anthropic 9/2024). When wired,
    the enriched text is embedded but the *original* chunk text is what we store
    in the payload — the LLM should never see the meta-context at inference.

    `dry_run=True` runs everything except the vector-store calls; useful for
    counting chunks or smoke-testing the chunking strategy.
    """
    chunker_kwargs = chunker_kwargs or {}
    chunks_with_doc: list[tuple[Chunk, Chunk]] = []  # (original_chunk, embed_chunk)

    for doc in documents:
        sections = document_to_sections(doc)
        if not sections:
            continue
        for section in sections:
            chunks = chunk_section(section, **chunker_kwargs)
            embed_chunks = (
                enricher.enrich(section.text, chunks) if enricher is not None else list(chunks)
            )
            chunks_with_doc.extend(zip(chunks, embed_chunks))

    total_chunks = len(chunks_with_doc)
    if total_chunks == 0:
        if not dry_run:
            # Caller may still want the collection created.
            store.ensure_collection(_probe_dim(embedder))
        return IngestStats(documents=len(documents), chunks=0, embedded=0, upserted=0)

    dim: int | None = None
    embedded = 0
    upserted = 0

    for batch in _batched(chunks_with_doc, batch_size):
        embed_texts = [embed_chunk.text for (_orig, embed_chunk) in batch]
        pairs = embedder.encode(embed_texts)
        if dim is None:
            dim = len(pairs[0].dense)
            if not dry_run:
                store.ensure_collection(dim)
        points = [
            _build_point(orig, pair) for (orig, _embed_chunk), pair in zip(batch, pairs)
        ]
        embedded += len(points)
        if not dry_run:
            store.upsert_batch(points)
            upserted += len(points)
        logger.info(
            "ingest: batch=%d cumulative_embedded=%d", len(points), embedded
        )

    return IngestStats(
        documents=len(documents),
        chunks=total_chunks,
        embedded=embedded,
        upserted=upserted,
    )


def _probe_dim(embedder: Embedder) -> int:
    """Single-text probe so the collection can be created on empty corpora."""
    pair = embedder.encode(["probe"])[0]
    return len(pair.dense)


def _build_point(chunk: Chunk, pair: EmbeddingPair) -> IngestPoint:
    return IngestPoint(
        id=_chunk_id(chunk),
        dense=list(pair.dense),
        sparse=dict(pair.sparse),
        payload=_chunk_payload(chunk),
    )


def _chunk_id(chunk: Chunk) -> str:
    key = f"{chunk.source}|{'/'.join(chunk.parent_path)}|{chunk.start_char}|{chunk.end_char}|{chunk.text[:160]}"
    return str(uuid.uuid5(_NAMESPACE, key))


def _chunk_payload(chunk: Chunk) -> dict[str, Any]:
    drug = chunk.drug_names[0] if chunk.drug_names else (chunk.parent_path[0] if chunk.parent_path else "")
    section = chunk.parent_path[-1] if chunk.parent_path else ""
    return {
        "text": chunk.text,
        "source": chunk.source,
        "drug": drug,
        "section": section,
        "parent_path": list(chunk.parent_path),
        "drug_names": list(chunk.drug_names),
        "start_char": chunk.start_char,
        "end_char": chunk.end_char,
    }


def _batched(items: Iterable[Any], size: int) -> Iterator[list[Any]]:
    batch: list[Any] = []
    for item in items:
        batch.append(item)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch


# Exposed for convenience: a hook for callers that already have Chunk objects.
__all__ = [
    "CorpusDocument",
    "CorpusSection",
    "Embedder",
    "Enricher",
    "IngestPoint",
    "IngestStats",
    "VectorStore",
    "document_to_sections",
    "ingest",
    "load_corpus",
]


# Suppress unused warning: `replace` is exported indirectly for callers that
# may want to mutate chunks before ingest.
_ = replace
