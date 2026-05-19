"""Ingest the VN pharma corpus into Qdrant.

Pipeline: load corpus → chunk → (optional) Contextual Retrieval enrichment →
embed with BGE-M3 (dense + sparse) → upsert to Qdrant.

Usage
-----

    # Dry run on the demo corpus — no Qdrant, no GPU. Prints chunk counts.
    python scripts/ingest_corpus.py --source data/demo_corpus.json --dry-run

    # Real ingest into a local Qdrant (assumes `make services-up` is running).
    python scripts/ingest_corpus.py \\
        --source data/demo_corpus.json \\
        --qdrant-url http://localhost:6333 \\
        --collection pharmagpt_vn_v01

    # Recreate the collection from scratch (drops existing data).
    python scripts/ingest_corpus.py --source ... --recreate

Notes
-----
- Embedder defaults to BGE-M3 (`BAAI/bge-m3`); requires GPU + FlagEmbedding.
  Override with --embedder lexical to use a CPU-only stand-in (demo only,
  produces a deterministic pseudo-dense vector). Useful for smoke tests.
- Contextual enrichment is off by default — it requires an indexing-time LLM
  and would otherwise silently embed naked chunks. Enable explicitly.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from collections.abc import Sequence
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from pharmagpt_vn.rag.embeddings import EmbeddingPair  # noqa: E402
from pharmagpt_vn.rag.ingest import (  # noqa: E402
    CorpusDocument,
    Embedder,
    IngestStats,
    VectorStore,
    ingest,
    load_corpus,
)

logger = logging.getLogger("ingest")


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    documents = _load_documents(args.source)
    logger.info("loaded %d documents from %s", len(documents), args.source)

    embedder = _build_embedder(args)
    store = _build_store(args)

    stats: IngestStats = ingest(
        documents,
        embedder=embedder,
        store=store,
        batch_size=args.batch_size,
        dry_run=args.dry_run,
    )
    logger.info(
        "done: documents=%d chunks=%d embedded=%d upserted=%d",
        stats.documents,
        stats.chunks,
        stats.embedded,
        stats.upserted,
    )
    return 0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", type=Path, required=True, help="Path to corpus JSON or JSONL file")
    p.add_argument(
        "--qdrant-url",
        default=os.getenv("QDRANT_URL", "http://localhost:6333"),
        help="Qdrant URL or ':memory:' for embedded local mode",
    )
    p.add_argument(
        "--qdrant-api-key",
        default=os.getenv("QDRANT_API_KEY"),
        help="Optional Qdrant Cloud API key",
    )
    p.add_argument(
        "--collection",
        default=os.getenv("QDRANT_COLLECTION", "pharmagpt_vn"),
        help="Target collection name",
    )
    p.add_argument("--batch-size", type=int, default=32, help="Embedder batch size")
    p.add_argument(
        "--embedder",
        choices=["bge-m3", "lexical"],
        default="bge-m3",
        help="bge-m3 (production, requires GPU+FlagEmbedding) or lexical (CPU stand-in for smoke tests)",
    )
    p.add_argument(
        "--device",
        default=os.getenv("EMBEDDER_DEVICE", "cuda"),
        help="Device for BGE-M3 (cuda / cpu)",
    )
    p.add_argument(
        "--recreate",
        action="store_true",
        help="Drop the collection before ingest (destructive; use only for clean rebuilds)",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Run chunking + embedding but skip vector-store writes",
    )
    p.add_argument("--log-level", default="INFO")
    return p.parse_args(argv)


def _load_documents(source: Path) -> list[CorpusDocument]:
    if not source.exists():
        raise SystemExit(f"corpus not found: {source}")
    return load_corpus(source)


def _build_embedder(args: argparse.Namespace) -> Embedder:
    if args.embedder == "bge-m3":
        from pharmagpt_vn.rag.embeddings import BGEM3Embedder

        return BGEM3Embedder(device=args.device)
    if args.embedder == "lexical":
        return _LexicalEmbedder()
    raise ValueError(f"unknown embedder: {args.embedder}")


def _build_store(args: argparse.Namespace) -> VectorStore:
    if args.dry_run:
        return _NoopStore()
    from pharmagpt_vn.rag.qdrant_store import QdrantVectorStore

    return QdrantVectorStore(
        url=args.qdrant_url,
        collection=args.collection,
        api_key=args.qdrant_api_key,
        recreate=args.recreate,
    )


class _NoopStore:
    """Used in --dry-run so the ingest pipeline still runs end-to-end."""

    def ensure_collection(self, dim: int) -> None:
        logger.info("dry-run: would create collection with dim=%d", dim)

    def upsert_batch(self, points) -> None:  # type: ignore[no-untyped-def]
        logger.info("dry-run: would upsert %d points", len(points))


class _LexicalEmbedder:
    """Deterministic CPU-only embedder for smoke tests / dry runs.

    NOT for production — produces a pseudo-dense vector from token-frequency
    hashing and a sparse map keyed by token-id. Same EmbeddingPair shape as
    BGE-M3 so the rest of the pipeline doesn't notice.
    """

    DIM = 256

    def encode(self, texts: Sequence[str]) -> list[EmbeddingPair]:
        import math
        import re
        from collections import Counter

        tok_re = re.compile(r"\w+", re.UNICODE)
        out: list[EmbeddingPair] = []
        for text in texts:
            tokens = tok_re.findall(text.lower())
            counts = Counter(tokens)
            dense = [0.0] * self.DIM
            for tok, c in counts.items():
                dense[hash(tok) % self.DIM] += float(c)
            norm = math.sqrt(sum(x * x for x in dense)) or 1.0
            dense = [x / norm for x in dense]
            sparse = {hash(tok) & 0xFFFFFFFF: float(c) for tok, c in counts.items()}
            out.append(EmbeddingPair(dense=dense, sparse=sparse))
        return out


if __name__ == "__main__":
    sys.exit(main())
