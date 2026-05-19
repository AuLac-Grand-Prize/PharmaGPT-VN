"""Qdrant adapter for the ingest pipeline.

Conforms to `pharmagpt_vn.rag.ingest.VectorStore`. Imports of `qdrant_client`
are deferred so unit tests that exercise `ingest` with stubs do not need the
dependency installed.

Collection layout matches what `HybridRetriever` expects:
  - dense  : named vector "dense"  (cosine, BGE-M3 = 1024 dim)
  - sparse : named sparse vector "sparse"
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

from pharmagpt_vn.rag.ingest import IngestPoint

logger = logging.getLogger(__name__)


class QdrantVectorStore:
    """Sync Qdrant adapter for one-shot ingest scripts.

    Args:
      url:        Qdrant URL (e.g. http://localhost:6333) or ":memory:" for
                  an embedded local instance (great for tests / smoke runs).
      collection: target collection name.
      api_key:    optional Qdrant Cloud API key.
      dense_vector_name / sparse_vector_name: must match the retriever side.
      recreate:   if True, drop the collection on `ensure_collection`. Use only
                  for clean re-ingest; ingest is otherwise idempotent because
                  IDs are stable uuid5(source|parent_path|offsets|text-prefix).
    """

    DEFAULT_DENSE_NAME = "dense"
    DEFAULT_SPARSE_NAME = "sparse"

    def __init__(
        self,
        url: str,
        collection: str,
        api_key: str | None = None,
        *,
        dense_vector_name: str = DEFAULT_DENSE_NAME,
        sparse_vector_name: str = DEFAULT_SPARSE_NAME,
        recreate: bool = False,
        client: Any | None = None,
    ) -> None:
        self._url = url
        self._collection = collection
        self._api_key = api_key
        self._dense_name = dense_vector_name
        self._sparse_name = sparse_vector_name
        self._recreate = recreate
        self._client = client  # Tests can inject an in-memory QdrantClient.

    # ------------------------------------------------------------------
    # VectorStore protocol
    # ------------------------------------------------------------------

    def ensure_collection(self, dim: int) -> None:
        client = self._connect()
        rest = self._rest()
        exists = client.collection_exists(self._collection)
        if exists and not self._recreate:
            logger.info("qdrant: collection '%s' exists, skip create", self._collection)
            return
        if exists and self._recreate:
            logger.warning("qdrant: dropping collection '%s' for recreate", self._collection)
            client.delete_collection(self._collection)
        client.create_collection(
            collection_name=self._collection,
            vectors_config={
                self._dense_name: rest.VectorParams(size=dim, distance=rest.Distance.COSINE),
            },
            sparse_vectors_config={
                self._sparse_name: rest.SparseVectorParams(),
            },
        )
        logger.info("qdrant: created collection '%s' (dense_dim=%d)", self._collection, dim)

    def upsert_batch(self, points: Sequence[IngestPoint]) -> None:
        if not points:
            return
        client = self._connect()
        rest = self._rest()
        struct_points = [self._to_point_struct(p, rest) for p in points]
        client.upsert(collection_name=self._collection, points=struct_points)
        logger.info("qdrant: upserted %d points into '%s'", len(points), self._collection)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _to_point_struct(self, p: IngestPoint, rest: Any) -> Any:
        indices, values = _split_sparse(p.sparse)
        return rest.PointStruct(
            id=p.id,
            vector={
                self._dense_name: p.dense,
                self._sparse_name: rest.SparseVector(indices=indices, values=values),
            },
            payload=p.payload,
        )

    def _connect(self) -> Any:
        if self._client is not None:
            return self._client
        from qdrant_client import QdrantClient  # type: ignore[import-not-found]

        if self._url == ":memory:":
            self._client = QdrantClient(":memory:")
        else:
            self._client = QdrantClient(url=self._url, api_key=self._api_key)
        return self._client

    @staticmethod
    def _rest() -> Any:
        from qdrant_client.http import models as rest  # type: ignore[import-not-found]

        return rest


def _split_sparse(weights: dict[int, float]) -> tuple[list[int], list[float]]:
    if not weights:
        return [], []
    indices = list(weights.keys())
    values = [float(weights[i]) for i in indices]
    return indices, values
