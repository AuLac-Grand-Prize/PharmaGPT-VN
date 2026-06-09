"""POST /v1/embeddings — BGE-M3 **dense** vectors, one per input (Plan §3.4.2).

The handler stays thin: it pulls an embedder from the DI factory
(`get_embedder`), encodes the whole batch in one call, and returns the dense
vectors in input order. Sparse / ColBERT outputs are intentionally not exposed
here — the response models dense only (`list[list[float]]`).

The embedder is injected via `Depends`, so tests override it through
`app.dependency_overrides[get_embedder]` with a fake that returns fixed-length
vectors — no model download, no torch, no network.
"""

from __future__ import annotations

from typing import Protocol

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from pharmagpt_vn.api.dependencies import get_embedder

router = APIRouter()


class _EncodedPair(Protocol):
    """Minimal shape the route needs from an embedder result: a dense vector."""

    dense: list[float]


class _Embedder(Protocol):
    """Structural type for anything the route can embed with.

    `BGEM3Embedder` satisfies this, and so does any test fake — the route never
    depends on the concrete class.
    """

    def encode(self, texts: list[str]) -> list[_EncodedPair]: ...


class EmbedRequest(BaseModel):
    inputs: list[str]
    model: str = "bge-m3"


class EmbedResponse(BaseModel):
    embeddings: list[list[float]]
    model: str
    dim: int


@router.post("/embeddings", response_model=EmbedResponse)
async def embeddings(
    req: EmbedRequest,
    embedder: _Embedder = Depends(get_embedder),
) -> EmbedResponse:
    # Empty input is a deterministic no-op: never invoke the embedder, return
    # an empty result with dim 0 (R2). Avoids a needless model load on [].
    if not req.inputs:
        return EmbedResponse(embeddings=[], model=req.model, dim=0)

    pairs = embedder.encode(req.inputs)
    dense_vectors = [list(pair.dense) for pair in pairs]
    dim = len(dense_vectors[0]) if dense_vectors else 0
    return EmbedResponse(embeddings=dense_vectors, model=req.model, dim=dim)
