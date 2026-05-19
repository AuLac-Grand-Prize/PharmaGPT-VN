"""Cross-encoder re-ranker (Plan §3.4.4 Stage C).

Wraps BGE-reranker-v2-m3 — runs only on a small candidate set so the cost is
acceptable. Production loads the model once at startup; this module exposes
both sync and async interfaces.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Iterable
from dataclasses import dataclass

from pharmagpt_vn.rag.retriever import RetrievedChunk

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RerankedChunk:
    chunk: RetrievedChunk
    rerank_score: float


class CrossEncoderReranker:
    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-v2-m3",
        use_fp16: bool = True,
        device: str = "cuda",
    ) -> None:
        self.model_name = model_name
        self.use_fp16 = use_fp16
        self.device = device
        self._model: object | None = None

    def _load(self) -> object:
        if self._model is not None:
            return self._model
        from FlagEmbedding import FlagReranker  # type: ignore[import-not-found]

        self._model = FlagReranker(self.model_name, use_fp16=self.use_fp16, devices=[self.device])
        return self._model

    def rerank(
        self, query: str, candidates: Iterable[RetrievedChunk], top_k: int = 10
    ) -> list[RerankedChunk]:
        cand_list = list(candidates)
        if not cand_list:
            return []
        try:
            model = self._load()
            scores = model.compute_score(  # type: ignore[attr-defined]
                [(query, c.text) for c in cand_list], normalize=True
            )
        except Exception:
            # Production must know when the cross-encoder is silently bypassed —
            # this is the difference between a working reranker and BM25 order
            # leaking into the answer. Log loud, fall back deterministically.
            logger.warning(
                "reranker fallback: cross-encoder load/score failed; using retrieval order",
                exc_info=True,
            )
            scores = [c.score for c in cand_list]
        ranked = [RerankedChunk(chunk=c, rerank_score=float(s)) for c, s in zip(cand_list, scores)]
        ranked.sort(key=lambda x: x.rerank_score, reverse=True)
        return ranked[:top_k]

    async def arerank(
        self, query: str, candidates: Iterable[RetrievedChunk], top_k: int = 10
    ) -> list[RerankedChunk]:
        """Async wrapper: off-loads the GPU/CPU-bound scoring to a thread so it
        doesn't block the event loop under concurrent /chat traffic."""
        cand_list = list(candidates)
        return await asyncio.to_thread(self.rerank, query, cand_list, top_k)
