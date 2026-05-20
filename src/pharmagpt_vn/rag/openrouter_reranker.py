"""OpenRouter (Cohere) reranker — replaces local cross-encoder.

Calls `POST {base_url}/rerank` with the Cohere-compatible schema OpenRouter
exposes. Returns `RerankedChunk` with the same shape as `CrossEncoderReranker`
so the orchestrator wiring stays identical.

Fallback: if the API call fails (network, 5xx, parse error) we keep the
retrieval order and log loud — production must notice when reranking is
silently bypassed.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable

import httpx

from pharmagpt_vn.rag.reranker import RerankedChunk
from pharmagpt_vn.rag.retriever import RetrievedChunk

logger = logging.getLogger(__name__)


class OpenRouterReranker:
    def __init__(
        self,
        api_key: str,
        model: str = "cohere/rerank-v3.5",
        base_url: str = "https://openrouter.ai/api/v1",
        *,
        timeout: float = 30.0,
        async_client: httpx.AsyncClient | None = None,
        sync_client: httpx.Client | None = None,
    ) -> None:
        self._api_key = api_key
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._async_client = async_client
        self._sync_client = sync_client

    # ------------------------------------------------------------------
    # Sync
    # ------------------------------------------------------------------

    def rerank(
        self, query: str, candidates: Iterable[RetrievedChunk], top_k: int = 10
    ) -> list[RerankedChunk]:
        cand_list = list(candidates)
        if not cand_list:
            return []
        payload, headers, url = self._build(query, cand_list, top_k)
        try:
            if self._sync_client is not None:
                resp = self._sync_client.post(url, headers=headers, json=payload)
            else:
                with httpx.Client(timeout=self._timeout) as client:
                    resp = client.post(url, headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()
        except Exception:
            return self._fallback(cand_list, top_k)
        return self._parse(data, cand_list, top_k)

    # ------------------------------------------------------------------
    # Async
    # ------------------------------------------------------------------

    async def arerank(
        self, query: str, candidates: Iterable[RetrievedChunk], top_k: int = 10
    ) -> list[RerankedChunk]:
        cand_list = list(candidates)
        if not cand_list:
            return []
        payload, headers, url = self._build(query, cand_list, top_k)
        try:
            if self._async_client is not None:
                resp = await self._async_client.post(url, headers=headers, json=payload)
            else:
                async with httpx.AsyncClient(timeout=self._timeout) as client:
                    resp = await client.post(url, headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()
        except Exception:
            return self._fallback(cand_list, top_k)
        return self._parse(data, cand_list, top_k)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build(
        self, query: str, cand_list: list[RetrievedChunk], top_k: int
    ) -> tuple[dict, dict, str]:
        payload: dict = {
            "model": self._model,
            "query": query,
            "documents": [c.text for c in cand_list],
            "top_n": top_k,
        }
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        return payload, headers, f"{self._base_url}/rerank"

    def _parse(
        self, data: dict, cand_list: list[RetrievedChunk], top_k: int
    ) -> list[RerankedChunk]:
        results = data.get("results") or []
        ranked: list[RerankedChunk] = []
        for r in results:
            try:
                idx = int(r["index"])
                score = float(r["relevance_score"])
            except (KeyError, TypeError, ValueError):
                continue
            if 0 <= idx < len(cand_list):
                ranked.append(RerankedChunk(chunk=cand_list[idx], rerank_score=score))
        if not ranked:
            return self._fallback(cand_list, top_k)
        ranked.sort(key=lambda x: x.rerank_score, reverse=True)
        return ranked[:top_k]

    @staticmethod
    def _fallback(cand_list: list[RetrievedChunk], top_k: int) -> list[RerankedChunk]:
        logger.warning(
            "openrouter reranker fallback: API failed or returned no results; "
            "using retrieval order",
            exc_info=True,
        )
        return [RerankedChunk(chunk=c, rerank_score=c.score) for c in cand_list[:top_k]]
