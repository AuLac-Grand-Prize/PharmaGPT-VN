"""Query rewriting + multi-query retrieval.

A single user phrasing rarely covers all the lexical variants present in the
corpus (e.g. "tiểu đường" vs "đái tháo đường", "suy thận" vs "eGFR thấp"). We
generate N rewrites of the query, retrieve for each variant concurrently, then
RRF-merge — a documented +15-30% recall lift in the literature.

Two rewriters here:
  - HeuristicVNRewriter — zero-LLM-cost synonym expansion for common VN pharma
    terms. Useful as a baseline and as a fallback when the LLM is offline.
  - LLMQueryRewriter   — wraps an injected LLM that returns N alternative phrasings.

MultiQueryRetriever orchestrates: rewrite → retrieve_each → fuse.
"""

from __future__ import annotations

import asyncio
import logging
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Protocol

from pharmagpt_vn.rag.retriever import (
    HybridRetriever,
    RetrievedChunk,
    reciprocal_rank_fusion,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Protocols
# ---------------------------------------------------------------------------


class QueryRewriter(Protocol):
    def rewrite(self, query: str, n: int = 3) -> list[str]: ...


class RewriterLLM(Protocol):
    """Plain-text rewriter LLM. Returns one rewrite per call.

    Production wires a small local model (Qwen 7B) via vLLM. The Protocol keeps
    this module independent of llm_client.
    """

    def rewrite_text(self, prompt: str) -> str: ...


# ---------------------------------------------------------------------------
# Heuristic VN pharma synonym rewriter (zero-cost baseline)
# ---------------------------------------------------------------------------

# Bidirectional pairs: ("a", "b") means we'll also produce a variant with a↔b.
_SYNONYMS_VN: tuple[tuple[str, str], ...] = (
    ("tiểu đường", "đái tháo đường"),
    ("đái tháo đường", "tiểu đường"),
    ("suy thận", "eGFR thấp"),
    ("eGFR thấp", "suy thận"),
    ("liều dùng", "liều"),
    ("liều dùng", "cách dùng"),
    ("tác dụng phụ", "tác dụng không mong muốn"),
    ("tác dụng phụ", "ADR"),
    ("chống chỉ định", "CCĐ"),
    ("tương tác thuốc", "drug interaction"),
    ("trẻ em", "nhi khoa"),
    ("người già", "người cao tuổi"),
    ("phụ nữ có thai", "thai kỳ"),
    ("phụ nữ mang thai", "thai kỳ"),
    ("quá liều", "ngộ độc"),
)

_FRAME_PREFIXES_VN: tuple[str, ...] = (
    "Hãy tra cứu thông tin về ",
    "Thuốc nào ",
    "Thông tin dược lý: ",
)


class HeuristicVNRewriter:
    """Generate query variants by substituting common VN pharma synonyms.

    Cheap and offline-safe. Never returns the original verbatim — `rewrite`'s
    contract is to return *alternative* phrasings; callers that want the original
    should run retrieve once on it before fusing with the rewrites.
    """

    def __init__(
        self,
        synonyms: Sequence[tuple[str, str]] = _SYNONYMS_VN,
        frame_prefixes: Sequence[str] = _FRAME_PREFIXES_VN,
    ) -> None:
        self._synonyms = tuple(synonyms)
        self._frames = tuple(frame_prefixes)

    def rewrite(self, query: str, n: int = 3) -> list[str]:
        if n <= 0:
            return []
        out: list[str] = []
        seen: set[str] = {_normalize(query)}
        # First: synonym swaps (most useful).
        for a, b in self._synonyms:
            if a in query.lower():
                # Preserve case where possible — match the original casing of the span.
                cand = _replace_case_insensitive(query, a, b)
                key = _normalize(cand)
                if key not in seen:
                    seen.add(key)
                    out.append(cand)
                    if len(out) >= n:
                        return out
        # Then: frame-prefix variants — light HyDE substitute when no synonym hit.
        for prefix in self._frames:
            cand = f"{prefix}{query}".strip()
            key = _normalize(cand)
            if key not in seen:
                seen.add(key)
                out.append(cand)
                if len(out) >= n:
                    return out
        return out


# ---------------------------------------------------------------------------
# LLM-based rewriter
# ---------------------------------------------------------------------------


REWRITE_PROMPT_TEMPLATE = (
    "Bạn là dược sĩ lâm sàng. Viết {n} cách diễn đạt KHÁC của câu hỏi sau, "
    "giữ NGUYÊN ý nhưng dùng thuật ngữ đồng nghĩa (tên thuốc, ICD-10, "
    "thuật ngữ Bộ Y tế). Mỗi cách trên 1 dòng, KHÔNG đánh số.\n\n"
    "Câu hỏi: {query}\n\n"
    "Diễn đạt khác:"
)


class LLMQueryRewriter:
    def __init__(self, llm: RewriterLLM, max_per_call: int = 4) -> None:
        self._llm = llm
        self._max = max_per_call

    def rewrite(self, query: str, n: int = 3) -> list[str]:
        if n <= 0:
            return []
        n = min(n, self._max)
        prompt = REWRITE_PROMPT_TEMPLATE.format(n=n, query=query)
        try:
            raw = self._llm.rewrite_text(prompt)
        except Exception:
            logger.warning("rewriter LLM failed; returning empty rewrites", exc_info=True)
            return []
        return _parse_rewrites(raw, n=n, original=query)


def _parse_rewrites(raw: str, n: int, original: str) -> list[str]:
    out: list[str] = []
    seen: set[str] = {_normalize(original)}
    for line in raw.splitlines():
        # Strip leading bullets / numbering the LLM may include despite instructions.
        cleaned = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", line).strip().strip('"').strip()
        if not cleaned:
            continue
        key = _normalize(cleaned)
        if key in seen:
            continue
        seen.add(key)
        out.append(cleaned)
        if len(out) >= n:
            break
    return out


# ---------------------------------------------------------------------------
# Multi-query retriever
# ---------------------------------------------------------------------------


@dataclass
class MultiQueryRetriever:
    """Run retrieval for the original query + N rewrites, fuse via RRF.

    Conforms to the same async retrieve(query, top_k, filters=...) signature as
    HybridRetriever — so `CachedRetriever`, ChatService, etc. can wrap it
    transparently.
    """

    inner: HybridRetriever
    rewriter: QueryRewriter
    n_rewrites: int = 3
    per_query_pool_multiplier: int = 2
    rrf_k: int = 60
    extra_chunks_for_fusion: list[RetrievedChunk] = field(default_factory=list)

    async def retrieve(
        self,
        query: str,
        top_k: int = 5,
        filters: dict[str, object] | None = None,
    ) -> list[RetrievedChunk]:
        queries = [query]
        rewrites = self.rewriter.rewrite(query, n=self.n_rewrites)
        queries.extend(rewrites)

        # Run all variants concurrently. Each pulls a slightly larger pool so
        # RRF has overlap to work with.
        pool = max(top_k * self.per_query_pool_multiplier, top_k)
        tasks = [
            asyncio.create_task(self.inner.retrieve(q, top_k=pool, filters=filters))
            for q in queries
        ]
        per_query_results = await asyncio.gather(*tasks)

        fused = reciprocal_rank_fusion(*per_query_results, k=self.rrf_k)
        return fused[:top_k]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _normalize(text: str) -> str:
    return " ".join(text.lower().split())


def _replace_case_insensitive(s: str, old: str, new: str) -> str:
    pattern = re.compile(re.escape(old), flags=re.IGNORECASE)
    return pattern.sub(new, s, count=1)


__all__ = [
    "HeuristicVNRewriter",
    "LLMQueryRewriter",
    "MultiQueryRetriever",
    "QueryRewriter",
    "RewriterLLM",
]
