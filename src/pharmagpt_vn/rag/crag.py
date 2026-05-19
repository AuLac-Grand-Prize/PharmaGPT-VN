"""Corrective RAG (CRAG) validation loop — Yan et al. 2024.

Purpose: after retrieve+rerank, grade whether the top context is *sufficient* to
answer the clinical query. Three outcomes:

  - sufficient   → proceed to generation
  - ambiguous    → proceed but mark trace; downstream may prefer conservative wording
  - insufficient → refuse and route to a human pharmacist (PharmaGPT KPI: 100%
                   clinical citations, hallucination ≤ 3%)

This is the lever that lets ChatService gracefully degrade instead of hallucinating
when retrieval fails — critical for the pharma domain.

Two graders supported:
  - `LLMRelevanceGrader` — small local model (Qwen 7B / Llama 8B) prompted to
    return a single label. Cost: 1 cheap call per query.
  - `HeuristicGrader`   — rerank-score thresholds. Zero extra cost; useful as a
    safety net or when the grader LLM is offline.

Production wires both: heuristic first (cheap reject), then LLM if borderline.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, Protocol

from pharmagpt_vn.rag.reranker import RerankedChunk

CRAGLabel = Literal["sufficient", "ambiguous", "insufficient"]

GRADER_PROMPT_TEMPLATE = (
    "Bạn là dược sĩ lâm sàng đang đánh giá xem các trích đoạn dưới đây có ĐỦ "
    "căn cứ để trả lời an toàn câu hỏi hay không. Trả lời DUY NHẤT một trong "
    "ba nhãn: SUFFICIENT, AMBIGUOUS, INSUFFICIENT.\n\n"
    "Tiêu chí:\n"
    "- SUFFICIENT: trích đoạn nói trực tiếp về thuốc/tình huống trong câu hỏi "
    "và bao phủ thông tin cần thiết.\n"
    "- AMBIGUOUS: có liên quan nhưng thiếu chi tiết quan trọng (liều, đối tượng, "
    "chống chỉ định).\n"
    "- INSUFFICIENT: không liên quan hoặc trái ngược.\n\n"
    "Câu hỏi: {query}\n\n"
    "Trích đoạn:\n{snippets}\n\n"
    "Nhãn:"
)


@dataclass(frozen=True)
class CRAGResult:
    label: CRAGLabel
    confidence: float
    reason: str = ""

    @property
    def is_sufficient(self) -> bool:
        return self.label == "sufficient"

    @property
    def should_refuse(self) -> bool:
        return self.label == "insufficient"


class RelevanceGrader(Protocol):
    def grade(self, query: str, candidates: Sequence[RerankedChunk]) -> CRAGResult: ...


class HeuristicGrader:
    """Grade purely from rerank scores. Cheap fallback / safety net.

    Defaults tuned for BGE-reranker-v2-m3 normalized scores (0..1):
      - top score ≥ 0.65  → sufficient
      - top score ≥ 0.35  → ambiguous
      - otherwise         → insufficient
    """

    def __init__(
        self,
        sufficient_threshold: float = 0.65,
        ambiguous_threshold: float = 0.35,
    ) -> None:
        if not 0.0 <= ambiguous_threshold <= sufficient_threshold <= 1.0:
            raise ValueError("require 0 ≤ ambiguous ≤ sufficient ≤ 1")
        self._sufficient = sufficient_threshold
        self._ambiguous = ambiguous_threshold

    def grade(self, query: str, candidates: Sequence[RerankedChunk]) -> CRAGResult:
        if not candidates:
            return CRAGResult(label="insufficient", confidence=1.0, reason="empty candidates")
        top = candidates[0].rerank_score
        if top >= self._sufficient:
            return CRAGResult(label="sufficient", confidence=top, reason=f"top={top:.2f}")
        if top >= self._ambiguous:
            return CRAGResult(label="ambiguous", confidence=top, reason=f"top={top:.2f}")
        return CRAGResult(label="insufficient", confidence=1.0 - top, reason=f"top={top:.2f}")


class LLMRelevanceGrader:
    """LLM-based grader. Production wires a small local model (Qwen 7B)."""

    def __init__(self, grader_llm: "GraderLLM", max_snippet_chars: int = 600) -> None:
        self._llm = grader_llm
        self._max_snippet_chars = max_snippet_chars

    def grade(self, query: str, candidates: Sequence[RerankedChunk]) -> CRAGResult:
        if not candidates:
            return CRAGResult(label="insufficient", confidence=1.0, reason="empty candidates")
        snippets = _format_snippets(candidates, self._max_snippet_chars)
        prompt = GRADER_PROMPT_TEMPLATE.format(query=query, snippets=snippets)
        raw = self._llm.grade_text(prompt).strip().upper()
        label = _parse_label(raw)
        return CRAGResult(label=label, confidence=0.0 if label == "ambiguous" else 1.0, reason=raw[:80])


class GraderLLM(Protocol):
    """Plain text-in / text-out interface — keeps this module independent of llm_client."""

    def grade_text(self, prompt: str) -> str: ...


class TieredGrader:
    """Heuristic first (fast); LLM only when heuristic is ambiguous.

    Matches production cost target: ~0 extra latency on confident cases,
    one cheap grader call on borderline.
    """

    def __init__(self, heuristic: HeuristicGrader, llm_grader: LLMRelevanceGrader) -> None:
        self._heuristic = heuristic
        self._llm_grader = llm_grader

    def grade(self, query: str, candidates: Sequence[RerankedChunk]) -> CRAGResult:
        first = self._heuristic.grade(query, candidates)
        if first.label != "ambiguous":
            return first
        return self._llm_grader.grade(query, candidates)


def _format_snippets(candidates: Sequence[RerankedChunk], max_chars: int) -> str:
    parts = []
    for i, rc in enumerate(candidates, start=1):
        text = rc.chunk.text[:max_chars]
        parts.append(f"[{i}] ({rc.chunk.source}) {text}")
    return "\n\n".join(parts)


def _parse_label(raw: str) -> CRAGLabel:
    if "INSUFFICIENT" in raw:
        return "insufficient"
    if "SUFFICIENT" in raw:
        return "sufficient"
    return "ambiguous"
