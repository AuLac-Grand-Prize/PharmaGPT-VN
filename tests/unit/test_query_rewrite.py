from __future__ import annotations

import pytest

from pharmagpt_vn.rag.query_rewrite import (
    HeuristicVNRewriter,
    LLMQueryRewriter,
    MultiQueryRetriever,
)
from pharmagpt_vn.rag.retriever import RetrievedChunk


# ---------------------------------------------------------------------------
# HeuristicVNRewriter
# ---------------------------------------------------------------------------


def test_heuristic_swaps_vn_synonym() -> None:
    rw = HeuristicVNRewriter()
    out = rw.rewrite("Tôi bị tiểu đường, dùng thuốc gì?", n=2)
    assert any("đái tháo đường" in r.lower() for r in out)


def test_heuristic_preserves_case_when_possible() -> None:
    rw = HeuristicVNRewriter()
    out = rw.rewrite("CCĐ metformin với suy thận?", n=2)
    # eGFR thấp variant should appear since "suy thận" is in synonyms
    assert any("egfr" in r.lower() for r in out)


def test_heuristic_returns_n_at_most() -> None:
    rw = HeuristicVNRewriter()
    out = rw.rewrite("liều dùng paracetamol", n=2)
    assert len(out) <= 2


def test_heuristic_skips_duplicate_of_original() -> None:
    rw = HeuristicVNRewriter()
    out = rw.rewrite("metformin", n=3)
    assert "metformin" not in [r.lower().strip() for r in out]


def test_heuristic_returns_empty_on_zero() -> None:
    rw = HeuristicVNRewriter()
    out = rw.rewrite("metformin", n=0)
    assert out == []


# ---------------------------------------------------------------------------
# LLMQueryRewriter
# ---------------------------------------------------------------------------


class _FixedLLM:
    def __init__(self, text: str) -> None:
        self._text = text

    def rewrite_text(self, prompt: str) -> str:
        return self._text


def test_llm_rewriter_parses_lines_strips_numbering_and_bullets() -> None:
    llm = _FixedLLM(
        "1. Metformin chống chỉ định ở người suy thận nặng?\n"
        "- CCĐ metformin với eGFR < 30?\n"
        "* khi nào không dùng metformin?\n"
    )
    rw = LLMQueryRewriter(llm)
    out = rw.rewrite("Metformin có dùng được khi suy thận không?", n=3)
    assert len(out) == 3
    assert not out[0].startswith(("1.", "-", "*"))


def test_llm_rewriter_dedupes_against_original() -> None:
    llm = _FixedLLM("metformin liều\nMetformin   liều\nmetformin: liều thường gặp\n")
    rw = LLMQueryRewriter(llm)
    out = rw.rewrite("metformin liều", n=3)
    # First two collapse to the original; only the third survives.
    assert out == ["metformin: liều thường gặp"]


def test_llm_rewriter_handles_llm_exception_gracefully(caplog: pytest.LogCaptureFixture) -> None:
    class _Broken:
        def rewrite_text(self, prompt: str) -> str:
            raise RuntimeError("offline")

    rw = LLMQueryRewriter(_Broken())
    out = rw.rewrite("anything", n=3)
    assert out == []


# ---------------------------------------------------------------------------
# MultiQueryRetriever
# ---------------------------------------------------------------------------


class _RecordingRetriever:
    def __init__(self) -> None:
        self.calls: list[tuple[str, int, dict | None]] = []

    async def retrieve(self, query: str, top_k: int = 5, filters=None):
        self.calls.append((query, top_k, filters))
        return [
            RetrievedChunk(
                text=f"chunk for {query}",
                source=f"src_{query.split()[0]}",
                score=1.0,
                metadata={"q": query},
            )
        ]


class _FixedRewriter:
    def __init__(self, variants: list[str]) -> None:
        self.variants = variants

    def rewrite(self, query: str, n: int = 3) -> list[str]:
        return list(self.variants[:n])


@pytest.mark.asyncio
async def test_multiquery_runs_original_plus_rewrites() -> None:
    inner = _RecordingRetriever()
    rw = _FixedRewriter(["q1", "q2"])
    mqr = MultiQueryRetriever(inner=inner, rewriter=rw, n_rewrites=2)  # type: ignore[arg-type]
    await mqr.retrieve("original", top_k=2)
    queries = [c[0] for c in inner.calls]
    assert queries == ["original", "q1", "q2"]


@pytest.mark.asyncio
async def test_multiquery_passes_filters_to_inner() -> None:
    inner = _RecordingRetriever()
    rw = _FixedRewriter([])
    mqr = MultiQueryRetriever(inner=inner, rewriter=rw)  # type: ignore[arg-type]
    await mqr.retrieve("q", top_k=3, filters={"drug": "Metformin"})
    assert all(c[2] == {"drug": "Metformin"} for c in inner.calls)


@pytest.mark.asyncio
async def test_multiquery_fuses_and_returns_top_k() -> None:
    inner = _RecordingRetriever()
    rw = _FixedRewriter(["q1", "q2"])
    mqr = MultiQueryRetriever(inner=inner, rewriter=rw, n_rewrites=2)  # type: ignore[arg-type]
    result = await mqr.retrieve("original", top_k=2)
    assert len(result) <= 2
