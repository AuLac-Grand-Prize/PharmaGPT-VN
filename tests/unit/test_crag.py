"""Tests for CRAG validation graders."""

from __future__ import annotations

import pytest

from pharmagpt_vn.rag.crag import (
    HeuristicGrader,
    LLMRelevanceGrader,
    TieredGrader,
)
from pharmagpt_vn.rag.reranker import RerankedChunk
from pharmagpt_vn.rag.retriever import RetrievedChunk


def _rc(score: float, text: str = "x", source: str = "s") -> RerankedChunk:
    return RerankedChunk(
        chunk=RetrievedChunk(text=text, source=source, score=score, metadata={}),
        rerank_score=score,
    )


class TestHeuristicGrader:
    def test_empty_candidates_are_insufficient(self) -> None:
        grader = HeuristicGrader()
        result = grader.grade("q", [])
        assert result.label == "insufficient"
        assert result.should_refuse is True

    def test_high_score_is_sufficient(self) -> None:
        grader = HeuristicGrader(sufficient_threshold=0.65, ambiguous_threshold=0.35)
        result = grader.grade("q", [_rc(0.9), _rc(0.5)])
        assert result.label == "sufficient"
        assert result.is_sufficient is True
        assert result.confidence == pytest.approx(0.9)

    def test_borderline_score_is_ambiguous(self) -> None:
        grader = HeuristicGrader(sufficient_threshold=0.65, ambiguous_threshold=0.35)
        result = grader.grade("q", [_rc(0.5)])
        assert result.label == "ambiguous"
        assert not result.is_sufficient
        assert not result.should_refuse

    def test_low_score_is_insufficient(self) -> None:
        grader = HeuristicGrader(sufficient_threshold=0.65, ambiguous_threshold=0.35)
        result = grader.grade("q", [_rc(0.1)])
        assert result.label == "insufficient"
        assert result.should_refuse is True

    def test_invalid_thresholds_raise(self) -> None:
        with pytest.raises(ValueError):
            HeuristicGrader(sufficient_threshold=0.3, ambiguous_threshold=0.5)
        with pytest.raises(ValueError):
            HeuristicGrader(sufficient_threshold=1.2)


class _StubGraderLLM:
    def __init__(self, response: str) -> None:
        self.response = response
        self.last_prompt: str | None = None

    def grade_text(self, prompt: str) -> str:
        self.last_prompt = prompt
        return self.response


class TestLLMRelevanceGrader:
    def test_parses_sufficient_label(self) -> None:
        grader = LLMRelevanceGrader(_StubGraderLLM("SUFFICIENT"))
        result = grader.grade("q", [_rc(0.4, text="some context")])
        assert result.label == "sufficient"

    def test_parses_insufficient_label(self) -> None:
        grader = LLMRelevanceGrader(_StubGraderLLM("INSUFFICIENT - không liên quan"))
        result = grader.grade("q", [_rc(0.4)])
        assert result.label == "insufficient"
        assert result.should_refuse

    def test_unparseable_response_defaults_to_ambiguous(self) -> None:
        grader = LLMRelevanceGrader(_StubGraderLLM("không rõ"))
        result = grader.grade("q", [_rc(0.4)])
        assert result.label == "ambiguous"

    def test_empty_candidates_skip_llm_call(self) -> None:
        stub = _StubGraderLLM("SUFFICIENT")
        grader = LLMRelevanceGrader(stub)
        result = grader.grade("q", [])
        assert result.label == "insufficient"
        assert stub.last_prompt is None

    def test_prompt_includes_query_and_snippets(self) -> None:
        stub = _StubGraderLLM("SUFFICIENT")
        grader = LLMRelevanceGrader(stub)
        grader.grade("liều metformin?", [_rc(0.5, text="Metformin 500mg", source="src1")])
        assert stub.last_prompt is not None
        assert "liều metformin?" in stub.last_prompt
        assert "Metformin 500mg" in stub.last_prompt
        assert "src1" in stub.last_prompt


class TestTieredGrader:
    def test_skips_llm_when_heuristic_is_confident(self) -> None:
        stub = _StubGraderLLM("SUFFICIENT")
        tiered = TieredGrader(HeuristicGrader(), LLMRelevanceGrader(stub))

        result = tiered.grade("q", [_rc(0.95)])
        assert result.label == "sufficient"
        assert stub.last_prompt is None  # LLM not invoked

    def test_calls_llm_when_heuristic_is_ambiguous(self) -> None:
        stub = _StubGraderLLM("INSUFFICIENT")
        tiered = TieredGrader(HeuristicGrader(), LLMRelevanceGrader(stub))

        result = tiered.grade("q", [_rc(0.5)])  # ambiguous band
        assert stub.last_prompt is not None
        assert result.label == "insufficient"

    def test_skips_llm_when_heuristic_says_insufficient(self) -> None:
        stub = _StubGraderLLM("SUFFICIENT")
        tiered = TieredGrader(HeuristicGrader(), LLMRelevanceGrader(stub))
        result = tiered.grade("q", [_rc(0.05)])
        assert result.label == "insufficient"
        assert stub.last_prompt is None
