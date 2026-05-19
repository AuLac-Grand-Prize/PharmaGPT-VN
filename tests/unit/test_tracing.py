from __future__ import annotations

import logging

import pytest

from pharmagpt_vn.core.refusal import Classification
from pharmagpt_vn.core.tracing import (
    InMemoryTracer,
    NoopTracer,
    Span,
    StructuredLogTracer,
)
from pharmagpt_vn.models.llm_client import GenerationRequest, GenerationResult
from pharmagpt_vn.rag.reranker import CrossEncoderReranker, RerankedChunk
from pharmagpt_vn.rag.retriever import HybridRetriever, RetrievedChunk
from pharmagpt_vn.services.chat_service import ChatMessage, ChatService


# ---------------------------------------------------------------------------
# Span basics
# ---------------------------------------------------------------------------


def test_span_records_duration_after_end() -> None:
    span = Span(name="x")
    span.end()
    assert span.duration_ms >= 0
    assert span.ended_at is not None


def test_noop_tracer_yields_span_and_ends_it() -> None:
    tracer = NoopTracer()
    with tracer.start_span("step", k=1) as sp:
        sp.set_attribute("hits", 7)
    assert sp.ended_at is not None
    assert sp.attributes["hits"] == 7
    assert sp.attributes["k"] == 1


def test_in_memory_tracer_captures_spans() -> None:
    tracer = InMemoryTracer()
    with tracer.start_span("a"):
        with tracer.start_span("b"):
            pass
    assert [s.name for s in tracer.spans] == ["a", "b"]


def test_structured_log_tracer_emits_one_line_per_span(
    caplog: pytest.LogCaptureFixture,
) -> None:
    tracer = StructuredLogTracer()
    with caplog.at_level(logging.INFO, logger="pharmagpt_vn.trace"):
        with tracer.start_span("step", foo="bar") as sp:
            sp.set_attribute("hits", 3)
    messages = [r.message for r in caplog.records]
    assert any('"span": "step"' in m for m in messages)
    assert any('"hits": 3' in m for m in messages)


def test_structured_log_tracer_marks_status_error_on_exception(
    caplog: pytest.LogCaptureFixture,
) -> None:
    tracer = StructuredLogTracer()
    with caplog.at_level(logging.INFO, logger="pharmagpt_vn.trace"):
        with pytest.raises(RuntimeError):
            with tracer.start_span("boom"):
                raise RuntimeError("nope")
    assert any('"status": "error"' in r.message for r in caplog.records)


def test_structured_log_tracer_coerces_unserializable_attrs(
    caplog: pytest.LogCaptureFixture,
) -> None:
    class _NotJsonable:
        def __repr__(self) -> str:
            return "<weird>"

    tracer = StructuredLogTracer()
    with caplog.at_level(logging.INFO, logger="pharmagpt_vn.trace"):
        with tracer.start_span("s") as sp:
            sp.set_attribute("obj", _NotJsonable())
    assert any("<weird>" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# ChatService integration — spans for classify/retrieve/rerank/generate
# ---------------------------------------------------------------------------


class _StubRetriever(HybridRetriever):
    def __init__(self, chunks: list[RetrievedChunk]) -> None:
        self._chunks = chunks

    async def retrieve(self, query: str, top_k: int = 5, filters=None):  # type: ignore[override]
        return list(self._chunks[:top_k])


class _PassthroughReranker(CrossEncoderReranker):
    def __init__(self) -> None:
        pass

    def rerank(self, query, candidates, top_k: int = 10):  # type: ignore[override]
        return [RerankedChunk(chunk=c, rerank_score=c.score) for c in list(candidates)[:top_k]]


class _Classifier:
    def classify(self, query: str) -> Classification:
        return Classification(label="clinical_safe", confidence=0.9)  # type: ignore[arg-type]


class _LLM:
    async def generate(self, req: GenerationRequest) -> GenerationResult:
        return GenerationResult(text="Metformin giảm liều khi suy thận [REF:1].", prompt_tokens=10, completion_tokens=15)


@pytest.mark.asyncio
async def test_chat_service_emits_expected_spans() -> None:
    tracer = InMemoryTracer()
    svc = ChatService(
        retriever=_StubRetriever(
            [RetrievedChunk(text="Metformin CCĐ eGFR<30.", source="s1", score=0.9, metadata={})]
        ),
        reranker=_PassthroughReranker(),
        refusal_classifier=_Classifier(),
        llm=_LLM(),
        known_drugs=["Metformin"],
        enforce_citations=True,
        tracer=tracer,
    )
    result = await svc.complete(
        [ChatMessage(role="user", content="liều metformin khi suy thận?")]
    )
    assert result.refused is False
    names = [s.name for s in tracer.spans]
    assert "classify" in names
    assert "retrieve" in names
    assert "rerank" in names
    assert "generate" in names
    assert "validate" in names
    # Each span has duration measured.
    for span in tracer.spans:
        assert span.ended_at is not None
        assert span.duration_ms >= 0
