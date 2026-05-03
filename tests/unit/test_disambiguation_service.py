from __future__ import annotations

import json

import pytest

from pharmagpt_vn.contracts.disambiguation import (
    Candidate,
    DisambiguationRequest,
    PatientContext,
    PrescriptionContext,
)
from pharmagpt_vn.models.llm_client import GenerationRequest, GenerationResult
from pharmagpt_vn.rag.retriever import HybridRetriever, RetrievedChunk
from pharmagpt_vn.services.disambiguation_service import DisambiguationService


class _StubRetriever(HybridRetriever):
    def __init__(self, by_query: dict[str, list[RetrievedChunk]]) -> None:
        self._by_query = by_query

    async def retrieve(self, query: str, top_k: int = 5) -> list[RetrievedChunk]:  # type: ignore[override]
        for key, chunks in self._by_query.items():
            if key.lower() in query.lower():
                return list(chunks[:top_k])
        return []


class _RecordingLLM:
    def __init__(self, response_text: str) -> None:
        self.last_prompt: str | None = None
        self._text = response_text

    async def generate(self, req: GenerationRequest) -> GenerationResult:
        self.last_prompt = req.prompt
        return GenerationResult(text=self._text)


def _request() -> DisambiguationRequest:
    return DisambiguationRequest(
        candidates=[
            Candidate(name="Metformin", strength="500mg", confidence=0.45),
            Candidate(name="Metronidazole", strength="500mg", confidence=0.40),
        ],
        context=PrescriptionContext(
            diagnosis="Đái tháo đường típ 2",
            patient=PatientContext(age=58, sex="F", comorbidities=["HTN"]),
            other_drugs_in_prescription=["Amlodipine 5mg"],
        ),
    )


@pytest.mark.asyncio
async def test_fallback_when_no_llm() -> None:
    retriever = _StubRetriever({"Metformin": []})
    svc = DisambiguationService(retriever)
    resp = await svc.rank(_request())
    assert resp.top_candidates[0].name == "Metformin"  # higher vision conf
    assert "fallback" in resp.top_candidates[0].reasoning.lower()


@pytest.mark.asyncio
async def test_llm_grounded_response_parsed() -> None:
    chunk = RetrievedChunk(
        text="Metformin first-line cho ĐTĐ típ 2 khi eGFR ≥ 45.",
        source="VN_metformin",
        score=0.9,
        metadata={},
    )
    retriever = _StubRetriever({"Metformin": [chunk]})
    llm_text = json.dumps(
        {
            "top_candidates": [
                {
                    "name": "Metformin",
                    "strength": "500mg",
                    "confidence": 0.97,
                    "reasoning": "Phù hợp ĐTĐ típ 2 [REF:1].",
                    "citations": ["REF:1"],
                }
            ],
            "latency_ms": 0,
        }
    )
    llm = _RecordingLLM(llm_text)
    svc = DisambiguationService(retriever, llm=llm)
    resp = await svc.rank(_request())
    assert resp.top_candidates[0].confidence == 0.97
    assert "REF:1" in resp.top_candidates[0].citations
    assert "Đái tháo đường" in (llm.last_prompt or "")


@pytest.mark.asyncio
async def test_unparseable_llm_response_falls_back() -> None:
    retriever = _StubRetriever({"Metformin": []})
    llm = _RecordingLLM("not even close to JSON")
    svc = DisambiguationService(retriever, llm=llm)
    resp = await svc.rank(_request())
    assert "fallback" in resp.top_candidates[0].reasoning.lower()


@pytest.mark.asyncio
async def test_monograph_collection_dedups() -> None:
    chunk = RetrievedChunk(text="x", source="src", score=1.0, metadata={})
    # Both queries hit the same chunk; collected list should not duplicate.
    retriever = _StubRetriever({"Metformin": [chunk], "Metronidazole": [chunk]})
    svc = DisambiguationService(retriever)
    monographs = await svc._collect_monographs(  # type: ignore[attr-defined]
        _request().candidates, _request().context
    )
    assert len(monographs) == 1
