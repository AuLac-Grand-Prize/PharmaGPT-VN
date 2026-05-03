"""ChatService orchestration tests with stubbed LLM / retriever / classifier."""

from __future__ import annotations

import pytest

from pharmagpt_vn.core.refusal import Classification
from pharmagpt_vn.models.llm_client import GenerationRequest, GenerationResult
from pharmagpt_vn.rag.reranker import CrossEncoderReranker, RerankedChunk
from pharmagpt_vn.rag.retriever import HybridRetriever, RetrievedChunk
from pharmagpt_vn.services.chat_service import ChatMessage, ChatService


class _StubRetriever(HybridRetriever):
    def __init__(self, chunks: list[RetrievedChunk]) -> None:
        self._chunks = chunks

    async def retrieve(self, query: str, top_k: int = 5) -> list[RetrievedChunk]:  # type: ignore[override]
        return list(self._chunks[:top_k])


class _PassthroughReranker(CrossEncoderReranker):
    def __init__(self) -> None:
        pass

    def rerank(  # type: ignore[override]
        self, query, candidates, top_k: int = 10
    ) -> list[RerankedChunk]:
        return [RerankedChunk(chunk=c, rerank_score=c.score) for c in list(candidates)[:top_k]]


class _Classifier:
    def __init__(self, label: str = "clinical_safe", confidence: float = 0.9) -> None:
        self._cls = Classification(label=label, confidence=confidence)  # type: ignore[arg-type]

    def classify(self, query: str) -> Classification:
        return self._cls


class _LLM:
    def __init__(self, text: str) -> None:
        self._text = text

    async def generate(self, req: GenerationRequest) -> GenerationResult:
        return GenerationResult(text=self._text, prompt_tokens=10, completion_tokens=20)


def _service(
    *,
    chunks: list[RetrievedChunk] | None = None,
    classifier_label: str = "clinical_safe",
    llm_text: str = "",
    known_drugs: list[str] | None = None,
) -> ChatService:
    return ChatService(
        retriever=_StubRetriever(chunks or []),
        reranker=_PassthroughReranker(),
        refusal_classifier=_Classifier(classifier_label),
        llm=_LLM(llm_text),
        known_drugs=known_drugs or [],
        enforce_citations=True,
    )


@pytest.mark.asyncio
async def test_refuses_out_of_scope_query() -> None:
    svc = _service(classifier_label="out_of_scope")
    result = await svc.complete([ChatMessage(role="user", content="Thời tiết hôm nay?")])
    assert result.refused is True
    assert "không phải" not in result.content.lower() or "dược" in result.content.lower()


@pytest.mark.asyncio
async def test_clinical_query_blocks_when_no_chunks() -> None:
    svc = _service(chunks=[])
    result = await svc.complete(
        [ChatMessage(role="user", content="liều metformin cho người suy thận?")]
    )
    assert result.refused is True
    assert "căn cứ" in result.content


@pytest.mark.asyncio
async def test_clinical_query_succeeds_with_grounded_answer() -> None:
    chunks = [
        RetrievedChunk(
            text="Metformin chống chỉ định khi eGFR < 30.",
            source="VN_pharmacopeia_metformin",
            score=0.9,
            metadata={"parent_path": ("Drug", "Metformin")},
        )
    ]
    answer = "Metformin nên giảm liều khi eGFR 30-45 [REF:1]."
    svc = _service(chunks=chunks, llm_text=answer, known_drugs=["Metformin"])
    result = await svc.complete(
        [ChatMessage(role="user", content="liều metformin cho người suy thận?")]
    )
    assert result.refused is False
    assert "[REF:1]" in result.content
    assert any(c.startswith("[REF:1]") for c in result.citations)


@pytest.mark.asyncio
async def test_safety_block_when_response_lacks_citations() -> None:
    chunks = [
        RetrievedChunk(
            text="Metformin chống chỉ định khi eGFR < 30.",
            source="VN_pharmacopeia",
            score=0.9,
            metadata={},
        )
    ]
    answer = "Cứ dùng metformin liều cao, không cần xét nghiệm gì cả."
    svc = _service(chunks=chunks, llm_text=answer)
    result = await svc.complete(
        [ChatMessage(role="user", content="liều metformin cho người suy thận?")]
    )
    assert result.refused is True
    assert result.finish_reason == "safety_block"


@pytest.mark.asyncio
async def test_pii_redaction_runs_before_retrieval() -> None:
    captured: list[str] = []

    class _Spy(_StubRetriever):
        async def retrieve(self, query: str, top_k: int = 5):
            captured.append(query)
            return await super().retrieve(query, top_k)

    svc = ChatService(
        retriever=_Spy(
            [RetrievedChunk(text="x", source="s", score=1.0, metadata={})]
        ),
        reranker=_PassthroughReranker(),
        refusal_classifier=_Classifier(),
        llm=_LLM("Trả lời an toàn [REF:1]."),
        enforce_citations=True,
    )
    await svc.complete(
        [ChatMessage(role="user", content="metformin SĐT 0912345678 cho bệnh nhân")]
    )
    assert captured
    assert "0912345678" not in captured[0]
    assert "[PHONE]" in captured[0]
