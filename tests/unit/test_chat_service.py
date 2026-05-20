"""ChatService orchestration tests with stubbed LLM / retriever / classifier."""

from __future__ import annotations

import pytest

from pharmagpt_vn.core.refusal import Classification
from pharmagpt_vn.models.llm_client import GenerationRequest, GenerationResult
from pharmagpt_vn.rag.crag import CRAGResult, RelevanceGrader
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
    crag_grader: RelevanceGrader | None = None,
) -> ChatService:
    return ChatService(
        retriever=_StubRetriever(chunks or []),
        reranker=_PassthroughReranker(),
        refusal_classifier=_Classifier(classifier_label),
        llm=_LLM(llm_text),
        known_drugs=known_drugs or [],
        enforce_citations=True,
        crag_grader=crag_grader,
    )


class _FixedGrader:
    """Returns a preset CRAG result regardless of input."""

    def __init__(self, label, confidence: float = 1.0) -> None:
        self._result = CRAGResult(label=label, confidence=confidence)

    def grade(self, query, candidates) -> CRAGResult:
        return self._result


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
        [ChatMessage(role="user", content="metformin SĐT 0912345678 hoặc +84987654321 cho bệnh nhân")]
    )
    assert captured
    assert "0912345678" not in captured[0]
    assert "+84987654321" not in captured[0]
    assert captured[0].count("[PHONE]") == 2


@pytest.mark.asyncio
async def test_crag_insufficient_refuses_before_generation() -> None:
    chunks = [
        RetrievedChunk(text="Đoạn không liên quan.", source="src", score=0.2, metadata={})
    ]
    svc = _service(
        chunks=chunks,
        llm_text="Không bao giờ được generate.",
        crag_grader=_FixedGrader("insufficient"),
    )
    result = await svc.complete(
        [ChatMessage(role="user", content="liều metformin cho người suy thận?")]
    )
    assert result.refused is True
    assert result.finish_reason == "crag_insufficient"
    assert result.trace is not None
    assert result.trace.crag is not None
    assert result.trace.crag.label == "insufficient"


@pytest.mark.asyncio
async def test_crag_sufficient_proceeds_to_generation() -> None:
    chunks = [
        RetrievedChunk(
            text="Metformin chống chỉ định khi eGFR < 30.",
            source="VN_pharmacopeia",
            score=0.9,
            metadata={},
        )
    ]
    svc = _service(
        chunks=chunks,
        llm_text="Metformin nên giảm liều [REF:1].",
        known_drugs=["Metformin"],
        crag_grader=_FixedGrader("sufficient"),
    )
    result = await svc.complete(
        [ChatMessage(role="user", content="liều metformin cho người suy thận?")]
    )
    assert result.refused is False
    assert result.trace is not None and result.trace.crag is not None
    assert result.trace.crag.label == "sufficient"


class _SequencedRetriever(HybridRetriever):
    """Returns a different chunk-list per call, in order."""

    def __init__(self, sequences: list[list[RetrievedChunk]]) -> None:
        self._sequences = sequences
        self._i = 0
        self.queries: list[str] = []

    async def retrieve(self, query: str, top_k: int = 5, filters=None):  # type: ignore[override]
        self.queries.append(query)
        if self._i >= len(self._sequences):
            return []
        chunks = list(self._sequences[self._i][:top_k])
        self._i += 1
        return chunks


class _SequencedGrader:
    def __init__(self, labels: list[str]) -> None:
        self._labels = labels
        self._i = 0

    def grade(self, query, candidates) -> CRAGResult:
        label = self._labels[min(self._i, len(self._labels) - 1)]
        self._i += 1
        return CRAGResult(label=label, confidence=1.0)  # type: ignore[arg-type]


class _FixedRewriter:
    def __init__(self, rewrites: list[str]) -> None:
        self._r = rewrites
        self.calls: int = 0

    def rewrite(self, query: str, n: int = 1) -> list[str]:
        self.calls += 1
        return list(self._r[:n])


@pytest.mark.asyncio
async def test_crag_retry_rewrites_query_when_initial_grade_insufficient() -> None:
    weak = [RetrievedChunk(text="weak", source="s1", score=0.1, metadata={})]
    strong = [
        RetrievedChunk(
            text="Metformin chống chỉ định khi eGFR < 30.",
            source="VN_pharmacopeia",
            score=0.9,
            metadata={},
        )
    ]
    retriever = _SequencedRetriever([weak, strong])
    grader = _SequencedGrader(["insufficient", "sufficient"])
    rewriter = _FixedRewriter(["metformin trong suy thận eGFR thấp"])
    svc = ChatService(
        retriever=retriever,
        reranker=_PassthroughReranker(),
        refusal_classifier=_Classifier("clinical_safe"),
        llm=_LLM("Metformin giảm liều khi suy thận [REF:1]."),
        known_drugs=["Metformin"],
        enforce_citations=True,
        crag_grader=grader,
        query_rewriter=rewriter,
        crag_max_retries=1,
    )
    result = await svc.complete(
        [ChatMessage(role="user", content="liều metformin cho người suy thận?")]
    )
    assert result.refused is False
    assert rewriter.calls == 1
    assert result.trace is not None
    assert result.trace.retries_used == 1
    assert result.trace.rewritten_query == "metformin trong suy thận eGFR thấp"
    assert result.trace.crag is not None
    assert result.trace.crag.label == "sufficient"


@pytest.mark.asyncio
async def test_crag_retry_exhausts_then_refuses() -> None:
    weak1 = [RetrievedChunk(text="w1", source="s1", score=0.1, metadata={})]
    weak2 = [RetrievedChunk(text="w2", source="s2", score=0.1, metadata={})]
    retriever = _SequencedRetriever([weak1, weak2])
    grader = _SequencedGrader(["insufficient", "insufficient"])
    rewriter = _FixedRewriter(["alt phrasing"])
    svc = ChatService(
        retriever=retriever,
        reranker=_PassthroughReranker(),
        refusal_classifier=_Classifier("clinical_safe"),
        llm=_LLM("never reached"),
        enforce_citations=True,
        crag_grader=grader,
        query_rewriter=rewriter,
        crag_max_retries=1,
    )
    result = await svc.complete(
        [ChatMessage(role="user", content="liều metformin cho người suy thận?")]
    )
    assert result.refused is True
    assert result.finish_reason == "crag_insufficient"
    assert result.trace is not None and result.trace.retries_used == 1


@pytest.mark.asyncio
async def test_crag_no_retry_when_rewriter_missing() -> None:
    weak = [RetrievedChunk(text="w", source="s", score=0.1, metadata={})]
    retriever = _SequencedRetriever([weak])
    grader = _SequencedGrader(["insufficient"])
    svc = ChatService(
        retriever=retriever,
        reranker=_PassthroughReranker(),
        refusal_classifier=_Classifier("clinical_safe"),
        llm=_LLM("never reached"),
        enforce_citations=True,
        crag_grader=grader,
        query_rewriter=None,
        crag_max_retries=3,
    )
    result = await svc.complete(
        [ChatMessage(role="user", content="liều metformin cho người suy thận?")]
    )
    assert result.refused is True
    assert result.trace is not None and result.trace.retries_used == 0


class _RecordingRetriever(HybridRetriever):
    """Records every query passed in and returns a deterministic chunk per query."""

    def __init__(self) -> None:
        self.queries: list[str] = []

    async def retrieve(self, query: str, top_k: int = 5, filters=None):  # type: ignore[override]
        self.queries.append(query)
        # Unique chunk per query so RRF fusion has multiple distinct sources to fuse.
        return [
            RetrievedChunk(
                text=f"chunk for: {query}",
                source=f"src::{abs(hash(query)) % 10000}",
                score=0.9,
                metadata={},
            )
        ]


class _StubDecomposer:
    def __init__(self, sub_queries: list[str]) -> None:
        self._sub = sub_queries

    def decompose(self, query: str) -> list[str]:
        return list(self._sub)


class _StubHyDE:
    def __init__(self, doc: str = "hypothetical doc body") -> None:
        self._doc = doc
        self.called = 0

    async def generate(self, query: str) -> str:
        self.called += 1
        return self._doc


@pytest.mark.asyncio
async def test_3branch_runs_all_branches_for_clinical_query() -> None:
    retriever = _RecordingRetriever()
    decomposer = _StubDecomposer(["sub query 1", "sub query 2"])
    hyde = _StubHyDE(doc="hyde fake answer about metformin")

    svc = ChatService(
        retriever=retriever,
        reranker=_PassthroughReranker(),
        refusal_classifier=_Classifier("clinical_safe"),
        llm=_LLM("Metformin giảm liều [REF:1]."),
        known_drugs=["Metformin"],
        enforce_citations=True,
        decomposer=decomposer,
        hyde=hyde,
        per_branch_top_k=10,
    )
    result = await svc.complete(
        [ChatMessage(role="user", content="liều metformin cho người suy thận?")]
    )
    assert result.refused is False
    # Branch A original query + Branch B 2 sub_queries + Branch C 1 hyde doc = 4 retrieves
    assert len(retriever.queries) == 4
    assert "liều metformin cho người suy thận?" in retriever.queries
    assert "sub query 1" in retriever.queries
    assert "sub query 2" in retriever.queries
    assert "hyde fake answer about metformin" in retriever.queries
    assert hyde.called == 1


@pytest.mark.asyncio
async def test_3branch_skips_hyde_for_non_clinical_query() -> None:
    retriever = _RecordingRetriever()
    decomposer = _StubDecomposer(["sub_a"])
    hyde = _StubHyDE()

    svc = ChatService(
        retriever=retriever,
        reranker=_PassthroughReranker(),
        refusal_classifier=_Classifier("clinical_safe"),
        llm=_LLM("Xin chào [REF:1]."),
        enforce_citations=False,
        decomposer=decomposer,
        hyde=hyde,
    )
    # "Xin chào" has no clinical keywords → is_clinical=False → Branch C skip
    await svc.complete([ChatMessage(role="user", content="Xin chào")])
    assert hyde.called == 0
    # Branch A (original) + Branch B (1 sub) = 2 retrieves
    assert len(retriever.queries) == 2


@pytest.mark.asyncio
async def test_3branch_falls_back_when_hyde_returns_empty() -> None:
    retriever = _RecordingRetriever()
    hyde = _StubHyDE(doc="")  # LLM returned empty / failed
    svc = ChatService(
        retriever=retriever,
        reranker=_PassthroughReranker(),
        refusal_classifier=_Classifier("clinical_safe"),
        llm=_LLM("Metformin giảm liều [REF:1]."),
        known_drugs=["Metformin"],
        enforce_citations=True,
        hyde=hyde,
    )
    result = await svc.complete(
        [ChatMessage(role="user", content="liều metformin?")]
    )
    assert result.refused is False
    # Branch A only (no decomposer); HyDE called but produced no doc → branch C skipped retrieve
    queries = retriever.queries
    assert "liều metformin?" in queries
    assert "" not in queries


@pytest.mark.asyncio
async def test_crag_skipped_for_non_clinical_queries() -> None:
    chunks = [RetrievedChunk(text="x", source="s", score=0.9, metadata={})]
    # Grader would refuse, but non-clinical query should bypass it.
    svc = ChatService(
        retriever=_StubRetriever(chunks),
        reranker=_PassthroughReranker(),
        refusal_classifier=_Classifier("clinical_safe"),
        llm=_LLM("Hello [REF:1]."),
        enforce_citations=False,
        crag_grader=_FixedGrader("insufficient"),
    )
    # Question has no clinical keywords → is_clinical=False → CRAG bypassed.
    result = await svc.complete([ChatMessage(role="user", content="Xin chào")])
    assert result.refused is False
    assert result.trace is not None and result.trace.crag is None
