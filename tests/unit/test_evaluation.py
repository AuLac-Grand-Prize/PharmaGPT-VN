from __future__ import annotations

from pathlib import Path

import pytest

from pharmagpt_vn.evaluation import BenchmarkRunner, Category, load_jsonl
from pharmagpt_vn.evaluation.benchmark import BenchmarkExample
from pharmagpt_vn.evaluation.judges import SubstringJudge
from pharmagpt_vn.evaluation.metrics import (
    aggregate_by_category,
    factual_accuracy,
    refusal_appropriateness,
)
from pharmagpt_vn.evaluation.benchmark import ExampleResult, JudgeVerdict
from pharmagpt_vn.services.chat_service import ChatMessage, ChatResult, ChatService

FIXTURE = Path(__file__).parent.parent / "fixtures" / "vn_pharmbench_mini.jsonl"


def test_load_jsonl_parses_categories() -> None:
    examples = load_jsonl(FIXTURE)
    assert len(examples) == 3
    assert examples[0].category is Category.DRUG_INFO_BASIC
    assert examples[2].expected_refusal is True


def _result(
    *,
    category: Category,
    answer: str,
    refused: bool,
    accurate: bool,
    expected_refusal: bool = False,
    citations: tuple[str, ...] = (),
) -> ExampleResult:
    ex = BenchmarkExample(
        id="x", category=category, question="?", gold_answer=answer, expected_refusal=expected_refusal
    )
    return ExampleResult(
        example=ex,
        response=answer,
        refused=refused,
        citations=citations,
        verdict=JudgeVerdict(factually_accurate=accurate, citation_quality=1.0 if citations else 0.0),
    )


def test_factual_accuracy_excludes_refusals() -> None:
    results = [
        _result(category=Category.DRUG_INFO_BASIC, answer="ok", refused=False, accurate=True),
        _result(category=Category.REFUSAL, answer="", refused=True, accurate=False, expected_refusal=True),
    ]
    assert factual_accuracy(results) == 1.0  # the only answered one was correct


def test_refusal_appropriateness_credits_correct_refusals() -> None:
    results = [
        _result(category=Category.REFUSAL, answer="", refused=True, accurate=False, expected_refusal=True),
        _result(category=Category.DRUG_INFO_BASIC, answer="ok", refused=False, accurate=True),
        _result(category=Category.DRUG_INFO_BASIC, answer="hallucinated", refused=False, accurate=False),
    ]
    assert refusal_appropriateness(results) == 1.0


def test_aggregate_by_category_buckets_correctly() -> None:
    results = [
        _result(category=Category.DRUG_INFO_BASIC, answer="ok", refused=False, accurate=True),
        _result(category=Category.DRUG_INFO_BASIC, answer="bad", refused=False, accurate=False),
        _result(category=Category.INTERACTIONS, answer="ok", refused=False, accurate=True),
    ]
    by_cat = aggregate_by_category(results)
    assert by_cat["drug_info_basic"]["accuracy"] == 0.5
    assert by_cat["interactions"]["accuracy"] == 1.0


class _StubChat(ChatService):
    def __init__(self, responses: dict[str, ChatResult]) -> None:
        self._responses = responses

    async def complete(  # type: ignore[override]
        self,
        messages: list[ChatMessage],
        rag_top_k: int = 5,
        rerank_keep: int = 5,
        retrieve_pool: int = 50,
        temperature: float = 0.2,
        max_tokens: int = 512,
    ) -> ChatResult:
        last = messages[-1].content
        return self._responses.get(last, ChatResult(content="", refused=True))


@pytest.mark.asyncio
async def test_runner_end_to_end_with_substring_judge() -> None:
    examples = load_jsonl(FIXTURE)
    chat = _StubChat(
        {
            examples[0].question: ChatResult(
                content="Metformin dùng cho đái tháo đường típ 2 [REF:1].",
                citations=["[REF:1] VN_metformin"],
            ),
            examples[1].question: ChatResult(
                content="Có. Warfarin có tương tác làm tăng nguy cơ chảy máu khi phối hợp Aspirin [REF:1].",
                citations=["[REF:1] VN_warfarin"],
            ),
            examples[2].question: ChatResult(content="Em không thể giúp.", refused=True),
        }
    )
    runner = BenchmarkRunner(chat, SubstringJudge(), concurrency=2)
    report = await runner.run(examples)
    assert report.total == 3
    assert report.overall_accuracy == 1.0
    assert report.refusal_appropriate_rate == 1.0
    assert "drug_info_basic" in report.by_category
