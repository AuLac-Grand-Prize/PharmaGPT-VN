"""VN-PharmBench runner — load JSONL, generate, judge, aggregate (Plan §3.6).

Categories follow the SFT taxonomy in Plan §3.2.2:
  drug_info_basic, drug_info_advanced, dosage_adjustment, interactions,
  contraindications, otc_counseling, refusal.

Each example has:
  id, category, question, gold_answer, gold_citations, expected_refusal (bool).

The runner is decoupled from the LLM and the judge so it can be unit-tested
deterministically, and so production can swap in Claude Opus or a local judge.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Iterable
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Protocol

from pharmagpt_vn.services.chat_service import ChatMessage, ChatResult, ChatService


class Category(str, Enum):
    DRUG_INFO_BASIC = "drug_info_basic"
    DRUG_INFO_ADVANCED = "drug_info_advanced"
    DOSAGE_ADJUSTMENT = "dosage_adjustment"
    INTERACTIONS = "interactions"
    CONTRAINDICATIONS = "contraindications"
    OTC_COUNSELING = "otc_counseling"
    REFUSAL = "refusal"


@dataclass(frozen=True)
class BenchmarkExample:
    id: str
    category: Category
    question: str
    gold_answer: str = ""
    gold_citations: tuple[str, ...] = ()
    expected_refusal: bool = False


@dataclass(frozen=True)
class JudgeVerdict:
    factually_accurate: bool
    citation_quality: float  # 0..1
    rationale: str = ""


@dataclass(frozen=True)
class ExampleResult:
    example: BenchmarkExample
    response: str
    refused: bool
    citations: tuple[str, ...]
    verdict: JudgeVerdict
    latency_ms: int = 0


@dataclass(frozen=True)
class BenchmarkReport:
    total: int
    overall_accuracy: float
    overall_citation_quality: float
    refusal_appropriate_rate: float
    by_category: dict[str, dict[str, float]] = field(default_factory=dict)
    failures: list[str] = field(default_factory=list)


class LLMJudge(Protocol):
    async def judge(
        self, example: BenchmarkExample, response: str, citations: list[str]
    ) -> JudgeVerdict: ...


class BenchmarkRunner:
    def __init__(self, chat: ChatService, judge: LLMJudge, concurrency: int = 4) -> None:
        self._chat = chat
        self._judge = judge
        self._sem = asyncio.Semaphore(concurrency)

    async def run(self, examples: Iterable[BenchmarkExample]) -> BenchmarkReport:
        results: list[ExampleResult] = []
        async for r in self._run_examples(examples):
            results.append(r)
        return self._aggregate(results)

    async def _run_examples(
        self, examples: Iterable[BenchmarkExample]
    ) -> AsyncIterator[ExampleResult]:
        examples_list = list(examples)
        coros = [self._run_one(ex) for ex in examples_list]
        for coro in asyncio.as_completed(coros):
            yield await coro

    async def _run_one(self, ex: BenchmarkExample) -> ExampleResult:
        async with self._sem:
            chat_result = await self._chat.complete([ChatMessage(role="user", content=ex.question)])
        verdict = await self._judge.judge(ex, chat_result.content, list(chat_result.citations))
        return ExampleResult(
            example=ex,
            response=chat_result.content,
            refused=chat_result.refused,
            citations=tuple(chat_result.citations),
            verdict=verdict,
        )

    @staticmethod
    def _aggregate(results: list[ExampleResult]) -> BenchmarkReport:
        from pharmagpt_vn.evaluation.metrics import (
            aggregate_by_category,
            citation_quality,
            factual_accuracy,
            refusal_appropriateness,
        )

        if not results:
            return BenchmarkReport(0, 0.0, 0.0, 0.0)

        return BenchmarkReport(
            total=len(results),
            overall_accuracy=factual_accuracy(results),
            overall_citation_quality=citation_quality(results),
            refusal_appropriate_rate=refusal_appropriateness(results),
            by_category=aggregate_by_category(results),
            failures=[r.example.id for r in results if not r.verdict.factually_accurate],
        )


# ---- I/O ---------------------------------------------------------------------


def load_jsonl(path: str | Path) -> list[BenchmarkExample]:
    out: list[BenchmarkExample] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        record = json.loads(line)
        out.append(_record_to_example(record))
    return out


def _record_to_example(record: dict) -> BenchmarkExample:
    return BenchmarkExample(
        id=str(record["id"]),
        category=Category(record["category"]),
        question=record["question"],
        gold_answer=record.get("gold_answer", ""),
        gold_citations=tuple(record.get("gold_citations", [])),
        expected_refusal=bool(record.get("expected_refusal", False)),
    )


def report_to_json(report: BenchmarkReport) -> str:
    return json.dumps(asdict(report), ensure_ascii=False, indent=2)
