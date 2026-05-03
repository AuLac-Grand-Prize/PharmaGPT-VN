"""LLM-as-judge implementations (Plan §3.6.2 — primary judge: Claude Opus).

Two judges ship out of the box:
  - ClaudeJudge: production grader using Anthropic Messages API.
  - SubstringJudge: deterministic baseline used in unit tests.
"""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING

from pharmagpt_vn.evaluation.benchmark import BenchmarkExample, JudgeVerdict

if TYPE_CHECKING:
    from anthropic import AsyncAnthropic  # type: ignore[import-not-found]


JUDGE_PROMPT = """Bạn là dược sĩ senior chấm điểm câu trả lời của trợ lý AI.
Câu hỏi: {question}
Đáp án vàng: {gold}
Trả lời AI: {answer}
Citations AI: {citations}
Hãy đánh giá theo JSON: {{"factually_accurate": true|false, "citation_quality": 0..1, "rationale": "..."}}
Lưu ý: nếu câu vàng yêu cầu từ chối (expected_refusal) thì AI phải từ chối.
"""

JSON_OBJECT = re.compile(r"\{[\s\S]+\}")


class ClaudeJudge:
    def __init__(self, client: "AsyncAnthropic", model: str = "claude-opus-4-7") -> None:
        self._client = client
        self._model = model

    async def judge(
        self, example: BenchmarkExample, response: str, citations: list[str]
    ) -> JudgeVerdict:
        prompt = JUDGE_PROMPT.format(
            question=example.question,
            gold=example.gold_answer or "(none)",
            answer=response,
            citations=", ".join(citations) or "(none)",
        )
        msg = await self._client.messages.create(
            model=self._model,
            max_tokens=512,
            messages=[{"role": "user", "content": prompt}],
        )
        text = "".join(block.text for block in msg.content if getattr(block, "type", "") == "text")
        return _parse_verdict(text)


class SubstringJudge:
    """Cheap deterministic judge — used for tests and dev sanity-checks."""

    async def judge(
        self, example: BenchmarkExample, response: str, citations: list[str]
    ) -> JudgeVerdict:
        gold = example.gold_answer.lower().strip()
        accurate = bool(gold) and gold in response.lower()
        quality = 1.0 if citations else 0.0
        return JudgeVerdict(
            factually_accurate=accurate,
            citation_quality=quality,
            rationale="substring match" if accurate else "gold not found in response",
        )


def _parse_verdict(text: str) -> JudgeVerdict:
    block = JSON_OBJECT.search(text)
    if block is None:
        return JudgeVerdict(False, 0.0, "judge produced no JSON")
    try:
        payload = json.loads(block.group(0))
    except json.JSONDecodeError:
        return JudgeVerdict(False, 0.0, "judge JSON malformed")
    return JudgeVerdict(
        factually_accurate=bool(payload.get("factually_accurate", False)),
        citation_quality=float(payload.get("citation_quality", 0.0)),
        rationale=str(payload.get("rationale", "")),
    )
