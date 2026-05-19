"""RAGAS-style RAG metrics (Es et al. 2023).

We compute three metrics by ourselves rather than depending on the `ragas`
package — for two reasons:

  1. The pharma KPI (faithfulness ≤ 3% hallucination) needs a judge we control.
  2. `ragas` pulls heavyweight deps and an opinionated LLM client; this module
     accepts any `RagasJudge` via Protocol so we plug Claude / a local model /
     a deterministic test stub identically.

Metrics
-------
- **faithfulness**: of the atomic claims in the answer, what fraction are
  supported by the retrieved context? 0..1, higher = less hallucination.
- **context_precision**: of the retrieved chunks, what fraction are relevant
  to the question? Higher = retriever didn't dilute the prompt.
- **context_recall**: of the gold-answer claims, what fraction appear in the
  retrieved context? Higher = retriever didn't miss the answer.

All three accept an injected `RagasJudge` that returns binary labels for
sub-decisions; the metric itself is just arithmetic on those labels.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Sample types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RagasSample:
    question: str
    answer: str
    contexts: tuple[str, ...]
    ground_truth: str = ""  # required for context_recall, optional otherwise


@dataclass(frozen=True)
class RagasScore:
    faithfulness: float
    context_precision: float
    context_recall: float

    @property
    def hallucination_rate(self) -> float:
        return 1.0 - self.faithfulness


@dataclass(frozen=True)
class FaithfulnessDetail:
    claims_total: int
    claims_supported: int

    @property
    def score(self) -> float:
        if self.claims_total == 0:
            return 1.0  # no claims = nothing to hallucinate
        return self.claims_supported / self.claims_total


# ---------------------------------------------------------------------------
# Judge protocol
# ---------------------------------------------------------------------------


class RagasJudge(Protocol):
    """Three binary helpers — one per metric.

    Implementations are free to call an LLM behind the scenes (this is what
    ragas does). A deterministic substring judge is included for tests.
    """

    def extract_claims(self, answer: str) -> list[str]: ...
    def is_claim_supported(self, claim: str, contexts: Sequence[str]) -> bool: ...
    def is_context_relevant(self, question: str, context: str) -> bool: ...
    def is_ground_truth_covered(
        self, ground_truth_claim: str, contexts: Sequence[str]
    ) -> bool: ...


# ---------------------------------------------------------------------------
# Metric functions
# ---------------------------------------------------------------------------


def faithfulness(sample: RagasSample, judge: RagasJudge) -> FaithfulnessDetail:
    claims = judge.extract_claims(sample.answer)
    supported = sum(1 for c in claims if judge.is_claim_supported(c, sample.contexts))
    return FaithfulnessDetail(claims_total=len(claims), claims_supported=supported)


def context_precision(sample: RagasSample, judge: RagasJudge) -> float:
    """% of retrieved chunks relevant to the question.

    Order-sensitive in the original RAGAS paper (uses average precision); we
    keep the simpler unweighted mean since our retriever already RRF-sorts.
    """
    if not sample.contexts:
        return 0.0
    relevant = sum(1 for ctx in sample.contexts if judge.is_context_relevant(sample.question, ctx))
    return relevant / len(sample.contexts)


def context_recall(sample: RagasSample, judge: RagasJudge) -> float:
    """% of gold-answer claims found in the retrieved context."""
    if not sample.ground_truth:
        return 0.0  # caller must provide gold for recall
    gt_claims = judge.extract_claims(sample.ground_truth)
    if not gt_claims:
        return 0.0
    covered = sum(1 for c in gt_claims if judge.is_ground_truth_covered(c, sample.contexts))
    return covered / len(gt_claims)


def score_sample(sample: RagasSample, judge: RagasJudge) -> RagasScore:
    return RagasScore(
        faithfulness=faithfulness(sample, judge).score,
        context_precision=context_precision(sample, judge),
        context_recall=context_recall(sample, judge),
    )


def score_dataset(samples: Sequence[RagasSample], judge: RagasJudge) -> RagasScore:
    """Aggregate by simple mean across samples."""
    if not samples:
        return RagasScore(0.0, 0.0, 0.0)
    scores = [score_sample(s, judge) for s in samples]
    n = len(scores)
    return RagasScore(
        faithfulness=sum(s.faithfulness for s in scores) / n,
        context_precision=sum(s.context_precision for s in scores) / n,
        context_recall=sum(s.context_recall for s in scores) / n,
    )


# ---------------------------------------------------------------------------
# Deterministic test/dev judge — splits on sentence boundary, checks substring.
# Useful for local sanity-checks before wiring a real LLM judge.
# ---------------------------------------------------------------------------


_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")
_CITATION_RE = re.compile(r"\[REF:\d+\]")


class SubstringRagasJudge:
    """Splits answer/ground_truth on sentence boundaries and checks substring
    containment in any context. Stop-word filtering is intentionally minimal —
    this exists to sanity-check the pipeline shape, not to replace Claude.
    """

    def extract_claims(self, answer: str) -> list[str]:
        cleaned = _CITATION_RE.sub("", answer)
        sents = [s.strip() for s in _SENT_SPLIT.split(cleaned) if s.strip()]
        return [s for s in sents if len(s.split()) >= 3]

    SUPPORT_OVERLAP_THRESHOLD = 0.6

    def is_claim_supported(self, claim: str, contexts: Sequence[str]) -> bool:
        tokens = _significant_tokens(claim)
        if not tokens:
            return False
        best = 0.0
        for ctx in contexts:
            ctx_l = ctx.lower()
            hits = sum(1 for tok in tokens if tok in ctx_l)
            best = max(best, hits / len(tokens))
        return best >= self.SUPPORT_OVERLAP_THRESHOLD

    def is_context_relevant(self, question: str, context: str) -> bool:
        q_tokens = _significant_tokens(question)
        ctx_l = context.lower()
        if not q_tokens:
            return False
        hits = sum(1 for t in q_tokens if t in ctx_l)
        return hits >= max(1, len(q_tokens) // 3)

    def is_ground_truth_covered(
        self, ground_truth_claim: str, contexts: Sequence[str]
    ) -> bool:
        return self.is_claim_supported(ground_truth_claim, contexts)


_STOPWORDS_VI = frozenset(
    {
        "là", "và", "của", "có", "không", "cho", "với", "khi", "trong", "này",
        "đó", "các", "một", "để", "được", "phải", "thì", "ở", "tại", "nào",
        "the", "a", "of", "and", "to", "in", "for",
    }
)


def _significant_tokens(text: str) -> list[str]:
    out: list[str] = []
    for tok in re.findall(r"\w+", text.lower(), flags=re.UNICODE):
        if tok in _STOPWORDS_VI:
            continue
        if len(tok) < 2:
            continue
        out.append(tok)
    return out


__all__ = [
    "FaithfulnessDetail",
    "RagasJudge",
    "RagasSample",
    "RagasScore",
    "SubstringRagasJudge",
    "context_precision",
    "context_recall",
    "faithfulness",
    "score_dataset",
    "score_sample",
]
