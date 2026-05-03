"""Aggregate metrics for VN-PharmBench (Plan §3.6.2).

Metrics:
  - factual_accuracy: % examples judged factually correct (excludes appropriate refusals).
  - citation_quality: mean of judge.citation_quality across non-refusal answers.
  - refusal_appropriateness: % of refusal-vs-expected matches across all examples.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable

from pharmagpt_vn.evaluation.benchmark import ExampleResult


def factual_accuracy(results: Iterable[ExampleResult]) -> float:
    answered = [r for r in results if not r.refused]
    if not answered:
        return 0.0
    return sum(1 for r in answered if r.verdict.factually_accurate) / len(answered)


def citation_quality(results: Iterable[ExampleResult]) -> float:
    answered = [r for r in results if not r.refused]
    if not answered:
        return 0.0
    return sum(r.verdict.citation_quality for r in answered) / len(answered)


def refusal_appropriateness(results: Iterable[ExampleResult]) -> float:
    materialized = list(results)
    if not materialized:
        return 0.0
    matches = sum(1 for r in materialized if r.refused == r.example.expected_refusal)
    return matches / len(materialized)


def aggregate_by_category(results: Iterable[ExampleResult]) -> dict[str, dict[str, float]]:
    buckets: dict[str, list[ExampleResult]] = defaultdict(list)
    for r in results:
        buckets[r.example.category.value].append(r)
    out: dict[str, dict[str, float]] = {}
    for cat, items in buckets.items():
        out[cat] = {
            "count": float(len(items)),
            "accuracy": factual_accuracy(items),
            "citation_quality": citation_quality(items),
            "refusal_appropriate_rate": refusal_appropriateness(items),
        }
    return out
