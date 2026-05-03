"""VN-PharmBench evaluation harness (Plan §3.6)."""

from pharmagpt_vn.evaluation.benchmark import (
    BenchmarkExample,
    BenchmarkReport,
    BenchmarkRunner,
    Category,
    ExampleResult,
    LLMJudge,
    load_jsonl,
)
from pharmagpt_vn.evaluation.metrics import (
    aggregate_by_category,
    citation_quality,
    factual_accuracy,
    refusal_appropriateness,
)

__all__ = [
    "BenchmarkExample",
    "BenchmarkReport",
    "BenchmarkRunner",
    "Category",
    "ExampleResult",
    "LLMJudge",
    "aggregate_by_category",
    "citation_quality",
    "factual_accuracy",
    "load_jsonl",
    "refusal_appropriateness",
]
