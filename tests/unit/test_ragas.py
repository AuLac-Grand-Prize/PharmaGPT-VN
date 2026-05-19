from __future__ import annotations

from pharmagpt_vn.evaluation.ragas import (
    FaithfulnessDetail,
    RagasSample,
    SubstringRagasJudge,
    context_precision,
    context_recall,
    faithfulness,
    score_dataset,
    score_sample,
)


def _sample_supported() -> RagasSample:
    return RagasSample(
        question="Metformin có dùng được khi suy thận không?",
        answer=(
            "Metformin chống chỉ định khi eGFR dưới 30. "
            "Cần giảm liều metformin khi eGFR 30 đến 45."
        ),
        contexts=(
            "Metformin chống chỉ định khi eGFR < 30 mL/phút.",
            "Với eGFR 30-45 mL/phút giới hạn metformin ≤ 1000mg/ngày.",
        ),
        ground_truth=(
            "Metformin chống chỉ định khi eGFR dưới 30. "
            "Khi eGFR 30-45 phải giảm liều metformin."
        ),
    )


def _sample_hallucinated() -> RagasSample:
    return RagasSample(
        question="Aspirin liều ở trẻ em?",
        answer="Aspirin có thể dùng tự do cho trẻ em từ 5 tuổi trở lên.",
        contexts=(
            "Aspirin không dùng cho trẻ em dưới 12 tuổi do nguy cơ hội chứng Reye.",
        ),
        ground_truth="Aspirin không dùng cho trẻ em dưới 12 tuổi vì hội chứng Reye.",
    )


def test_faithfulness_high_when_claims_supported() -> None:
    detail: FaithfulnessDetail = faithfulness(_sample_supported(), SubstringRagasJudge())
    assert detail.claims_total >= 1
    assert detail.score >= 0.5


def test_faithfulness_low_for_hallucination() -> None:
    detail: FaithfulnessDetail = faithfulness(_sample_hallucinated(), SubstringRagasJudge())
    assert detail.score < 0.5  # claim contradicts the only context


def test_context_precision_full_when_all_relevant() -> None:
    p = context_precision(_sample_supported(), SubstringRagasJudge())
    assert p >= 0.5


def test_context_precision_zero_for_empty_contexts() -> None:
    sample = RagasSample(question="q?", answer="a", contexts=())
    assert context_precision(sample, SubstringRagasJudge()) == 0.0


def test_context_recall_finds_gold_claims_in_contexts() -> None:
    r = context_recall(_sample_supported(), SubstringRagasJudge())
    assert r > 0.0


def test_context_recall_zero_without_ground_truth() -> None:
    s = RagasSample(question="q?", answer="a", contexts=("ctx",))
    assert context_recall(s, SubstringRagasJudge()) == 0.0


def test_score_sample_returns_all_three_metrics() -> None:
    score = score_sample(_sample_supported(), SubstringRagasJudge())
    assert 0.0 <= score.faithfulness <= 1.0
    assert 0.0 <= score.context_precision <= 1.0
    assert 0.0 <= score.context_recall <= 1.0


def test_hallucination_rate_is_complement_of_faithfulness() -> None:
    score = score_sample(_sample_hallucinated(), SubstringRagasJudge())
    assert abs(score.hallucination_rate - (1 - score.faithfulness)) < 1e-9


def test_score_dataset_averages_across_samples() -> None:
    score = score_dataset(
        [_sample_supported(), _sample_hallucinated()], SubstringRagasJudge()
    )
    assert 0.0 < score.faithfulness < 1.0


def test_score_dataset_empty_returns_zeros() -> None:
    z = score_dataset([], SubstringRagasJudge())
    assert (z.faithfulness, z.context_precision, z.context_recall) == (0.0, 0.0, 0.0)


def test_judge_strips_citations_when_extracting_claims() -> None:
    judge = SubstringRagasJudge()
    claims = judge.extract_claims("Metformin chống chỉ định khi eGFR dưới 30 [REF:1].")
    assert claims
    assert "[REF:1]" not in claims[0]
