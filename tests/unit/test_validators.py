from pharmagpt_vn.core.validators import (
    DosageRange,
    extract_citation_ids,
    validate_citations,
    validate_dosage_sanity,
    validate_drug_names,
    validate_tone,
)


def test_extract_citation_ids() -> None:
    assert extract_citation_ids("Aspirin gây xuất huyết [REF:1]. Liều thấp [REF:2].") == {1, 2}


def test_citation_passes_when_all_refs_known_and_coverage_high() -> None:
    text = "Metformin chống chỉ định ở suy thận nặng [REF:1]. Liều thường 500mg [REF:2]."
    result, report = validate_citations(text, available_refs={1, 2})
    assert result.passed
    assert report.coverage == 1.0


def test_citation_fails_when_invalid_ref() -> None:
    text = "Sai sự thật [REF:99]."
    result, _ = validate_citations(text, available_refs={1, 2})
    assert not result.passed
    assert "99" in result.detail


def test_citation_fails_when_coverage_below_threshold() -> None:
    text = "Câu một không có nguồn. Câu hai cũng không. Câu ba có [REF:1]."
    result, report = validate_citations(text, available_refs={1}, min_coverage=0.85)
    assert not result.passed
    assert report.cited_sentences == 1
    assert report.sentence_count == 3


def test_drug_name_known() -> None:
    assert validate_drug_names("Metformin 500mg", ["Metformin", "Insulin"]).passed


def test_drug_name_unknown_flags() -> None:
    result = validate_drug_names("Phantomidine 100mg", ["Metformin"])
    assert not result.passed
    assert "phantomidine" in result.detail.lower()


def test_dosage_sanity_in_range() -> None:
    assert validate_dosage_sanity(
        "Metformin 500 mg",
        [DosageRange("Metformin", 500, 2000)],
    ).passed


def test_dosage_sanity_absurd() -> None:
    result = validate_dosage_sanity(
        "Metformin 50000 mg mỗi ngày",
        [DosageRange("Metformin", 500, 2000)],
    )
    assert not result.passed


def test_tone_rejects_self_medication_advice() -> None:
    assert not validate_tone("Bạn nên tự mua thuốc kháng sinh.").passed


def test_tone_passes_safe_phrasing() -> None:
    assert validate_tone("Vui lòng hỏi dược sĩ trước khi dùng.").passed
