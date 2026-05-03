from pharmagpt_vn.core.guardrails import is_clinical_query, redact_pii


def test_redact_cccd() -> None:
    assert "[CCCD]" in redact_pii("Số CCCD của tôi là 012345678901")


def test_redact_phone() -> None:
    assert "[PHONE]" in redact_pii("Liên hệ tôi 0912345678")


def test_clinical_query_detection() -> None:
    assert is_clinical_query("metformin có tương tác với warfarin không?") is True


def test_non_clinical_query() -> None:
    assert is_clinical_query("Hôm nay thời tiết thế nào?") is False
