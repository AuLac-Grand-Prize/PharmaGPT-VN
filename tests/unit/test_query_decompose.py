"""HeuristicVNDecomposer — drug detection, context pairing, fallbacks."""

from __future__ import annotations

from pharmagpt_vn.rag.query_decompose import HeuristicVNDecomposer


def test_multi_drug_query_emits_interaction_and_context_pairs() -> None:
    d = HeuristicVNDecomposer(max_sub=10, known_drugs=["Metformin", "Amlodipine"])
    out = d.decompose(
        "Metformin 500mg cho bn nữ người cao tuổi suy thận "
        "có dùng được với Amlodipine không?"
    )
    out_lower = [s.lower() for s in out]
    assert any("metformin tương tác amlodipine" in s for s in out_lower)
    assert any("metformin suy thận" in s for s in out_lower)
    assert any("amlodipine suy thận" in s for s in out_lower)
    assert any("metformin người cao tuổi" in s for s in out_lower)
    assert len(out) <= 10


def test_single_drug_with_context() -> None:
    d = HeuristicVNDecomposer(known_drugs=["Metformin"])
    out = d.decompose("Metformin có dùng được cho bn suy thận không?")
    assert out == ["Metformin suy thận"]


def test_no_match_falls_back_to_original() -> None:
    d = HeuristicVNDecomposer(known_drugs=["Metformin"])
    out = d.decompose("xin chào dược sĩ")
    assert out == ["xin chào dược sĩ"]


def test_capitalized_fallback_when_no_whitelist() -> None:
    d = HeuristicVNDecomposer()  # no known_drugs
    out = d.decompose("Metformin và Amlodipine dùng cùng được không?")
    out_lower = [s.lower() for s in out]
    assert any("metformin tương tác amlodipine" in s for s in out_lower)


def test_max_sub_caps_output() -> None:
    d = HeuristicVNDecomposer(max_sub=2, known_drugs=["A", "B", "C"])
    out = d.decompose("A và B với C cho bn suy thận")
    assert len(out) <= 2


def test_dedupe_normalized() -> None:
    d = HeuristicVNDecomposer(known_drugs=["Metformin"])
    # Same drug + same context appears once even if context name in query is repeated.
    out = d.decompose("Metformin suy thận suy thận suy thận")
    assert out.count("Metformin suy thận") == 1


def test_conjunction_split_as_last_resort() -> None:
    # No drug names, no clinical context — fall through to conjunction split.
    d = HeuristicVNDecomposer()
    out = d.decompose("tác dụng phụ và liều dùng")
    assert len(out) == 2
    assert "tác dụng phụ" in out
    assert "liều dùng" in out
