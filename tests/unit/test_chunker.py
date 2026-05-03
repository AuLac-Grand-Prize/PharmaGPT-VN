from pharmagpt_vn.rag.chunker import Section, chunk_section


def test_short_section_returns_single_chunk() -> None:
    section = Section(
        text="Metformin chống chỉ định khi eGFR < 30.",
        source="VN_pharmacopeia",
        parent_path=("Drug", "Metformin", "Contraindications"),
        drug_names=("Metformin",),
    )
    chunks = chunk_section(section, max_tokens=100)
    assert len(chunks) == 1
    assert chunks[0].drug_names == ("Metformin",)
    assert chunks[0].parent_path[-1] == "Contraindications"


def test_long_section_splits_at_sentence_boundary() -> None:
    sentences = [f"Câu số {i} mô tả thông tin chi tiết về thuốc." for i in range(40)]
    section = Section(
        text=" ".join(sentences),
        source="src",
        parent_path=("Doc",),
    )
    chunks = chunk_section(section, max_tokens=50, min_tokens=10, overlap_ratio=0.15)
    assert len(chunks) >= 2
    for c in chunks:
        # Every chunk must end on a sentence terminator.
        assert c.text.rstrip().endswith((".", "!", "?"))
