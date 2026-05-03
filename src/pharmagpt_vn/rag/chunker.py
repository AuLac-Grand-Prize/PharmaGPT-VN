"""Hierarchical document chunker (Plan §3.4.3).

Strategy:
  - Respect document structure (drug → section → paragraph)
  - No mid-sentence cuts; max 800 tok, min 100 tok
  - 15% overlap between adjacent chunks
  - Carry parent metadata so retrieval can show "Drug X / Section Y" context
  - Augment text with drug-name list for lexical (BM25) match

Token counting is delegated to a callable so tests can use a cheap whitespace counter
while production wires `tiktoken`.
"""

from collections.abc import Callable
from dataclasses import dataclass, field

DEFAULT_MAX_TOKENS = 800
DEFAULT_MIN_TOKENS = 100
DEFAULT_OVERLAP_RATIO = 0.15
SENTENCE_END = (".", "!", "?", "。")


@dataclass(frozen=True)
class Chunk:
    text: str
    source: str
    parent_path: tuple[str, ...]
    drug_names: tuple[str, ...] = field(default_factory=tuple)
    start_char: int = 0
    end_char: int = 0


@dataclass(frozen=True)
class Section:
    """A pre-segmented section (e.g. one drug entry) ready to chunk."""

    text: str
    source: str
    parent_path: tuple[str, ...]
    drug_names: tuple[str, ...] = field(default_factory=tuple)


def _count_whitespace_tokens(text: str) -> int:
    return len(text.split())


def chunk_section(
    section: Section,
    *,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    min_tokens: int = DEFAULT_MIN_TOKENS,
    overlap_ratio: float = DEFAULT_OVERLAP_RATIO,
    token_counter: Callable[[str], int] = _count_whitespace_tokens,
) -> list[Chunk]:
    """Split one section into overlapping chunks, never mid-sentence."""
    if token_counter(section.text) <= max_tokens:
        return [
            Chunk(
                text=section.text,
                source=section.source,
                parent_path=section.parent_path,
                drug_names=section.drug_names,
                start_char=0,
                end_char=len(section.text),
            )
        ]

    sentences = _split_sentences(section.text)
    chunks: list[Chunk] = []
    buffer: list[str] = []
    buffer_tokens = 0
    cursor = 0

    for sent in sentences:
        sent_tokens = token_counter(sent)
        if buffer_tokens + sent_tokens > max_tokens and buffer_tokens >= min_tokens:
            text = " ".join(buffer).strip()
            chunks.append(
                Chunk(
                    text=text,
                    source=section.source,
                    parent_path=section.parent_path,
                    drug_names=section.drug_names,
                    start_char=cursor,
                    end_char=cursor + len(text),
                )
            )
            cursor += int(len(text) * (1 - overlap_ratio))
            tail_tokens = int(buffer_tokens * overlap_ratio)
            buffer = _tail_by_tokens(buffer, tail_tokens, token_counter)
            buffer_tokens = sum(token_counter(s) for s in buffer)
        buffer.append(sent)
        buffer_tokens += sent_tokens

    if buffer:
        text = " ".join(buffer).strip()
        chunks.append(
            Chunk(
                text=text,
                source=section.source,
                parent_path=section.parent_path,
                drug_names=section.drug_names,
                start_char=cursor,
                end_char=cursor + len(text),
            )
        )
    return chunks


def chunk_corpus(
    sections: list[Section],
    **kwargs: object,
) -> list[Chunk]:
    """Chunk a list of sections, preserving source ordering."""
    out: list[Chunk] = []
    for s in sections:
        out.extend(chunk_section(s, **kwargs))  # type: ignore[arg-type]
    return out


def _split_sentences(text: str) -> list[str]:
    sentences: list[str] = []
    buf: list[str] = []
    for token in text.split():
        buf.append(token)
        if token.endswith(SENTENCE_END):
            sentences.append(" ".join(buf))
            buf = []
    if buf:
        sentences.append(" ".join(buf))
    return sentences


def _tail_by_tokens(
    buffer: list[str], target_tokens: int, token_counter: Callable[[str], int]
) -> list[str]:
    if target_tokens <= 0 or not buffer:
        return []
    tail: list[str] = []
    accumulated = 0
    for sent in reversed(buffer):
        tail.insert(0, sent)
        accumulated += token_counter(sent)
        if accumulated >= target_tokens:
            break
    return tail
