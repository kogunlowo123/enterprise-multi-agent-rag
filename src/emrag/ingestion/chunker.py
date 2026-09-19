"""Sentence-aware chunking with overlap and Markdown section tracking."""

from __future__ import annotations

from emrag.models import Chunk, Document
from emrag.text import split_sentences


def _units(text: str) -> list[tuple[str, str | None]]:
    """Split ``text`` into ``(sentence, section)`` pairs, tracking Markdown headings."""
    units: list[tuple[str, str | None]] = []
    section: str | None = None
    for sentence in split_sentences(text):
        if sentence.startswith("#"):
            section = sentence.lstrip("#").strip() or section
        units.append((sentence, section))
    return units


def _hard_split(sentence: str, max_chars: int) -> list[str]:
    """Break an over-long sentence on word boundaries."""
    parts: list[str] = []
    current: list[str] = []
    length = 0
    for word in sentence.split():
        if current and length + len(word) + 1 > max_chars:
            parts.append(" ".join(current))
            current, length = [], 0
        current.append(word)
        length += len(word) + 1
    if current:
        parts.append(" ".join(current))
    return parts


def _join(window: list[tuple[str, str | None]]) -> str:
    """Join sentences with spaces, keeping Markdown headings on their own paragraph."""
    parts: list[str] = []
    previous = ""
    for sentence, _ in window:
        if parts:
            parts.append("\n\n" if sentence.startswith("#") or previous.startswith("#") else " ")
        parts.append(sentence)
        previous = sentence
    return "".join(parts).strip()


def chunk_document(document: Document, *, max_chars: int, overlap_chars: int) -> list[Chunk]:
    """Split ``document`` into overlapping chunks of at most ``max_chars`` characters.

    Chunks end on sentence boundaries where possible. The trailing sentences of each
    chunk, up to ``overlap_chars`` characters, are repeated at the start of the next
    so answers spanning a boundary remain retrievable.
    """
    if max_chars <= overlap_chars:
        raise ValueError("max_chars must be greater than overlap_chars")

    expanded: list[tuple[str, str | None]] = []
    for sentence, section in _units(document.text):
        if len(sentence) > max_chars:
            expanded.extend((part, section) for part in _hard_split(sentence, max_chars))
        else:
            expanded.append((sentence, section))

    chunks: list[Chunk] = []
    window: list[tuple[str, str | None]] = []

    def emit() -> None:
        text = _join(window)
        if not text:
            return
        position = len(chunks)
        # Label the chunk with the section of its first body sentence, not a bare heading.
        section = next((sec for sent, sec in window if not sent.startswith("#")), window[0][1])
        chunks.append(
            Chunk(
                id=f"{document.id}:{position}",
                doc_id=document.id,
                source=document.source,
                text=text,
                position=position,
                section=section,
            )
        )

    for unit in expanded:
        size = sum(len(s) + 1 for s, _ in window) + len(unit[0])
        if window and size > max_chars:
            emit()
            carried: list[tuple[str, str | None]] = []
            carried_len = 0
            for prior in reversed(window):
                if carried_len + len(prior[0]) + 1 > overlap_chars:
                    break
                carried.insert(0, prior)
                carried_len += len(prior[0]) + 1
            window = carried
        window.append(unit)

    if window and (not chunks or _join(window) != chunks[-1].text):
        emit()
    return chunks
