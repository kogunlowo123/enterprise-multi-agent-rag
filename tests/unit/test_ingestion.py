"""Unit tests for chunking and document loading."""

from __future__ import annotations

from itertools import pairwise
from pathlib import Path

import pytest

from emrag.errors import IngestionError
from emrag.ingestion import chunk_document, load_documents
from emrag.models import Document


def _doc(text: str) -> Document:
    return Document(id="d1", source="s.md", text=text)


class TestChunker:
    def test_short_document_is_one_chunk(self) -> None:
        chunks = chunk_document(_doc("One. Two."), max_chars=200, overlap_chars=20)
        assert [c.text for c in chunks] == ["One. Two."]
        assert chunks[0].id == "d1:0"

    def test_chunks_respect_max_and_overlap(self) -> None:
        sentences = [f"Sentence number {i} has some filler words." for i in range(12)]
        chunks = chunk_document(_doc(" ".join(sentences)), max_chars=120, overlap_chars=45)
        assert len(chunks) > 3
        assert all(len(c.text) <= 120 for c in chunks)
        for previous, current in pairwise(chunks):
            last_sentence = previous.text.split(". ")[-1].rstrip(".")
            assert last_sentence.split(" has")[0] in current.text
        assert [c.position for c in chunks] == list(range(len(chunks)))

    def test_tracks_markdown_section(self) -> None:
        text = "# Top\n\n## Alpha\n\nAlpha text here.\n\n## Beta\n\nBeta text here."
        chunks = chunk_document(_doc(text), max_chars=40, overlap_chars=0)
        sections = {c.section for c in chunks}
        assert {"Alpha", "Beta"} <= sections

    def test_heading_stays_on_own_paragraph(self) -> None:
        chunks = chunk_document(_doc("## Head\n\nBody sentence."), max_chars=200, overlap_chars=0)
        assert chunks[0].text == "## Head\n\nBody sentence."

    def test_long_sentence_is_hard_split(self) -> None:
        chunks = chunk_document(_doc("word " * 100), max_chars=50, overlap_chars=0)
        assert len(chunks) > 5
        assert all(len(c.text) <= 50 for c in chunks)

    def test_rejects_bad_parameters(self) -> None:
        with pytest.raises(ValueError, match="greater"):
            chunk_document(_doc("x"), max_chars=10, overlap_chars=10)

    def test_empty_document_yields_no_chunks(self) -> None:
        assert chunk_document(_doc("   "), max_chars=100, overlap_chars=10) == []


class TestLoader:
    def test_loads_directory_recursively(self, tmp_path: Path) -> None:
        (tmp_path / "a.md").write_text("Alpha document.", encoding="utf-8")
        (tmp_path / "sub").mkdir()
        (tmp_path / "sub" / "b.txt").write_text("Beta document.", encoding="utf-8")
        docs, skipped = load_documents(tmp_path, max_file_bytes=1000)
        assert sorted(d.source for d in docs) == ["a.md", "sub/b.txt"]
        assert skipped == []

    def test_document_ids_are_stable(self, tmp_path: Path) -> None:
        (tmp_path / "a.md").write_text("Alpha.", encoding="utf-8")
        first, _ = load_documents(tmp_path, max_file_bytes=1000)
        second, _ = load_documents(tmp_path, max_file_bytes=1000)
        assert first[0].id == second[0].id

    def test_loads_single_file(self, tmp_path: Path) -> None:
        target = tmp_path / "one.md"
        target.write_text("Only file.", encoding="utf-8")
        docs, _ = load_documents(target, max_file_bytes=1000)
        assert [d.source for d in docs] == ["one.md"]

    def test_reports_skips(self, tmp_path: Path) -> None:
        (tmp_path / "ok.md").write_text("Fine.", encoding="utf-8")
        (tmp_path / "image.png").write_bytes(b"\x89PNG")
        (tmp_path / "binary.txt").write_bytes(b"abc\x00def")
        (tmp_path / "empty.md").write_text("  \n", encoding="utf-8")
        (tmp_path / "big.md").write_text("x" * 500, encoding="utf-8")
        docs, skipped = load_documents(tmp_path, max_file_bytes=100)
        assert [d.source for d in docs] == ["ok.md"]
        reasons = " | ".join(skipped)
        for expected in ("unsupported extension", "binary content", "empty", "exceeds limit"):
            assert expected in reasons

    def test_skips_symlinks(self, tmp_path: Path) -> None:
        real = tmp_path / "real.md"
        real.write_text("Real.", encoding="utf-8")
        link = tmp_path / "link.md"
        try:
            link.symlink_to(real)
        except (OSError, NotImplementedError):
            pytest.skip("symlinks unavailable on this platform")
        docs, skipped = load_documents(tmp_path, max_file_bytes=1000)
        assert [d.source for d in docs] == ["real.md"]
        assert any("symbolic link" in s for s in skipped)

    def test_missing_path_raises(self, tmp_path: Path) -> None:
        with pytest.raises(IngestionError):
            load_documents(tmp_path / "nope", max_file_bytes=10)
