"""Filesystem document loader with size limits and symlink protection."""

from __future__ import annotations

import hashlib
from pathlib import Path

from emrag.errors import IngestionError
from emrag.models import Document

SUPPORTED_SUFFIXES = frozenset({".txt", ".md", ".markdown", ".rst"})


def _document_id(source: str) -> str:
    return hashlib.sha256(source.encode("utf-8")).hexdigest()[:16]


def load_documents(
    path: Path,
    *,
    max_file_bytes: int,
    suffixes: frozenset[str] = SUPPORTED_SUFFIXES,
) -> tuple[list[Document], list[str]]:
    """Load text documents from a file or directory tree.

    Symbolic links are never followed, oversized files and non-text content are
    skipped, and every skip is reported so ingestion is auditable.

    Args:
        path: File or directory to load.
        max_file_bytes: Files larger than this are skipped.
        suffixes: Lowercase file extensions to accept.

    Returns:
        A tuple of loaded documents and human-readable skip reasons.

    Raises:
        IngestionError: If ``path`` does not exist.
    """
    if not path.exists():
        raise IngestionError(f"path does not exist: {path}")

    root = path if path.is_dir() else path.parent
    candidates = sorted(p for p in path.rglob("*")) if path.is_dir() else [path]

    documents: list[Document] = []
    skipped: list[str] = []
    for candidate in candidates:
        if candidate.is_dir():
            continue
        relative = candidate.relative_to(root).as_posix()
        if candidate.is_symlink():
            skipped.append(f"{relative}: symbolic link")
            continue
        if candidate.suffix.lower() not in suffixes:
            skipped.append(f"{relative}: unsupported extension")
            continue
        size = candidate.stat().st_size
        if size > max_file_bytes:
            skipped.append(f"{relative}: {size} bytes exceeds limit")
            continue
        raw = candidate.read_bytes()
        if b"\x00" in raw:
            skipped.append(f"{relative}: binary content")
            continue
        text = raw.decode("utf-8", errors="replace").strip()
        if not text:
            skipped.append(f"{relative}: empty")
            continue
        documents.append(
            Document(
                id=_document_id(relative), source=relative, text=text, metadata={"path": relative}
            )
        )
    return documents, skipped
