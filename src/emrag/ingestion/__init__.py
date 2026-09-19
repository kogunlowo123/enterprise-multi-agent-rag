"""Document loading and chunking."""

from emrag.ingestion.chunker import chunk_document
from emrag.ingestion.loader import load_documents

__all__ = ["chunk_document", "load_documents"]
