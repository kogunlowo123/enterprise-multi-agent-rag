"""Enterprise multi-agent retrieval-augmented generation."""

from emrag.config import Settings
from emrag.container import build_service
from emrag.models import Answer, IngestReport
from emrag.service import RAGService

__version__ = "0.1.0"

__all__ = ["Answer", "IngestReport", "RAGService", "Settings", "__version__", "build_service"]
