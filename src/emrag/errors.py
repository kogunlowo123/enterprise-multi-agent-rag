"""Exception hierarchy for the emrag package.

Every error raised deliberately by the library derives from :class:`EmragError`,
so callers can catch a single base type at API boundaries.
"""

from __future__ import annotations


class EmragError(Exception):
    """Base class for all errors raised by emrag."""


class ConfigurationError(EmragError):
    """Raised when settings are missing, inconsistent or invalid."""


class SecurityError(EmragError):
    """Raised when an input violates a security policy (for example prompt injection)."""


class IngestionError(EmragError):
    """Raised when a corpus cannot be loaded or indexed."""


class StoreError(EmragError):
    """Raised when a vector store operation fails."""


class ProviderError(EmragError):
    """Raised when an upstream model provider returns a non-retryable failure."""


class TransientProviderError(ProviderError):
    """Raised for retryable upstream failures such as rate limits or 5xx responses."""


class GraphError(EmragError):
    """Raised when the workflow graph is mis-wired or exceeds its step budget."""
