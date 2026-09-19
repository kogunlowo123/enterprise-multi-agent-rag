"""Input validation, prompt-injection screening and secret redaction.

Two trust boundaries are enforced here:

* User queries are validated for size, control characters and injection phrasing
  before they reach any agent.
* Corpus text is untrusted. Passages that look like instructions aimed at a model
  are quarantined at ingestion time, and credentials found in documents are
  redacted before anything is embedded, stored or logged.
"""

from __future__ import annotations

import re
import unicodedata

from emrag.errors import SecurityError

REDACTION = "[REDACTED]"

_INJECTION_PATTERNS: dict[str, re.Pattern[str]] = {
    "ignore_instructions": re.compile(
        r"\b(ignore|disregard|forget|override)\b[^.\n]{0,40}\b(previous|prior|above|earlier|all|any)\b"
        r"[^.\n]{0,40}\b(instructions?|rules?|prompts?|guidelines?)\b",
        re.IGNORECASE,
    ),
    "role_reassignment": re.compile(
        r"\byou\s+are\s+now\b|\bact\s+as\s+(?:an?\s+)?(?:unrestricted|jailbroken)\b", re.IGNORECASE
    ),
    "prompt_exfiltration": re.compile(
        r"\b(reveal|print|show|repeat|leak|output)\b[^.\n]{0,30}\b(system\s+prompt|hidden\s+prompt|"
        r"instructions\s+above|api\s+key|secret)\b",
        re.IGNORECASE,
    ),
    "role_tags": re.compile(r"<\s*/?\s*(system|assistant|developer)\s*>", re.IGNORECASE),
    "chat_template_tokens": re.compile(
        r"<\|(?:im_start|im_end|system|endoftext)\|>", re.IGNORECASE
    ),
}

_SECRET_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"sk-ant-[A-Za-z0-9_\-]{16,}"),
    re.compile(r"sk-[A-Za-z0-9_\-]{20,}"),
    re.compile(r"github_pat_[A-Za-z0-9_]{20,}"),
    re.compile(r"gh[pousr]_[A-Za-z0-9]{30,}"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._\-]{16,}"),
    re.compile(
        r"-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----[\s\S]*?-----END "
        r"(?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----"
    ),
    re.compile(
        r"(?i)\b(password|passwd|secret|api[_-]?key|token)\b\s*[:=]\s*['\"]?"
        r"(?!\[REDACTED\])[^\s'\",;]{6,}"
    ),
)

_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def scan_for_injection(text: str) -> list[str]:
    """Return the names of every injection heuristic that matches ``text``."""
    return [name for name, pattern in _INJECTION_PATTERNS.items() if pattern.search(text)]


def redact_secrets(text: str) -> tuple[str, int]:
    """Replace credential-shaped substrings with a placeholder.

    Returns:
        A tuple of the sanitized text and the number of substitutions performed.
    """
    total = 0
    for pattern in _SECRET_PATTERNS:
        text, count = pattern.subn(_replacement, text)
        total += count
    return text, total


def _replacement(match: re.Match[str]) -> str:
    """Keep the key name of ``key=value`` pairs so redacted logs stay readable."""
    value = match.group(0)
    key = re.match(r"(?i)(password|passwd|secret|api[_-]?key|token)\s*[:=]", value)
    if key:
        return f"{key.group(0)} {REDACTION}"
    return REDACTION


def redact(text: str) -> str:
    """Convenience wrapper returning only the redacted text."""
    return redact_secrets(text)[0]


def validate_query(query: str, *, max_chars: int, reject_injection: bool = True) -> str:
    """Normalize and validate an end-user query.

    Args:
        query: Raw user input.
        max_chars: Maximum accepted length after normalization.
        reject_injection: Reject queries that match prompt-injection heuristics.

    Returns:
        The normalized query.

    Raises:
        SecurityError: If the query is empty, oversized or matches an injection rule.
    """
    cleaned = unicodedata.normalize("NFKC", query)
    cleaned = _CONTROL_CHARS.sub(" ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if not cleaned:
        raise SecurityError("query is empty")
    if len(cleaned) > max_chars:
        raise SecurityError(f"query exceeds {max_chars} characters")
    if reject_injection:
        findings = scan_for_injection(cleaned)
        if findings:
            raise SecurityError(f"query rejected by injection screening: {', '.join(findings)}")
    return cleaned
