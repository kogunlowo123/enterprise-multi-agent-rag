"""Shared text-processing primitives: tokenization, stemming, sentence splitting."""

from __future__ import annotations

import re

_STOPWORD_TEXT = (
    "a about above after again against all also am an and any are as at be because been "
    "before being below between both but by can could did do does doing down during each "
    "few for from further had has have having he her here hers him his how i if in into "
    "is it its just me more most my no nor not of off on once only or other our out over "
    "own same she should so may must shall some such than that the their them then there "
    "these they this those through to too under until up very was we were what when where "
    "which while who whom why will with would you your "
)
STOPWORDS = frozenset(_STOPWORD_TEXT.split())

CITATION = re.compile(r"\[\d+(?:\s*,\s*\d+)*\]")
_WORD = re.compile(r"[a-z0-9]+(?:[.,][0-9]+)*")
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+(?=[\"'(\[A-Z0-9])")
_TRAILING_CITATIONS = re.compile(
    r"([.!?])\s*(\[\d+(?:\s*,\s*\d+)*\](?:\s*\[\d+(?:\s*,\s*\d+)*\])*)"
)


def stem(token: str) -> str:
    """Apply a deliberately light suffix stripper.

    The same function is applied to queries and documents, so consistency matters
    more than linguistic accuracy.
    """
    if len(token) > 4 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 5 and token.endswith("ing"):
        return token[:-3]
    if len(token) > 4 and token.endswith("ed"):
        return token[:-2]
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def tokenize(text: str) -> list[str]:
    """Lowercase ``text`` and split it into alphanumeric tokens (numbers keep decimals)."""
    return _WORD.findall(text.lower())


def content_tokens(text: str) -> list[str]:
    """Tokenize, drop stopwords and stem. Order and duplicates are preserved."""
    return [stem(tok) for tok in tokenize(text) if tok not in STOPWORDS]


def split_sentences(text: str) -> list[str]:
    """Split ``text`` into sentences, treating blank lines and headings as boundaries."""
    sentences: list[str] = []
    for block in re.split(r"\n\s*\n", text):
        block = " ".join(line.strip() for line in block.splitlines() if line.strip())
        if not block:
            continue
        if block.startswith("#"):
            sentences.append(block)
            continue
        sentences.extend(part.strip() for part in _SENTENCE_BOUNDARY.split(block) if part.strip())
    return sentences


def normalize_citations(text: str) -> str:
    """Move citation markers that trail sentence punctuation in front of it.

    ``"The limit is 5. [1]"`` becomes ``"The limit is 5 [1]."`` so each sentence
    carries its citations and can be validated independently.
    """
    return _TRAILING_CITATIONS.sub(lambda m: f" {m.group(2)}{m.group(1)}", text)


def parse_citations(text: str) -> list[int]:
    """Return the citation indices found in ``text`` in order of appearance."""
    indices: list[int] = []
    for marker in CITATION.findall(text):
        indices.extend(int(part) for part in re.findall(r"\d+", marker))
    return indices
