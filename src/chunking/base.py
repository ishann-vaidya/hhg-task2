"""Text chunk data model and token utilities."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import tiktoken

from config.settings import CHUNK_SIZE_TOKENS

# Lazy singleton — tiktoken encodings are expensive to create
_ENCODER: tiktoken.Encoding | None = None


def get_token_encoder() -> tiktoken.Encoding:
    global _ENCODER
    if _ENCODER is None:
        _ENCODER = tiktoken.get_encoding("cl100k_base")
    return _ENCODER


def count_tokens(text: str) -> int:
    """Approximate token count (works reasonably for English + Indic text)."""
    if not text:
        return 0
    return len(get_token_encoder().encode(text))


@dataclass
class Chunk:
    """One chunk of text with metadata for retrieval, citations, and guardrails."""

    text: str
    chunk_index: int
    strategy: str
    token_count: int

    # Document / passage metadata
    passage_id: str
    query_id: int | None = None
    passage_index: int | None = None
    language: str | None = None
    query_type: str | None = None
    is_selected: bool | None = None
    source_doc: str | None = None

    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def make_passage_id(query_id: int | None, passage_index: int | None) -> str:
    return f"q{query_id}_p{passage_index}"


def validate_chunks(chunks: list[Chunk], *, min_tokens: int = 3) -> list[str]:
    """Return human-readable warnings if chunks look malformed."""
    warnings: list[str] = []
    if not chunks:
        warnings.append("No chunks produced.")
        return warnings

    for chunk in chunks:
        if chunk.token_count < min_tokens:
            warnings.append(
                f"Chunk {chunk.chunk_index} ({chunk.strategy}) is very short: "
                f"{chunk.token_count} tokens — '{chunk.text[:40]}...'"
            )
        if chunk.token_count > CHUNK_SIZE_TOKENS * 3:
            warnings.append(
                f"Chunk {chunk.chunk_index} ({chunk.strategy}) is unusually large: "
                f"{chunk.token_count} tokens."
            )
    return warnings
