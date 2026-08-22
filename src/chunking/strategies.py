"""Four chunking strategies for the RAG pipeline."""

from __future__ import annotations

import re
from typing import Any

import numpy as np

from config.settings import (
    CHUNK_OVERLAP_TOKENS,
    CHUNK_SIZE_TOKENS,
    SEMANTIC_EMBEDDING_MODEL,
    SEMANTIC_MAX_CHUNK_TOKENS,
    SEMANTIC_SIMILARITY_THRESHOLD,
)
from src.chunking.base import Chunk, count_tokens, make_passage_id

# Sentence boundaries: Latin and Indic punctuation (। is Devanagari danda)
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?।\n])\s+")

# Lazy-loaded embedding model (only needed for semantic chunking)
_EMBEDDING_MODEL = None


def _split_sentences(text: str) -> list[str]:
    parts = _SENTENCE_SPLIT_RE.split(text.strip())
    return [p.strip() for p in parts if p.strip()]


def _get_embedding_model():
    global _EMBEDDING_MODEL
    if _EMBEDDING_MODEL is None:
        from sentence_transformers import SentenceTransformer

        _EMBEDDING_MODEL = SentenceTransformer(SEMANTIC_EMBEDDING_MODEL)
    return _EMBEDDING_MODEL


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def _wrap_raw_chunks(
    raw_texts: list[str],
    *,
    strategy: str,
    metadata: dict[str, Any],
) -> list[Chunk]:
    passage_id = metadata.get("passage_id") or make_passage_id(
        metadata.get("query_id"), metadata.get("passage_index")
    )
    chunks: list[Chunk] = []
    for i, text in enumerate(raw_texts):
        text = text.strip()
        if not text:
            continue
        chunks.append(
            Chunk(
                text=text,
                chunk_index=i,
                strategy=strategy,
                token_count=count_tokens(text),
                passage_id=passage_id,
                query_id=metadata.get("query_id"),
                passage_index=metadata.get("passage_index"),
                language=metadata.get("language"),
                query_type=metadata.get("query_type"),
                is_selected=metadata.get("is_selected"),
                source_doc=metadata.get("source_doc"),
                extra=dict(metadata.get("extra") or {}),
            )
        )
    return chunks


def chunk_fixed_size(
    text: str,
    metadata: dict[str, Any] | None = None,
    *,
    chunk_size: int = CHUNK_SIZE_TOKENS,
) -> list[Chunk]:
    """
    Strategy 1 — Fixed-size chunks with no overlap (baseline).

    Uses LangChain's RecursiveCharacterTextSplitter with token counting.
    """
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    metadata = metadata or {}
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=0,
        length_function=count_tokens,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    raw = splitter.split_text(text)
    return _wrap_raw_chunks(raw, strategy="fixed_size", metadata=metadata)


def chunk_fixed_overlap(
    text: str,
    metadata: dict[str, Any] | None = None,
    *,
    chunk_size: int = CHUNK_SIZE_TOKENS,
    overlap: int = CHUNK_OVERLAP_TOKENS,
) -> list[Chunk]:
    """
    Strategy 2 — Fixed-size chunks with token overlap.

    Overlap helps retrieval catch facts that would otherwise sit on a chunk boundary.
    """
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    metadata = metadata or {}
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        length_function=count_tokens,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    raw = splitter.split_text(text)
    return _wrap_raw_chunks(raw, strategy="fixed_overlap", metadata=metadata)


def chunk_semantic(
    text: str,
    metadata: dict[str, Any] | None = None,
    *,
    similarity_threshold: float = SEMANTIC_SIMILARITY_THRESHOLD,
    max_chunk_tokens: int = SEMANTIC_MAX_CHUNK_TOKENS,
) -> list[Chunk]:
    """
    Strategy 3 — Semantic chunking.

    1. Split text into sentences.
    2. Embed each sentence locally (sentence-transformers — no API key).
    3. Start a new chunk when adjacent-sentence similarity drops below threshold
       OR the running chunk would exceed max_chunk_tokens.
    """
    metadata = metadata or {}
    sentences = _split_sentences(text)

    if not sentences:
        return []

    if len(sentences) == 1:
        return _wrap_raw_chunks(sentences, strategy="semantic", metadata=metadata)

    model = _get_embedding_model()
    embeddings = model.encode(sentences, normalize_embeddings=True)

    groups: list[list[str]] = [[sentences[0]]]
    group_tokens = count_tokens(sentences[0])

    for i in range(1, len(sentences)):
        sim = _cosine_similarity(embeddings[i - 1], embeddings[i])
        next_tokens = count_tokens(sentences[i])
        would_exceed = group_tokens + next_tokens > max_chunk_tokens
        topic_shift = sim < similarity_threshold

        if topic_shift or would_exceed:
            groups.append([sentences[i]])
            group_tokens = next_tokens
        else:
            groups[-1].append(sentences[i])
            group_tokens += next_tokens

    raw = [" ".join(group) for group in groups]
    return _wrap_raw_chunks(raw, strategy="semantic", metadata=metadata)


def chunk_metadata_aware(
    text: str,
    metadata: dict[str, Any] | None = None,
    *,
    chunk_size: int = CHUNK_SIZE_TOKENS,
    overlap: int = CHUNK_OVERLAP_TOKENS,
) -> list[Chunk]:
    """
    Strategy 4 — Metadata-aware chunking.

    Chunks at natural paragraph/sentence boundaries first, then applies
    size limits. Every chunk carries full passage metadata (ID, language,
    query type, selection flag) for citations and guardrails later.
    """
    metadata = dict(metadata or {})
    metadata.setdefault(
        "passage_id",
        make_passage_id(metadata.get("query_id"), metadata.get("passage_index")),
    )
    metadata.setdefault("source_doc", metadata.get("passage_id"))

    # Prefer paragraph boundaries, then sentences — keeps metadata aligned to source
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text.strip()) if p.strip()]
    if not paragraphs:
        paragraphs = [text.strip()]

    raw_chunks: list[str] = []
    buffer = ""

    def flush_buffer() -> None:
        nonlocal buffer
        if buffer.strip():
            raw_chunks.append(buffer.strip())
        buffer = ""

    for para in paragraphs:
        para_tokens = count_tokens(para)
        if para_tokens <= chunk_size:
            candidate = f"{buffer} {para}".strip() if buffer else para
            if count_tokens(candidate) <= chunk_size:
                buffer = candidate
            else:
                flush_buffer()
                buffer = para
        else:
            # Paragraph too large — sentence-level split with overlap
            flush_buffer()
            sentences = _split_sentences(para)
            sent_buffer = ""
            for sentence in sentences:
                candidate = f"{sent_buffer} {sentence}".strip() if sent_buffer else sentence
                if count_tokens(candidate) <= chunk_size:
                    sent_buffer = candidate
                else:
                    if sent_buffer:
                        raw_chunks.append(sent_buffer)
                    sent_buffer = sentence
            if sent_buffer:
                raw_chunks.append(sent_buffer)

    flush_buffer()

    # Add overlap between consecutive chunks for retrieval continuity
    if overlap > 0 and len(raw_chunks) > 1:
        overlapped: list[str] = [raw_chunks[0]]
        for i in range(1, len(raw_chunks)):
            prev_words = raw_chunks[i - 1].split()
            # Approximate overlap tokens → words (rough heuristic)
            tail_words = prev_words[-max(overlap, 1) :]
            prefix = " ".join(tail_words)
            overlapped.append(f"{prefix} {raw_chunks[i]}".strip())
        raw_chunks = overlapped

    chunks = _wrap_raw_chunks(raw_chunks, strategy="metadata_aware", metadata=metadata)
    for chunk in chunks:
        chunk.extra.update(
            {
                "chunking_notes": "paragraph-first with sentence fallback",
                "parent_passage_id": metadata["passage_id"],
            }
        )
    return chunks


STRATEGIES = {
    "fixed_size": chunk_fixed_size,
    "fixed_overlap": chunk_fixed_overlap,
    "semantic": chunk_semantic,
    "metadata_aware": chunk_metadata_aware,
}


def apply_strategy(name: str, text: str, metadata: dict[str, Any] | None = None) -> list[Chunk]:
    if name not in STRATEGIES:
        raise ValueError(f"Unknown strategy '{name}'. Choose from: {list(STRATEGIES)}")
    return STRATEGIES[name](text, metadata)
