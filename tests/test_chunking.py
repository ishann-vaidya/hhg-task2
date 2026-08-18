"""Unit tests for Phase 1 chunking strategies."""

import pytest
from src.chunking.base import Chunk, count_tokens, validate_chunks
from src.chunking.strategies import (
    STRATEGIES,
    apply_strategy,
    chunk_fixed_overlap,
    chunk_fixed_size,
    chunk_metadata_aware,
    chunk_semantic,
)


@pytest.fixture
def sample_passage_record():
    return {
        "passage_text": (
            "एक कंपनी एक विशिष्ट देश में निगमित होती है, अक्सर उस देश के एक छोटे उपसमूह, "
            "जैसे कि एक राज्य या प्रांत, की सीमाओं के भीतर। निगम तब उस राज्य में निगमन "
            "के कानूनों द्वारा शासित होता है। एक निगम या तो निजी या सार्वजनिक हो सकता है।"
        ),
        "query_id": 101,
        "passage_index": 0,
        "language": "hi",
        "query_type": "DESCRIPTION",
        "is_selected": True,
    }


def test_all_strategies_registered():
    expected = {"fixed_size", "fixed_overlap", "semantic", "metadata_aware"}
    assert set(STRATEGIES.keys()) == expected


def test_fixed_size_chunking(sample_passage_record):
    chunks = chunk_fixed_size(
        sample_passage_record["passage_text"],
        metadata=sample_passage_record,
        chunk_size=50,
    )
    assert len(chunks) >= 1
    for chunk in chunks:
        assert isinstance(chunk, Chunk)
        assert chunk.strategy == "fixed_size"
        assert chunk.token_count > 0
        assert chunk.passage_id == "q101_p0"


def test_fixed_overlap_chunking(sample_passage_record):
    chunks = chunk_fixed_overlap(
        sample_passage_record["passage_text"],
        metadata=sample_passage_record,
        chunk_size=50,
        overlap=10,
    )
    assert len(chunks) >= 1
    for chunk in chunks:
        assert isinstance(chunk, Chunk)
        assert chunk.strategy == "fixed_overlap"


def test_semantic_chunking(sample_passage_record):
    chunks = chunk_semantic(
        sample_passage_record["passage_text"],
        metadata=sample_passage_record,
    )
    assert len(chunks) >= 1
    for chunk in chunks:
        assert isinstance(chunk, Chunk)
        assert chunk.strategy == "semantic"


def test_metadata_aware_chunking(sample_passage_record):
    chunks = chunk_metadata_aware(
        sample_passage_record["passage_text"],
        metadata=sample_passage_record,
        chunk_size=50,
    )
    assert len(chunks) >= 1
    for chunk in chunks:
        assert isinstance(chunk, Chunk)
        assert chunk.strategy == "metadata_aware"
        assert chunk.query_id == 101
        assert chunk.language == "hi"
        assert chunk.is_selected is True


def test_validate_chunks(sample_passage_record):
    chunks = chunk_fixed_size(sample_passage_record["passage_text"], metadata=sample_passage_record)
    warnings = validate_chunks(chunks)
    assert isinstance(warnings, list)
