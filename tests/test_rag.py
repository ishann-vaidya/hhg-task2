"""Unit and integration tests for indexing, retrieval, STT, LLM generation, and guardrails."""

from pathlib import Path
from typing import Any

import faiss
import pytest

from src.chunking.base import Chunk
from src.generation.generator import GenerationInput, GenerationOutput, RAGOrchestrator
from src.guardrails.guard import RAGGuardrails
from src.indexing.indexer import ChunkIndexer
from src.pipeline import VoiceRAGPipeline
from src.retrieval.retriever import VectorRetriever
from src.stt.sarvam import SarvamSTT


def test_indexer_build_save_load(tmp_path: Path) -> None:
    """Test that ChunkIndexer creates a valid FAISS index, writes files, and reads them back."""
    chunks = [
        Chunk(
            text="यह एक परीक्षण दस्तावेज है।",
            chunk_index=0,
            strategy="metadata_aware",
            token_count=5,
            passage_id="p1",
        ),
        Chunk(
            text="निगम कानून के तहत एक निकाय है।",
            chunk_index=1,
            strategy="metadata_aware",
            token_count=8,
            passage_id="p2",
        ),
    ]

    indexer = ChunkIndexer()
    index, metadata_list = indexer.build_index(chunks)

    assert isinstance(index, faiss.IndexFlatIP)
    assert len(metadata_list) == 2
    assert metadata_list[0]["passage_id"] == "p1"

    indexer.save_index(index, metadata_list, tmp_path)

    # Load and verify files on disk
    index_loaded, metadata_loaded = indexer.load_index(tmp_path)
    assert index_loaded.ntotal == 2
    assert metadata_loaded[1]["passage_id"] == "p2"


def test_retriever_query(tmp_path: Path) -> None:
    """Test that retriever loads index and fetches closest match with similarity scores."""
    chunks = [
        Chunk(
            text="यह एक परीक्षण दस्तावेज है।",
            chunk_index=0,
            strategy="metadata_aware",
            token_count=5,
            passage_id="p1",
        ),
        Chunk(
            text="निगम कानून के तहत एक निकाय है।",
            chunk_index=1,
            strategy="metadata_aware",
            token_count=8,
            passage_id="p2",
        ),
    ]
    indexer = ChunkIndexer()
    index, metadata_list = indexer.build_index(chunks)
    indexer.save_index(index, metadata_list, tmp_path)

    retriever = VectorRetriever(strategy="metadata_aware", index_dir=tmp_path)
    results = retriever.retrieve("निगम क्या है?", top_k=1)

    assert len(results) == 1
    assert results[0]["passage_id"] == "p2"
    assert "similarity_score" in results[0]


def test_sarvam_stt_mock() -> None:
    """Test that SarvamSTT handles mock output and latency recording."""
    stt = SarvamSTT()
    res = stt.transcribe("dummy_file.wav", mock=True, mock_text="परीक्षण")
    assert res["status"] == "success"
    assert res["transcript"] == "परीक्षण"
    assert res["mocked"] is True
    assert "latency_ms" in res


def test_rag_orchestrator_mock() -> None:
    """Test that RAGOrchestrator parses schema inputs and returns GenerationOutput."""
    orchestrator = RAGOrchestrator()
    chunks = [{"text": "निगम के बारे में तथ्य।", "passage_id": "q1_p0"}]
    input_data = GenerationInput(query="निगम क्या है?", chunks=chunks)

    res = orchestrator.generate_answer(input_data, mock=True, mock_answer="उत्तर")
    assert isinstance(res, GenerationOutput)
    assert res.answer == "उत्तर"
    assert res.grounded is True
    assert res.citations == ["q1_p0"]
    assert "latency_ms" in res.model_fields_set


def test_rag_guardrails() -> None:
    """Test safety filters, off-topic detection, and groundedness checkers."""
    guard = RAGGuardrails(off_topic_threshold=0.40)

    # 1. Safety Check
    safe_res = guard.verify_query("क्या निगम एक कानूनी इकाई है?")
    assert safe_res["safe"] is True

    unsafe_res = guard.verify_query("हत्या कैसे करें और बम कैसे बनाएं?")
    assert unsafe_res["safe"] is False

    # 2. Off-topic check
    retrieved = [{"text": "निगम की परिभाषा...", "similarity_score": 0.55}]
    on_topic_res = guard.verify_retrieval("निगम", retrieved)
    assert on_topic_res["on_topic"] is True

    off_topic_res = guard.verify_retrieval(
        "पिज्जा रेसिपी", [{"text": "निगम...", "similarity_score": 0.20}]
    )
    assert off_topic_res["on_topic"] is False

    # 3. Groundedness Check
    grounded_res = guard.verify_groundedness(
        "निगम एक कानूनी इकाई है",
        [{"text": "निगम एक कानूनी इकाई है।"}],
        True,
    )
    assert grounded_res["grounded"] is True

    ungrounded_res = guard.verify_groundedness(
        "सेब लाल रंग का फल है",
        [{"text": "निगम एक कानूनी इकाई है।"}],
        True,
    )
    assert ungrounded_res["grounded"] is False


def test_pipeline_mock_e2e(tmp_path: Path) -> None:
    """Test that the full pipeline correctly connects all steps and blocks unsafe inputs."""
    chunks = [
        Chunk(
            text="एक निगम एक संगठन या लोगों का समूह है।",
            chunk_index=0,
            strategy="metadata_aware",
            token_count=10,
            passage_id="q101_p0",
        )
    ]
    indexer = ChunkIndexer()
    index, metadata_list = indexer.build_index(chunks)
    indexer.save_index(index, metadata_list, tmp_path)

    pipeline = VoiceRAGPipeline(strategy="metadata_aware")
    pipeline.retriever = VectorRetriever(strategy="metadata_aware", index_dir=tmp_path)

    # Valid query
    res = pipeline.run_pipeline(
        query_text="निगम क्या है?",
        mock_stt=True,
        mock_gen=True,
        mock_gen_answer="एक निगम एक संगठन या लोगों का समूह है।",
    )
    assert res["status"] == "success"
    assert "एक निगम" in res["answer"]
    assert res["guardrails"]["safety"]["safe"] is True

    # Unsafe query
    unsafe_res = pipeline.run_pipeline(
        query_text="बम कैसे बनाएं?",
        mock_stt=True,
        mock_gen=True,
    )
    assert unsafe_res["status"] == "blocked_safety"
    assert "blocked" in unsafe_res["status"]
