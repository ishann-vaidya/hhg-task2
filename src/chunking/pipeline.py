"""Chunking utilities — apply strategies to passages and persist output."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from config.settings import PROCESSED_DATA_DIR
from src.chunking.base import Chunk, make_passage_id, validate_chunks
from src.chunking.strategies import STRATEGIES, apply_strategy


def passage_record_to_metadata(record: dict[str, Any]) -> dict[str, Any]:
    """Build metadata dict from a flat passage row (Phase 0 DataFrame format)."""
    query_id = record.get("query_id")
    passage_index = record.get("passage_index")
    passage_id = make_passage_id(query_id, passage_index)
    return {
        "passage_id": passage_id,
        "query_id": query_id,
        "passage_index": passage_index,
        "language": record.get("language"),
        "query_type": record.get("query_type"),
        "is_selected": record.get("is_selected"),
        "source_doc": passage_id,
    }


def chunk_passage(record: dict[str, Any], strategy: str) -> list[Chunk]:
    text = record.get("passage_text") or record.get("text") or ""
    metadata = passage_record_to_metadata(record)
    return apply_strategy(strategy, text, metadata)


def chunk_passages_df(
    df: pd.DataFrame,
    strategy: str,
    *,
    max_passages: int | None = None,
) -> list[Chunk]:
    """Chunk every passage in a DataFrame with one strategy."""
    all_chunks: list[Chunk] = []
    subset = df.head(max_passages) if max_passages else df
    for _, row in subset.iterrows():
        all_chunks.extend(chunk_passage(row.to_dict(), strategy))
    return all_chunks


def chunk_all_strategies(
    df: pd.DataFrame,
    *,
    max_passages: int | None = None,
) -> dict[str, list[Chunk]]:
    return {
        name: chunk_passages_df(df, name, max_passages=max_passages)
        for name in STRATEGIES
    }


def save_chunks(chunks: list[Chunk], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk.to_dict(), ensure_ascii=False) + "\n")


def load_chunks(input_path: Path) -> list[Chunk]:
    chunks: list[Chunk] = []
    with input_path.open("r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            extra = data.pop("extra", {})
            chunks.append(Chunk(**data, extra=extra))
    return chunks


def persist_strategy_outputs(
    results: dict[str, list[Chunk]],
    output_dir: Path | None = None,
) -> dict[str, Path]:
    """Save each strategy's chunks to data/processed/chunks/<strategy>.jsonl."""
    base = output_dir or (PROCESSED_DATA_DIR / "chunks")
    paths: dict[str, Path] = {}
    for strategy, chunks in results.items():
        path = base / f"{strategy}.jsonl"
        save_chunks(chunks, path)
        paths[strategy] = path
    return paths


def summarize_strategy_results(results: dict[str, list[Chunk]]) -> pd.DataFrame:
    rows = []
    for strategy, chunks in results.items():
        token_counts = [c.token_count for c in chunks]
        rows.append(
            {
                "strategy": strategy,
                "num_chunks": len(chunks),
                "avg_tokens": round(sum(token_counts) / len(token_counts), 1) if token_counts else 0,
                "min_tokens": min(token_counts) if token_counts else 0,
                "max_tokens": max(token_counts) if token_counts else 0,
            }
        )
    return pd.DataFrame(rows)
