"""Load and explore the ai4bharat/MSMARCO-XI dataset."""

from __future__ import annotations

from typing import Any

import pandas as pd
from datasets import Dataset, load_dataset

from config.settings import (
    DATASET_NAME,
    DEFAULT_LANGUAGE,
    DEFAULT_SPLIT,
    LANGUAGE_FILE_PREFIX,
    SUBSET_SIZE,
)


def _parquet_filename(language: str, split: str) -> str:
    """Return the HF parquet path for a language/split, e.g. validation/hinval.parquet."""
    prefix = LANGUAGE_FILE_PREFIX[language]
    suffix = "train" if split == "train" else "val"
    return f"{split}/{prefix}{suffix}.parquet"


def load_msmarco_subset(
    language: str = DEFAULT_LANGUAGE,
    split: str = DEFAULT_SPLIT,
    n: int = SUBSET_SIZE,
    *,
    streaming: bool = False,
) -> Dataset:
    """
    Load the first *n* examples from MSMARCO-XI for a given language/split.

    The dataset on Hugging Face is stored as one parquet file per language
    (e.g. ``validation/hinval.parquet`` for Hindi validation). We load that
    file directly so we don't have to scan the full 55 GB corpus.

    Parameters
    ----------
    language:
        Short code, e.g. ``"hi"`` for Hindi.
    split:
        ``"train"`` or ``"validation"``.
    n:
        Maximum number of examples to keep.
    streaming:
        If True, stream rows from the parquet file (lower memory, slightly slower).
    """
    if language not in LANGUAGE_FILE_PREFIX:
        raise ValueError(
            f"Unknown language '{language}'. Choose from: {list(LANGUAGE_FILE_PREFIX)}"
        )

    parquet_path = _parquet_filename(language, split)
    data_files = {split: parquet_path}

    if streaming:
        ds = load_dataset(
            DATASET_NAME,
            data_files=data_files,
            split=split,
            streaming=True,
        )
        rows: list[dict[str, Any]] = []
        for i, row in enumerate(ds):
            if i >= n:
                break
            rows.append(dict(row))
        return Dataset.from_list(rows)

    ds = load_dataset(DATASET_NAME, data_files=data_files, split=split)
    return ds.select(range(min(n, len(ds))))


def extract_passages_from_example(
    example: dict[str, Any],
    *,
    use_translated: bool = True,
) -> list[dict[str, Any]]:
    """
    Flatten one MSMARCO-XI example into individual passage records.

    Each record carries metadata needed later for citations and guardrails.
    """
    passages_block = example.get("passages") or {}
    text_key = "Translated_passages" if use_translated else "English_passages"
    texts: list[str] = passages_block.get(text_key) or []
    selected_flags: list[int] = passages_block.get("is_selected") or []

    records: list[dict[str, Any]] = []
    for idx, text in enumerate(texts):
        if not text or not str(text).strip():
            continue
        records.append(
            {
                "query_id": example.get("query_id"),
                "passage_index": idx,
                "is_selected": bool(selected_flags[idx]) if idx < len(selected_flags) else False,
                "language": example.get("target_lang") if use_translated else "eng_Latn",
                "query": example.get("query"),
                "query_type": example.get("query_type"),
                "passage_text": text,
                "answer": example.get("Answer"),
                "eng_query": example.get("Eng_Query"),
                "eng_answer": example.get("Eng_Answer"),
            }
        )
    return records


def examples_to_passages_df(
    examples: Dataset | list[dict[str, Any]],
    *,
    use_translated: bool = True,
) -> pd.DataFrame:
    """Convert a batch of examples into a flat passages DataFrame."""
    all_records: list[dict[str, Any]] = []
    for ex in examples:
        all_records.extend(extract_passages_from_example(ex, use_translated=use_translated))
    return pd.DataFrame(all_records)


def summarize_dataset(examples: Dataset) -> dict[str, Any]:
    """Return high-level stats for a loaded subset."""
    if len(examples) == 0:
        return {"num_examples": 0}

    first = examples[0]
    passage_counts = []
    selected_counts = []
    query_types: dict[str, int] = {}

    for ex in examples:
        passages = ex.get("passages") or {}
        texts = passages.get("Translated_passages") or []
        flags = passages.get("is_selected") or []
        passage_counts.append(len(texts))
        selected_counts.append(sum(1 for f in flags if f == 1))
        qt = ex.get("query_type") or "UNKNOWN"
        query_types[qt] = query_types.get(qt, 0) + 1

    return {
        "num_examples": len(examples),
        "language": first.get("target_lang"),
        "source_lang": first.get("source_lang"),
        "avg_passages_per_query": round(sum(passage_counts) / len(passage_counts), 2),
        "avg_selected_passages": round(sum(selected_counts) / len(selected_counts), 2),
        "total_passage_slots": sum(passage_counts),
        "query_type_counts": query_types,
        "sample_columns": list(examples.column_names),
    }
