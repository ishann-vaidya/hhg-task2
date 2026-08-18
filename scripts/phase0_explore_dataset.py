#!/usr/bin/env python3
"""
Phase 0 — Site Prep: Environment & Data

Loads a subset of ai4bharat/MSMARCO-XI, prints structure/stats, and shows samples.

Usage (from project root, with venv activated):
    python scripts/phase0_explore_dataset.py
    python scripts/phase0_explore_dataset.py --language hi --split validation --n 100
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Windows terminals often default to cp1252 — force UTF-8 for Indic script output
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Allow imports from project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import (  # noqa: E402
    AVAILABLE_LANGUAGES,
    DATASET_NAME,
    DEFAULT_LANGUAGE,
    DEFAULT_SPLIT,
    SUBSET_SIZE,
)
from src.data.loader import (  # noqa: E402
    examples_to_passages_df,
    load_msmarco_subset,
    summarize_dataset,
)


def _truncate(text: str, max_len: int = 120) -> str:
    text = " ".join(text.split())
    return text if len(text) <= max_len else text[: max_len - 3] + "..."


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 0: explore MSMARCO-XI subset")
    parser.add_argument("--language", default=DEFAULT_LANGUAGE, choices=AVAILABLE_LANGUAGES)
    parser.add_argument("--split", default=DEFAULT_SPLIT, choices=["train", "validation"])
    parser.add_argument("--n", type=int, default=min(50, SUBSET_SIZE), help="Examples to load (default: 50 for quick test)")
    parser.add_argument("--streaming", action="store_true", help="Stream from HF instead of downloading split")
    args = parser.parse_args()

    print("=" * 70)
    print("Phase 0 — MSMARCO-XI Dataset Exploration")
    print("=" * 70)
    print(f"Dataset : {DATASET_NAME}")
    print(f"Language: {args.language}  |  Split: {args.split}  |  N: {args.n}")
    print(f"Full dataset is ~55 GB across all languages — we use a small subset for the hackathon.")
    print(f"Configured hackathon subset size (config/settings.py): {SUBSET_SIZE:,} examples")
    print()

    print("Loading dataset (first run downloads from Hugging Face — may take a few minutes)...")
    examples = load_msmarco_subset(
        language=args.language,
        split=args.split,
        n=args.n,
        streaming=args.streaming,
    )
    print(f"[OK] Loaded {len(examples)} examples\n")

    # ── Structure summary ────────────────────────────────────────────────
    summary = summarize_dataset(examples)
    print("── Dataset structure ──")
    print(f"  Columns       : {summary['sample_columns']}")
    print(f"  Target language: {summary.get('language')}")
    print(f"  Source language: {summary.get('source_lang')}")
    print(f"  Avg passages/query: {summary.get('avg_passages_per_query')}")
    print(f"  Avg selected passages (relevant): {summary.get('avg_selected_passages')}")
    print(f"  Total passage slots in subset: {summary.get('total_passage_slots')}")
    print(f"  Query types   : {summary.get('query_type_counts')}")
    print()

    # ── Sample queries ───────────────────────────────────────────────────
    print("── Sample queries (translated + English) ──")
    for i in range(min(3, len(examples))):
        ex = examples[i]
        print(f"\n  [{i}] query_id={ex['query_id']}  type={ex['query_type']}")
        print(f"      Query (translated): {_truncate(ex['query'])}")
        print(f"      Query (English)   : {_truncate(ex.get('Eng_Query') or '')}")
        print(f"      Answer (translated): {_truncate(ex.get('Answer') or '')}")

    # ── Sample passages ──────────────────────────────────────────────────
    print("\n── Sample passages from first example ──")
    ex0 = examples[0]
    passages = ex0["passages"]
    for j in range(min(3, len(passages["Translated_passages"]))):
        sel = passages["is_selected"][j]
        print(f"\n  passage[{j}]  selected={bool(sel)}")
        print(f"    Translated: {_truncate(passages['Translated_passages'][j], 160)}")
        print(f"    English   : {_truncate(passages['English_passages'][j], 160)}")

    # ── Flat passages DataFrame ──────────────────────────────────────────
    df = examples_to_passages_df(examples)
    print("\n── Flat passages DataFrame ──")
    print(f"  Shape: {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"  Columns: {list(df.columns)}")
    print(f"  Selected (relevant) passages: {df['is_selected'].sum()} / {len(df)}")
    print("\n  First 3 rows:")
    print(df[["query_id", "passage_index", "is_selected", "query_type"]].head(3).to_string(index=False))

    print("\n" + "=" * 70)
    print("Phase 0 PASS — You can load passages and inspect samples.")
    print("Next: Phase 1 — implement multiple chunking strategies.")
    print("=" * 70)


if __name__ == "__main__":
    main()
