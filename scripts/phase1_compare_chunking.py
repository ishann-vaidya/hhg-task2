#!/usr/bin/env python3
"""
Phase 1 — Chunking Strategies

Compares four chunking approaches side-by-side on real MSMARCO-XI passages.

Usage (from project root, venv activated):
    python scripts/phase1_compare_chunking.py
    python scripts/phase1_compare_chunking.py --passages 5 --save
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import (  # noqa: E402
    CHUNK_OVERLAP_TOKENS,
    CHUNK_SIZE_TOKENS,
    DEFAULT_LANGUAGE,
    DEFAULT_SPLIT,
)
from src.chunking import (  # noqa: E402
    STRATEGIES,
    chunk_all_strategies,
    chunk_passage,
    persist_strategy_outputs,
    summarize_strategy_results,
    validate_chunks,
)
from src.data.loader import examples_to_passages_df, load_msmarco_subset  # noqa: E402


def _preview(text: str, max_len: int = 100) -> str:
    text = " ".join(text.split())
    return text if len(text) <= max_len else text[: max_len - 3] + "..."


def _print_passage_header(record: dict, index: int) -> None:
    print(f"\n{'=' * 72}")
    print(f"Passage {index}  |  id=q{record['query_id']}_p{record['passage_index']}")
    print(f"  language={record['language']}  selected={record['is_selected']}  type={record['query_type']}")
    print(f"  text ({len(record['passage_text'])} chars): {_preview(record['passage_text'], 140)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 1: compare chunking strategies")
    parser.add_argument("--language", default=DEFAULT_LANGUAGE)
    parser.add_argument("--split", default=DEFAULT_SPLIT)
    parser.add_argument("--passages", type=int, default=3, help="Number of sample passages to chunk")
    parser.add_argument("--save", action="store_true", help="Save chunks to data/processed/chunks/")
    args = parser.parse_args()

    print("=" * 72)
    print("Phase 1 — Chunking Strategy Comparison")
    print("=" * 72)
    print(f"Config: chunk_size={CHUNK_SIZE_TOKENS} tokens, overlap={CHUNK_OVERLAP_TOKENS} tokens")
    print(f"Strategies: {', '.join(STRATEGIES)}")
    print()
    print("Note: semantic chunking downloads a small local model on first run (~90 MB).")
    print("      No Sarvam or Groq API keys needed for this phase.\n")

    print("Loading sample passages...")
    examples = load_msmarco_subset(language=args.language, split=args.split, n=10)
    df = examples_to_passages_df(examples)
    sample_df = df.head(args.passages)
    print(f"[OK] Loaded {len(sample_df)} passages\n")

    # ── Per-passage side-by-side comparison ───────────────────────────────
    for i, row in enumerate(sample_df.to_dict("records")):
        _print_passage_header(row, i)
        print(f"\n  {'Strategy':<18} {'#Chunks':>8} {'Tokens (min/avg/max)':>24}")
        print(f"  {'-' * 54}")

        for name in STRATEGIES:
            chunks = chunk_passage(row, name)
            if not chunks:
                print(f"  {name:<18} {0:>8} {'—':>24}")
                continue
            tokens = [c.token_count for c in chunks]
            stats = f"{min(tokens)}/{sum(tokens)//len(tokens)}/{max(tokens)}"
            print(f"  {name:<18} {len(chunks):>8} {stats:>24}")

            for j, chunk in enumerate(chunks[:2]):  # show first 2 chunks per strategy
                print(f"      [{j}] ({chunk.token_count} tok) {_preview(chunk.text, 90)}")
            if len(chunks) > 2:
                print(f"      ... +{len(chunks) - 2} more chunks")

            warnings = validate_chunks(chunks)
            for w in warnings:
                print(f"      WARNING: {w}")

    # ── Aggregate stats across all sample passages ────────────────────────
    print(f"\n{'=' * 72}")
    print(f"Aggregate stats across {args.passages} passages")
    print("=" * 72)
    all_results = chunk_all_strategies(sample_df, max_passages=args.passages)
    summary = summarize_strategy_results(all_results)
    print(summary.to_string(index=False))

    if args.save:
        paths = persist_strategy_outputs(all_results)
        print("\nSaved chunk files:")
        for strategy, path in paths.items():
            print(f"  {strategy}: {path}")

    print(f"\n{'=' * 72}")
    print("Phase 1 PASS — Four chunking strategies implemented and compared.")
    print("Next: Phase 2 — embed chunks and build a FAISS index.")
    print("=" * 72)


if __name__ == "__main__":
    main()
