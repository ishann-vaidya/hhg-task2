#!/usr/bin/env python3
"""
Phase 2 — Embeddings & FAISS Index

Loads the MSMARCO-XI subset, chunks it using all 4 strategies, embeds the chunks,
and builds FAISS vector databases for each strategy.

Usage:
    python scripts/phase2_build_index.py --n 100
"""

import argparse
import sys
from pathlib import Path

# Force UTF-8 for Windows stdout
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import (  # noqa: E402
    DEFAULT_LANGUAGE,
    DEFAULT_SPLIT,
    INDEX_DIR,
    SUBSET_SIZE,
)
from src.chunking.pipeline import (  # noqa: E402
    chunk_all_strategies,
    persist_strategy_outputs,
    summarize_strategy_results,
)
from src.data.loader import examples_to_passages_df, load_msmarco_subset  # noqa: E402
from src.indexing.indexer import ChunkIndexer  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 2: Build FAISS Vector Indexes")
    parser.add_argument(
        "--n",
        type=int,
        default=SUBSET_SIZE,
        help=f"Number of examples to load and index (default: {SUBSET_SIZE})",
    )
    parser.add_argument(
        "--language",
        default=DEFAULT_LANGUAGE,
        help=f"Language code to load (default: {DEFAULT_LANGUAGE})",
    )
    parser.add_argument(
        "--split",
        default=DEFAULT_SPLIT,
        choices=["train", "validation"],
        help=f"Dataset split to load (default: {DEFAULT_SPLIT})",
    )
    parser.add_argument(
        "--strategies",
        nargs="+",
        default=["fixed_size", "fixed_overlap", "semantic", "metadata_aware"],
        help="List of chunking strategies to index",
    )
    args = parser.parse_args()

    print("=" * 72)
    print("Phase 2 — Building FAISS Indexes")
    print("=" * 72)
    print(f"Loading {args.n} examples for language '{args.language}' split '{args.split}'...")

    if args.language == "en":
        # Load default Hindi validation parquet, but flat-map the English columns
        examples = load_msmarco_subset(language="hi", split=args.split, n=args.n)
        df = examples_to_passages_df(examples, use_translated=False)
    else:
        examples = load_msmarco_subset(language=args.language, split=args.split, n=args.n)
        df = examples_to_passages_df(examples, use_translated=True)
    print(f"[OK] Loaded {len(df)} total passage slots.")

    # 2. Chunking
    print("\nRunning chunking strategies...")
    all_results = chunk_all_strategies(df)

    # Filter strategies based on command arguments
    filtered_results = {s: all_results[s] for s in args.strategies if s in all_results}

    # Print summary
    summary = summarize_strategy_results(filtered_results)
    print("\nChunking summary:")
    print(summary.to_string(index=False))

    # Persist JSONL files
    persist_strategy_outputs(filtered_results)

    # 3. Build FAISS index for each strategy
    print("\nInitializing ChunkIndexer...")
    indexer = ChunkIndexer()

    for strategy, chunks in filtered_results.items():
        print(f"\nBuilding FAISS index for strategy: '{strategy}' ({len(chunks)} chunks)...")
        index, metadata_list = indexer.build_index(chunks)

        output_dir = INDEX_DIR / strategy / args.language
        print(f"Saving index to {output_dir}...")
        indexer.save_index(index, metadata_list, output_dir)
        print(f"[OK] Index saved successfully (FAISS size: {index.ntotal} vectors).")

    print("\n" + "=" * 72)
    print("Phase 2 PASS — Vector databases built and saved.")
    print("Next: Phase 3 — retrieval query baseline.")
    print("=" * 72)


if __name__ == "__main__":
    main()
