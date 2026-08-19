#!/usr/bin/env python3
"""
Phase 3 — Retrieval & Latency Baseline

Runs retrieval queries against the FAISS index and prints the retrieved chunks
along with similarity scores and latency metrics.

Usage:
    python scripts/phase3_retrieve.py --query "कॉर्पोरेशन क्या है?" --strategy metadata_aware
"""

import argparse
import sys
import time
from pathlib import Path

# Force UTF-8 for Windows stdout
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.retrieval.retriever import VectorRetriever  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 3: Query FAISS retrieval")
    parser.add_argument("--query", default="कॉर्पोरेशन क्या है?", help="Query to run")
    parser.add_argument("--strategy", default="metadata_aware", help="Chunking strategy index to search")
    parser.add_argument("--top_k", type=int, default=3, help="Top k results to retrieve")
    args = parser.parse_args()

    print("=" * 72)
    print("Phase 3 — Vector Database Retrieval")
    print("=" * 72)
    print(f"Strategy: {args.strategy}")
    print(f"Query   : {args.query}")
    print()

    try:
        retriever = VectorRetriever(strategy=args.strategy)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please build the indexes first by running scripts/phase2_build_index.py")
        sys.exit(1)

    # Run multiple times to benchmark retrieval latency (warmup first)
    print("Running warmup search...")
    retriever.retrieve(args.query, top_k=args.top_k)

    print("Running benchmark search...")
    latencies = []
    results = []
    for _ in range(5):
        start = time.perf_counter()
        results = retriever.retrieve(args.query, top_k=args.top_k)
        latencies.append((time.perf_counter() - start) * 1000)

    avg_latency = sum(latencies) / len(latencies)
    print(f"[OK] Retrieved {len(results)} chunks.")
    print(
        f"Benchmark Latency (across 5 runs): {avg_latency:.2f} ms (Min: {min(latencies):.2f} ms, Max: {max(latencies):.2f} ms)"
    )
    print()

    print("── Retrieved Results ──")
    for i, res in enumerate(results):
        print(f"\n[{i}] Similarity: {res['similarity_score']:.4f}  |  id: {res['passage_id']}")
        print(f"    Text: {res['text']}")
        print(f"    Parent: {res.get('source_doc')}  | Strategy: {res['strategy']}")

    print("\n" + "=" * 72)
    print(f"Phase 3 PASS — Vector database retrieval functional. Average latency: {avg_latency:.2f} ms")
    print("Next: Phase 4 — speech-to-text integration.")
    print("=" * 72)


if __name__ == "__main__":
    main()
