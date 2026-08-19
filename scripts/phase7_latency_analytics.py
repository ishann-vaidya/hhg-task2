#!/usr/bin/env python3
"""
Phase 7 — Latency Analytics

Runs a benchmark over a suite of test queries, calculates P50 / P70 / P100 latency,
and outputs a performance report.

Usage:
    python scripts/phase7_latency_analytics.py --n 30
"""

import argparse
import json
import sys
from pathlib import Path

# Force UTF-8 for Windows stdout
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.analytics.latency import summarize_pipeline_latencies  # noqa: E402
from src.data.loader import load_msmarco_subset  # noqa: E402
from src.pipeline import VoiceRAGPipeline  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 7: Run Latency Benchmarking")
    parser.add_argument("--n", type=int, default=30, help="Number of benchmark queries to run")
    parser.add_argument("--strategy", default="metadata_aware", help="Retrieval index strategy to search")
    parser.add_argument("--live-stt", action="store_true", help="Call live Sarvam STT (requires API key)")
    parser.add_argument("--live-gen", action="store_true", help="Call live Groq LLM (requires API key)")
    args = parser.parse_args()

    print("=" * 72)
    print("Phase 7 — Latency Analytics & Benchmarking")
    print("=" * 72)
    print(f"Strategy: {args.strategy}")
    print(f"Number of queries: {args.n}")
    print(f"STT Mode: {'LIVE' if args.live_stt else 'MOCK (Simulated 150ms delay)'}")
    print(f"LLM Mode: {'LIVE' if args.live_gen else 'MOCK (Simulated 120ms delay)'}")
    print()

    # Check index availability
    index_dir = PROJECT_ROOT / "data" / "indexes" / args.strategy
    if not index_dir.exists():
        print(f"Error: FAISS Index not found in {index_dir}")
        print("Please build indices first: python scripts/phase2_build_index.py --n 50")
        sys.exit(1)

    print("Loading benchmark queries from MSMARCO-XI...")
    examples = load_msmarco_subset(n=args.n)
    queries = [ex.get("query") for ex in examples if ex.get("query")]
    print(f"[OK] Extracted {len(queries)} queries.")
    print()

    pipeline = VoiceRAGPipeline(strategy=args.strategy)

    all_latencies = []
    statuses = []

    print("Running benchmarking loop (warmup query first)...")
    pipeline.run_pipeline(
        query_text="वार्मअप प्रश्न",
        mock_stt=not args.live_stt,
        mock_gen=not args.live_gen,
    )

    for i, query in enumerate(queries[: args.n]):
        sys.stdout.write(f"\rProcessing query {i + 1}/{args.n}...")
        sys.stdout.flush()

        res = pipeline.run_pipeline(
            audio_path="dummy.wav",
            mock_stt=not args.live_stt,
            mock_gen=not args.live_gen,
            mock_stt_text=query,
        )

        all_latencies.append(res["latencies"])
        statuses.append(res["status"])

    print("\nBenchmarking complete.\n")

    summary = summarize_pipeline_latencies(all_latencies)

    # Print results table
    print(f"{'Pipeline Step':<20} | {'P50 (ms)':<10} | {'P70 (ms)':<10} | {'P100 (ms)':<10}")
    print("-" * 59)
    for step, metrics in summary.items():
        print(
            f"{step:<20} | {metrics['p50']:<10.2f} | {metrics['p70']:<10.2f} | {metrics['p100']:<10.2f}"
        )

    # Write to data/latency_report.json
    report_path = PROJECT_ROOT / "data" / "latency_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "config": {
                    "strategy": args.strategy,
                    "n": args.n,
                    "live_stt": args.live_stt,
                    "live_gen": args.live_gen,
                },
                "summary": summary,
                "status_counts": {s: statuses.count(s) for s in set(statuses)},
            },
            f,
            indent=2,
        )

    print()
    print(f"[OK] Saved latency analytics report to {report_path}")
    print("=" * 72)


if __name__ == "__main__":
    main()
