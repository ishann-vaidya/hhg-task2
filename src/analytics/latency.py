"""Latency analytics helpers to compute P50, P70, and P100 percentiles."""


def calculate_percentiles(times: list[float]) -> dict[str, float]:
    """Calculate P50, P70, and P100 percentiles for a list of millisecond durations."""
    if not times:
        return {"p50": 0.0, "p70": 0.0, "p100": 0.0}

    times_sorted = sorted(times)
    n = len(times_sorted)

    def get_p(p: float) -> float:
        idx = int(round(p * (n - 1)))
        return float(times_sorted[max(0, min(idx, n - 1))])

    return {
        "p50": round(get_p(0.50), 2),
        "p70": round(get_p(0.70), 2),
        "p100": round(get_p(1.00), 2),
    }


def summarize_pipeline_latencies(
    all_latencies: list[dict[str, float]],
) -> dict[str, dict[str, float]]:
    """Compute percentiles for each step and total duration across multiple test runs."""
    keys = [
        "stt",
        "safety_guard",
        "retrieval",
        "off_topic_guard",
        "generation",
        "groundedness_guard",
        "total",
    ]
    summary = {}
    for key in keys:
        times = [run[key] for run in all_latencies if key in run]
        summary[key] = calculate_percentiles(times)
    return summary
