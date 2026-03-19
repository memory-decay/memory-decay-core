"""Check if current search surface is exhausted and expansion is needed."""

import json
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parent.parent
HISTORY = ROOT / "experiments" / "history.jsonl"


def analyze_convergence() -> dict:
    """Analyze convergence state of the auto-research loop."""
    if not HISTORY.exists():
        return {
            "error": "history.jsonl not found",
            "consecutive_no_gain": 0,
            "best_score": 0.0,
            "ceiling_gap": 0.0,
            "structural_diversity": {},
            "exhausted": False,
        }

    records = []
    with open(HISTORY, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass

    if not records:
        return {
            "consecutive_no_gain": 0,
            "best_score": 0.0,
            "ceiling_gap": 0.0,
            "structural_diversity": {},
            "exhausted": False,
            "message": "No experiment records found",
        }

    # Check convergence: how many consecutive non-improvements?
    recent = records[-20:]
    consecutive_no_gain = 0
    for r in reversed(recent):
        if r.get("status") == "improved":
            break
        consecutive_no_gain += 1

    # Check ceiling: best vs theoretical upper bound
    best = max((r.get("overall_score", 0) for r in records), default=0)
    theoretical_ceiling = 0.347  # From docs/current-analysis.md

    # Check structural diversity: how many distinct decay function types tried?
    keywords = Counter()
    for r in records:
        h = r.get("hypothesis", "").lower()
        for kw in ["jost", "gompertz", "sigmoid", "hyperbolic", "power", "bi-exp", "bi_exp", "dual", "piecewise"]:
            if kw in h:
                keywords[kw] += 1

    return {
        "consecutive_no_gain": consecutive_no_gain,
        "best_score": round(best, 4),
        "ceiling_gap": round(theoretical_ceiling - best, 4),
        "structural_diversity": dict(keywords),
        "exhausted": consecutive_no_gain >= 10 and best > 0.20,
        "num_experiments": len(records),
        "message": (
            "SEARCH EXHAUSTED — consider widening search surface"
            if consecutive_no_gain >= 10 and best > 0.20
            else "Search active — continue loop"
        ),
    }


if __name__ == "__main__":
    result = analyze_convergence()
    print(json.dumps(result, indent=2))
