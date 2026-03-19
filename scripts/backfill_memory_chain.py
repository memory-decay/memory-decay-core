"""Backfill memory_chain/ from history.jsonl for rounds already run."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HISTORY = ROOT / "experiments" / "history.jsonl"
CHAIN_DIR = ROOT / "memory_chain"

import sys
sys.path.insert(0, str(ROOT / "src"))
from memory_decay.memory_chain_indexer import MemoryChainIndexer, RoundData


def decide_next_direction(record: dict, prev_record: dict | None) -> str:
    """Infer next direction from hypothesis, delta, and score trend."""
    overall = record.get("overall_score", 0)
    status = record.get("status", "")
    hypothesis = record.get("hypothesis", "")

    # Detect score trend if previous record is available
    trend = ""
    if prev_record is not None:
        prev_overall = prev_record.get("overall_score", 0)
        delta = overall - prev_overall
        if delta > 0.05:
            trend = " (improving)"
        elif delta < -0.05:
            trend = " (declining)"

    if status == "improved":
        return f"Continue exploring: {hypothesis[:100]}"
    elif overall < 0.10:
        return f"Explore fundamentally different decay architecture{trend}"
    elif overall < 0.20:
        return f"Refine parameters around current decay form{trend}"
    else:
        return f"Fine-tune; structural changes no longer improving{trend}"


def main():
    idx = MemoryChainIndexer(CHAIN_DIR)

    if not HISTORY.exists():
        raise FileNotFoundError(
            f"history.jsonl not found at {HISTORY}. "
            "Run experiments before backfilling."
        )

    records = []
    with open(HISTORY, encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"WARNING: Skipping malformed line {lineno}: {e}")

    for i, rec in enumerate(records):
        round_num = i
        experiment = rec.get("experiment", f"exp_{i:04d}")

        # Map old field names to canonical
        scores = {
            "overall_score": rec.get("overall_score", rec.get("overall", 0)),
            "retrieval_score": rec.get("retrieval_score", rec.get("retrieval", 0)),
            "plausibility_score": rec.get("plausibility_score", rec.get("plausibility", 0)),
            "recall_mean": rec.get("recall_mean", 0),
            "mrr_mean": rec.get("mrr_mean", 0),
            "precision_lift": rec.get("precision_lift", 0),
        }

        prev_rec = records[i - 1] if i > 0 else None
        next_dir = decide_next_direction(rec, prev_rec)

        observations = [f"Status: {rec.get('status', 'unknown')}"]
        if rec.get("delta"):
            observations.append(f"Delta vs previous: {rec['delta']:+.4f}")

        decisions = []
        open_questions = ["Is this the global optimum or just local?"]

        data = RoundData(
            round_num=round_num,
            experiment=experiment,
            scores=scores,
            hypothesis=rec.get("hypothesis", ""),
            observations=observations,
            decisions=decisions,
            open_questions=open_questions,
            next_direction=next_dir,
            parent_round=i - 1 if i > 0 else None,
        )
        idx.write_round(data)
        print(f"  Wrote round_{round_num:04d}.md ({experiment})")

    print(f"\nBackfill complete: {len(records)} rounds written")


if __name__ == "__main__":
    main()
