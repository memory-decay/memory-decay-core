"""Run Eval v2 summaries on the baseline trio."""

from __future__ import annotations

import json
import os
from pathlib import Path

from memory_decay.cross_validator import run_kfold


EXPERIMENTS = {
    "baseline": Path("experiments/exp_0000"),
    "jost_p4": Path("experiments/exp_0315"),
    "assoc_boost_best": Path("experiments/exp_0338"),
}


def main() -> None:
    output_path = Path(os.environ.get("EVAL_V2_OUTPUT", "outputs/eval_v2_summary.json"))
    output_path.parent.mkdir(parents=True, exist_ok=True)

    summary = {}
    for name, exp_dir in EXPERIMENTS.items():
        result = run_kfold(exp_dir, k=5, cache_dir=Path("cache"))
        summary[name] = {
            "mean": result.get("mean", {}),
            "std": result.get("std", {}),
            "worst_fold": result.get("worst_fold", {}),
            "fold_deltas": result.get("fold_deltas", []),
        }

    output_path.write_text(json.dumps(summary, indent=2))

    ranked = sorted(
        summary.items(),
        key=lambda item: item[1]["mean"].get("eval_v2_score", 0.0),
        reverse=True,
    )
    for rank, (name, result) in enumerate(ranked, start=1):
        print(f"{rank}. {name}: eval_v2={result['mean'].get('eval_v2_score', 0.0):.4f}")


if __name__ == "__main__":
    main()
