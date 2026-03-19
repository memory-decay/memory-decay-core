"""Eval set builder: identifies systematic weaknesses from memory chain and generates targeted evaluation sets."""

from __future__ import annotations

import json
from pathlib import Path


WEAKNESS_TYPES = ["recall", "mrr", "correlation", "selectivity", "precision_lift"]


class EvalSetBuilder:
    """Analyze memory chain and history to identify systematic weaknesses."""

    def __init__(self, history_path: Path, chain_dir: Path | None = None):
        self.history_path = Path(history_path)
        self.chain_dir = chain_dir or (self.history_path.parent.parent / "memory_chain")
        self._load_history()

    def _load_history(self) -> None:
        self.records = []
        if self.history_path.exists():
            with open(self.history_path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            self.records.append(json.loads(line))
                        except json.JSONDecodeError:
                            pass

    def identify_memory_weakness(self) -> dict:
        """Identify the most impactful systematic weakness from experiment history."""
        if not self.records:
            return {"type": "recall", "description": "No history available", "confidence": "LOW"}


        # Find which metric has the most room to improve
        metric_gaps = self._compute_metric_gaps()

        # Find which decay function types have been systematically unexplored
        unexplored = self._find_unexplored_structural_slots()

        # Pick the highest-impact weakness
        worst_metric = max(metric_gaps, key=metric_gaps.__getitem__)

        return {
            "type": worst_metric,
            "description": f"{worst_metric} is the bottleneck (gap={metric_gaps[worst_metric]:.4f})",
            "gap": metric_gaps[worst_metric],
            "unexplored_structures": unexplored,
            "confidence": "HIGH" if len(self.records) > 30 else "MEDIUM",
        }

    def _compute_metric_gaps(self) -> dict[str, float]:
        """Compute gap between current performance and theoretical upper bounds."""
        theoretical = {
            "recall": 0.390,
            "mrr": 0.390,
            "correlation": 0.5,
            "selectivity": 0.5,
            "precision_lift": 0.10,
        }
        current = {
            "recall": max((r.get("recall_mean", 0) for r in self.records), default=0.0),
            "mrr": max((r.get("mrr_mean", 0) for r in self.records), default=0.0),
            "correlation": max((r.get("corr_score", 0) for r in self.records), default=0.0),
            "selectivity": 0.0,  # Not tracked in v1 history
            "precision_lift": max((r.get("precision_lift", 0) for r in self.records), default=0.0),
        }
        return {k: theoretical[k] - current.get(k, 0) for k in theoretical}

    def _find_unexplored_structural_slots(self) -> list[str]:
        """Identify decay function types not yet systematically explored."""
        tried = set()
        for r in self.records:
            h = r.get("hypothesis", "").lower()
            for kw in ["jost", "gompertz", "hyperbolic", "power_law", "bi_exp", "dual", "piecewise"]:
                if kw in h:
                    tried.add(kw)
        all_types = ["jost", "gompertz", "hyperbolic", "power_law", "bi_exp", "dual", "piecewise"]
        return [t for t in all_types if t not in tried]

    def generate_targeted_eval_set(self, weakness_type: str, output_path: Path) -> Path:
        """Generate a targeted evaluation set for a specific weakness type."""
        weakness = self.identify_memory_weakness()
        eval_spec = {
            "weakness_type": weakness_type,
            "focus_regime": self._regime_for_weakness(weakness_type),
            "confidence": weakness["confidence"],
            "generated_from": f"{len(self.records)} historical experiments",
        }
        output_path.write_text(json.dumps(eval_spec, indent=2))
        return output_path

    def _regime_for_weakness(self, weakness: str) -> str:
        regimes = {
            "recall": "mid-delay recall (ticks 80-150), top-5 retrieval",
            "mrr": "high-precision ranking, single correct answer in top-3",
            "correlation": "activation-recall correlation across all importance levels",
            "selectivity": "discriminating high-importance from low-importance recall",
            "precision_lift": "pruning incorrect associations above threshold",
        }
        return regimes.get(weakness, "general retrieval")
