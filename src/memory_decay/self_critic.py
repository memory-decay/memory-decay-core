"""M27-style self-critique agent: analyzes failure trajectories and generates optimization directions."""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any, Optional

from openai import OpenAI

from memory_decay.utils import extract_json_from_markdown


FAILURE_PATTERNS_SYSTEM = """You are analyzing the failure trajectory of an autonomous memory-decay experiment.
You have access to the persistent memory chain documenting every experiment run so far.

Your task is to:
1. Identify WHY recent experiments failed (systematic patterns, not just scores)
2. Identify what is holding back the overall_score ceiling
3. Suggest concrete, actionable next directions

Key context about the experiment:
- 46 experiments recorded (exp_0000 through exp_0345)
- Current best: exp_0315 (overall=0.2228) with Jost-plus-sigmoid decay
- Search converged: structural alternatives after exp_0315 failed (0.19x range)
- Theoretical upper bound on overall_score is ~0.347
- The bottleneck is recall_mean (0.40) and mrr_mean (0.24) — both limited by embedding ceiling

IMPORTANT: Your analysis should be grounded in the actual memory_chain data.
Focus on systematic failure patterns, not just scoring individual experiments.
"""

FAILURE_PATTERNS_USER_TEMPLATE = """## Memory Chain (Recent Rounds)

{memory_chain}

## Experiment History (Last 10)
{history_last_10}

## Task
Based on the memory chain and history above:

1. **Failure Pattern Analysis**: What systematic patterns caused recent experiments to fail? Look for:
   - Repeated structural changes that all converged to similar scores
   - Parameter changes that improved one metric but hurt another
   - Theoretical constraints hitting ceiling effects

2. **Root Cause Diagnosis**: Why did the search converge at exp_0315?
   - What does exp_0315's Jost-plus-sigmoid do differently?
   - Why do alternatives (Gompertz floor, decoupled impact/stability, dual-sigmoid) fail?

3. **Next Direction**: Concrete, actionable hypothesis for the next round.
   - What specific change should be tried?
   - What does theory predict will happen?
   - What is the risk/预期?

Respond in JSON:
{{
  "observations": [
    "Observation 1 about failure patterns",
    "Observation 2 about root causes"
  ],
  "diagnosis": "2-3 sentence diagnosis of why search converged",
  "next_direction": "Specific actionable hypothesis for next experiment",
  "expected_impact": "HIGH/MEDIUM/LOW — how much overall_score improvement is expected",
  "risk": "What could go wrong with this direction"
}}
"""


class SelfCritic:
    """Self-critique agent that analyzes failure trajectories."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4o-mini",
        base_url: Optional[str] = None,
    ):
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY required")
        kwargs: dict[str, Any] = {"api_key": self.api_key}
        if base_url:
            kwargs["base_url"] = base_url
        self.client = OpenAI(**kwargs)
        self.model = model

    def critique_round(
        self,
        memory_chain_summary: str,
        history_last_10: list[dict],
        round_num: int,
    ) -> dict[str, Any]:
        """Generate self-critique for the current round."""
        user_msg = FAILURE_PATTERNS_USER_TEMPLATE.format(
            memory_chain=memory_chain_summary,
            history_last_10=json.dumps(history_last_10, indent=2),
        )

        response_text: Optional[str] = None
        for attempt in range(3):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    max_tokens=2048,
                    timeout=60.0,
                    messages=[
                        {"role": "system", "content": FAILURE_PATTERNS_SYSTEM},
                        {"role": "user", "content": user_msg},
                    ],
                )
                response_text = (response.choices[0].message.content or "").strip()
                break
            except Exception as e:
                print(f"  API attempt {attempt + 1}/3 failed: {e}")
                if attempt < 2:
                    time.sleep(2 ** attempt)
                else:
                    return {
                        "observations": ["API failed, defaulting to continuation"],
                        "diagnosis": "API unavailable",
                        "next_direction": "Continue current decay parameter refinement",
                        "expected_impact": "MEDIUM",
                        "risk": "Unknown due to API failure",
                        "round_num": round_num,
                    }

        # Parse JSON from response
        assert response_text is not None
        try:
            result = extract_json_from_markdown(response_text)
        except (json.JSONDecodeError, re.error):
            result = {
                "observations": ["Could not parse critique"],
                "diagnosis": "Parse failure",
                "next_direction": "Continue parameter refinement",
                "expected_impact": "MEDIUM",
                "risk": "Unknown",
            }

        result["round_num"] = round_num
        return result

    def critique_from_chain(
        self,
        chain_dir: Path,
        history_path: Path,
        round_num: int,
    ) -> dict[str, Any]:
        """Convenience: load chain and history, then critique."""
        from memory_decay.memory_chain_indexer import MemoryChainIndexer

        idx = MemoryChainIndexer(chain_dir)
        chain_summary = idx.read_chain_summary(from_round=max(0, round_num - 10))

        history = []
        if history_path.exists():
            with open(history_path, encoding="utf-8") as f:
                history = [json.loads(line) for line in f]

        last_10 = history[-10:] if len(history) >= 10 else history
        return self.critique_round(chain_summary, last_10, round_num)
