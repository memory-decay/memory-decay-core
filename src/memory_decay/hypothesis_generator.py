"""M27-style hypothesis generator with memory chain context."""

from __future__ import annotations

import json
import os
import re
import time
from typing import Any, Optional

from openai import OpenAI

from memory_decay.utils import extract_json_from_markdown


HYPOTHESIS_GENERATION_SYSTEM = """You are the hypothesis generator for an autonomous memory-decay experiment.
You receive persistent memory chain context from previous rounds and generate specific, testable hypotheses.

The decay function interface you can modify:
    def compute_decay(activation, impact, stability, mtype, params) -> float:
        # activation: float 0-1, current memory activation
        # impact: float 0-1, memory importance (given, not learned)
        # stability: float 0-1, reinforcement level
        # mtype: "fact" or "episode"
        # params: dict of tunable floats

You control: decay function logic + params.json
You CANNOT control: evaluator, runner, graph, dataset, cache

Current best (exp_0315, overall=0.2228, CV=4.8%):
  - Jost-plus-sigmoid: floor = sigmoid(importance) * floor_max
  - Jost decay: da/dt = -lambda * (excess ** jost_power)
  - Theoretical ceiling: ~0.347 (embedding-limited recall=0.39)

Failure patterns to avoid repeating:
  - Gompertz floor: collapsed plausibility to 0.50
  - Decoupled impact/stability: recall dropped to 0.28
  - Dual-sigmoid floor: could not preserve final score
  - Piecewise Jost: converged to same weaker regime
"""

HYPOTHESIS_GENERATION_USER_TEMPLATE = """## Memory Chain Context (Recent Rounds)
{memory_chain}

## Current Best Decay Function
```python
{def_code}
```

## Current Best Parameters
```json
{params}
```

## Task
Generate a specific, testable hypothesis for the next experiment round.

Your hypothesis must:
1. Address a gap that hasn't been systematically explored yet
2. Be grounded in the memory chain learnings
3. Specify both the decay function change AND the params
4. Predict what the expected outcome will be

Respond in JSON:
{{
  "hypothesis": "2-3 sentence description of what to try and why",
  "decay_fn_code": "Complete python code for compute_decay function",
  "params": {{ "param_name": value, ... }},
  "predicted_effect": "What metric should improve and by how much",
  "risk": "What could go wrong"
}}
"""


class HypothesisGenerator:
    """Generate hypotheses with M27-style memory chain context."""

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

    def generate_hypothesis(
        self,
        memory_chain_context: str,
        current_decay_fn: str,
        current_params: dict,
        round_num: int,
    ) -> dict[str, Any]:
        """Generate next hypothesis using memory chain context."""
        user_msg = HYPOTHESIS_GENERATION_USER_TEMPLATE.format(
            memory_chain=memory_chain_context,
            def_code=current_decay_fn,
            params=json.dumps(current_params, indent=2),
        )

        response_text: Optional[str] = None
        for attempt in range(3):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    max_tokens=2048,
                    timeout=60.0,
                    messages=[
                        {"role": "system", "content": HYPOTHESIS_GENERATION_SYSTEM},
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
                    return {"error": "API unavailable", "round_num": round_num}

        if response_text is None:
            return {"error": "No response from API", "round_num": round_num}

        # Parse JSON from response
        try:
            result = extract_json_from_markdown(response_text)
        except (json.JSONDecodeError, re.error):
            return {"error": "Parse failure", "raw": response_text[:500], "round_num": round_num}

        result["round_num"] = round_num
        return result
