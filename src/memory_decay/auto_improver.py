"""LLM-driven auto-improvement loop for decay parameter optimization."""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Optional

from openai import OpenAI


# Pre-defined guidance levels
GUIDANCE = {
    "minimal": (
        "Improve overall_score with emphasis on retrieval_score. "
        "Avoid near-perfect retention across all ticks."
    ),
    "default": """You are optimizing a memory decay system. The system models human memory using:
- Decay functions: exponential or power law
- Impact modifier: higher impact slows forgetting
- Stability reinforcement: direct recalls and cascades increase long-term stability
- Per-type parameters: facts and episodes have separate decay rates
Optimize for overall_score, but treat retrieval_score as the primary driver and plausibility_score as a guardrail.""",
    "expert": """You are optimizing a reinforcement-aware memory decay system.

Tune these behaviors directly:
- lambda_fact / lambda_episode: base forgetting speed for each memory type
- beta_fact / beta_episode: power-law forgetting speed for each memory type
- alpha: how much memory impact slows forgetting
- stability_weight (rho): how much accumulated stability slows future decay
- stability_decay (mu): how quickly reinforcement fades if not revisited
- reinforcement_gain_direct (r_direct): how much a directly recalled memory hardens
- reinforcement_gain_assoc (r_assoc): how much associated memories harden
- stability_cap: upper bound on reinforcement strength

Prefer parameter sets that:
1. Improve retrieval_score across the threshold grid, not just at one threshold
2. Keep plausibility_score healthy via positive activation-recall correlation and smooth curves
3. Avoid pathological flat recall curves near 1.0 across all ticks
4. Use cascade reinforcement conservatively relative to direct reinforcement""",
}


class AutoImprover:
    """LLM-driven iterative optimization of memory decay parameters."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4o-mini",
        guidance_level: str = "default",
        base_url: Optional[str] = None,
    ):
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OpenAI API key required. Set OPENAI_API_KEY env var "
                "or pass api_key parameter."
            )

        kwargs = {"api_key": self.api_key}
        if base_url:
            kwargs["base_url"] = base_url
        self.client = OpenAI(**kwargs)
        self.model = model

        if guidance_level not in GUIDANCE:
            raise ValueError(
                f"Invalid guidance_level: {guidance_level}. "
                f"Choose from: {list(GUIDANCE.keys())}"
            )
        self.guidance = GUIDANCE[guidance_level]
        self.guidance_level = guidance_level
        self._history: list[dict] = []

    def propose_parameters(
        self,
        current_params: dict,
        evaluation_history: list[dict],
        iteration: int,
        total_budget: int,
    ) -> dict:
        """Analyze results and propose new parameters."""
        prompt = self._build_prompt(
            current_params, evaluation_history, iteration, total_budget
        )

        # Retry with exponential backoff for API errors
        text = None
        for attempt in range(3):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    max_tokens=2048,
                    timeout=60,
                    messages=[{"role": "user", "content": prompt}],
                )
                text = response.choices[0].message.content.strip()
                break
            except Exception as e:
                print(f"    API attempt {attempt + 1}/3 failed: {e}")
                if attempt < 2:
                    time.sleep(2 ** attempt)
                else:
                    print("    All API attempts failed, keeping current params")
                    return current_params

        if "```json" in text:
            text = text.split("```json")[1].split("```")[0].strip()
        elif "```" in text:
            text = text.split("```")[1].split("```")[0].strip()

        try:
            result = json.loads(text)
        except json.JSONDecodeError:
            # Try to find any JSON object in the response
            match = re.search(r"\{[^{}]*\}", text, re.DOTALL)
            if match:
                try:
                    result = json.loads(match.group())
                except json.JSONDecodeError:
                    print(f"    Could not parse LLM response, keeping current params")
                    return current_params
            else:
                # Try multiline JSON with nested braces
                match = re.search(r"\{[\s\S]*\}", text)
                if match:
                    try:
                        result = json.loads(match.group())
                    except json.JSONDecodeError:
                        print(f"    Could not parse LLM response, keeping current params")
                        return current_params
                else:
                    print(f"    No JSON found in LLM response, keeping current params")
                    return current_params

        new_params = result.get("parameters", current_params)
        validated = self._validate_params(new_params, current_params)

        record = {
            "iteration": iteration,
            "current_params": current_params,
            "proposed_params": validated,
            "last_overall": evaluation_history[-1].get(
                "overall_score", evaluation_history[-1].get("composite_score", 0)
            )
            if evaluation_history
            else 0,
            "reasoning": result.get("reasoning", text[:500]),
        }
        self._history.append(record)

        return validated

    def _build_prompt(
        self,
        current_params: dict,
        evaluation_history: list[dict],
        iteration: int,
        total_budget: int,
    ) -> str:
        recent = evaluation_history[-5:] if evaluation_history else []

        return f"""{self.guidance}

## Current Parameters
```json
{json.dumps(current_params, indent=2)}
```

## Recent Evaluation Results (last 5 snapshots)
```json
{json.dumps(recent, indent=2)}
```

## Iteration Info
- Current iteration: {iteration} / {total_budget}
- Guidance level: {self.guidance_level}

## Task
Based on the evaluation results, propose new parameter values to improve the overall_score.

Analyze:
1. Is retrieval_score improving across thresholds, not just at one cutoff?
2. Is plausibility_score supported by positive activation-recall correlation?
3. Is reinforcement too strong, causing unrealistically flat retention?
4. Are direct and associative reinforcement gains proportionate?
5. Is the forgetting curve smooth without collapsing to zero too quickly?

Respond with JSON:
```json
{{
  "reasoning": "Your analysis of the current results and reasoning for changes",
  "parameters": {{
    "lambda_fact": <float, 0.001-0.5>,
    "lambda_episode": <float, 0.001-0.5>,
    "beta_fact": <float, 0.01-2.0>,
    "beta_episode": <float, 0.01-2.0>,
    "alpha": <float, 0.0-2.0>,
    "stability_weight": <float, 0.0-5.0>,
    "stability_decay": <float, 0.0-0.5>,
    "reinforcement_gain_direct": <float, 0.0-1.0>,
    "reinforcement_gain_assoc": <float, 0.0-1.0>,
    "stability_cap": <float, 0.1-5.0>
  }}
}}
```

JSON만 출력해주세요. 설명은 reasoning 필드에 넣어주세요."""

    def _canonicalize_param_names(self, proposed: dict) -> dict:
        aliases = {
            "rho": "stability_weight",
            "mu": "stability_decay",
            "r_direct": "reinforcement_gain_direct",
            "r_assoc": "reinforcement_gain_assoc",
        }
        canonical = dict(proposed)
        for alias, canonical_name in aliases.items():
            if alias in canonical and canonical_name not in canonical:
                canonical[canonical_name] = canonical[alias]
        return canonical

    def _validate_params(self, proposed: dict, current: dict) -> dict:
        proposed = self._canonicalize_param_names(proposed)
        ranges = {
            "lambda_fact": (0.001, 0.5),
            "lambda_episode": (0.001, 0.5),
            "beta_fact": (0.01, 2.0),
            "beta_episode": (0.01, 2.0),
            "alpha": (0.0, 2.0),
            "stability_weight": (0.0, 5.0),
            "stability_decay": (0.0, 0.5),
            "reinforcement_gain_direct": (0.0, 1.0),
            "reinforcement_gain_assoc": (0.0, 1.0),
            "stability_cap": (0.1, 5.0),
        }

        validated = {}
        for key, (lo, hi) in ranges.items():
            value = proposed.get(key, current.get(key, lo))
            try:
                value = float(value)
            except (TypeError, ValueError):
                value = current.get(key, lo)
            validated[key] = max(lo, min(hi, value))

        return validated

    def should_stop(
        self,
        evaluation_history: list[dict],
        iteration: int,
        total_budget: int,
        patience: int = 4,
    ) -> bool:
        if iteration >= total_budget:
            return True

        recent = evaluation_history[-patience:] if evaluation_history else []
        if len(recent) >= patience:
            scores = [
                h.get("overall_score", h.get("composite_score", 0)) for h in recent
            ]
            if max(scores) == scores[0]:
                return True

        if len(recent) >= 3:
            recalls = [h.get("recall_rate", 0) for h in recent]
            if all(r > 0.95 for r in recalls):
                return True

        return False

    @property
    def history(self) -> list[dict]:
        return list(self._history)
