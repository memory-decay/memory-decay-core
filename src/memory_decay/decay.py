"""Time-based decay engine for memory activation and stability scores."""

from __future__ import annotations

import math
from typing import Literal

from .graph import MemoryGraph


class DecayEngine:
    """Applies decay functions to memory activation scores over time.

    Supports exponential and power law decay with per-type parameters
    and impact modifiers. Designed so that memories inserted at different
    ticks start decaying from their insertion tick.
    """

    def __init__(
        self,
        graph: MemoryGraph,
        decay_type: Literal["exponential", "power_law"] = "exponential",
        params: dict | None = None,
        custom_decay_fn=None,
    ):
        self._graph = graph
        self.decay_type = decay_type
        self.current_tick = 0
        self._custom_decay_fn = custom_decay_fn

        # Tuned for gradual decay over ~200 ticks
        # Exponential: e^(-0.02) per tick → after 100 ticks: 0.135 (fact, low impact)
        # Power law: (t)^(-0.08) per tick → slower initial decay, longer tail
        self._params = {
            "lambda_fact": 0.02,
            "lambda_episode": 0.035,
            "beta_fact": 0.08,
            "beta_episode": 0.12,
            "alpha": 0.5,
            "stability_weight": 0.8,
            "stability_decay": 0.01,
            "reinforcement_gain_direct": 0.2,
            "reinforcement_gain_assoc": 0.05,
            "stability_cap": 1.0,
        }
        if params:
            self._params.update(params)

    def get_params(self) -> dict:
        return dict(self._params)

    def set_params(self, new_params: dict) -> None:
        self._params.update(new_params)

    def _compute_decay(
        self,
        initial_activation: float,
        impact: float,
        stability: float,
        mtype: str,
    ) -> float:
        """Compute activation after decay.

        Exponential: A_new = A₀ * exp(-λ_eff)
          where λ_eff = λ / ((1 + α * impact) * (1 + ρ * stability))
        Power law:   A_new = A₀ * (1 + β_eff)^(-1)
          where β_eff = β / ((1 + α * impact) * (1 + ρ * stability))
        """
        if self._custom_decay_fn is not None:
            result = self._custom_decay_fn(
                initial_activation, impact, stability, mtype, self._params
            )
            return min(max(result, 0.0), 1.0)

        alpha = self._params["alpha"]
        rho = self._params["stability_weight"]
        impact_factor = 1.0 + alpha * impact
        stability_factor = 1.0 + rho * stability
        combined_factor = max(impact_factor * stability_factor, 1e-9)

        if self.decay_type == "exponential":
            lam = (
                self._params["lambda_fact"]
                if mtype == "fact"
                else self._params["lambda_episode"]
            )
            effective_lambda = lam / combined_factor
            decayed = initial_activation * math.exp(-effective_lambda)
        else:
            beta = (
                self._params["beta_fact"]
                if mtype == "fact"
                else self._params["beta_episode"]
            )
            effective_beta = beta / combined_factor
            decayed = initial_activation / ((1.0 + effective_beta) ** 1.0)

        return min(max(decayed, 0.0), 1.0)

    def tick(self) -> None:
        """Advance time by 1 step.

        Activation decays every tick once a memory has been created.
        Stability also decays slowly so reinforcement has a long but finite effect.
        """
        self.current_tick += 1

        for nid, attrs in self._graph._graph.nodes(data=True):
            if attrs.get("type") in ("unknown", None):
                continue

            if self.current_tick < attrs.get("created_tick", 0):
                continue

            initial_activation = float(attrs["activation_score"])
            mtype = attrs["type"]
            impact = float(attrs["impact"])
            stability = float(attrs.get("stability_score", 0.0))

            new_activation = self._compute_decay(
                initial_activation, impact, stability, mtype
            )
            new_stability = max(
                0.0, stability * (1.0 - self._params["stability_decay"])
            )

            self._graph._graph.nodes[nid]["activation_score"] = new_activation
            self._graph._graph.nodes[nid]["stability_score"] = min(
                new_stability, self._params["stability_cap"]
            )
