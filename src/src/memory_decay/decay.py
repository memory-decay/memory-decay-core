"""Time-based decay engine for memory activation scores."""

from __future__ import annotations

import math
from typing import Literal

from .graph import MemoryGraph


class DecayEngine:
    """Applies decay functions to memory activation scores over time.

    Supports exponential and power law decay with per-type parameters
    and impact modifiers.
    """

    def __init__(
        self,
        graph: MemoryGraph,
        decay_type: Literal["exponential", "power_law"] = "exponential",
        params: dict | None = None,
    ):
        self._graph = graph
        self.decay_type = decay_type
        self.current_tick = 0

        # Default parameters tuned for meaningful decay over ~100 ticks
        self._params = {
            "lambda_fact": 0.03,
            "lambda_episode": 0.05,
            "beta_fact": 0.15,
            "beta_episode": 0.25,
            "alpha": 0.5,  # impact modifier: higher = less decay for high-impact
        }
        if params:
            self._params.update(params)

    def get_params(self) -> dict:
        return dict(self._params)

    def set_params(self, new_params: dict) -> None:
        self._params.update(new_params)

    def _compute_decay(
        self, initial_activation: float, delta_t: int, impact: float, mtype: str
    ) -> float:
        """Compute activation after decay.

        Exponential: A(t) = A₀ * exp(-λ_eff * Δt)
          where λ_eff = λ / (1 + α * impact)  → high impact → lower effective λ → slower decay
        Power law:   A(t) = A₀ * (Δt + 1)^(-β_eff)
          where β_eff = β / (1 + α * impact)  → high impact → lower effective β → slower decay
        """
        alpha = self._params["alpha"]
        impact_factor = 1.0 + alpha * impact  # ranges from 1.1 (low impact) to 2.0 (high)

        if self.decay_type == "exponential":
            lam = (
                self._params["lambda_fact"]
                if mtype == "fact"
                else self._params["lambda_episode"]
            )
            effective_lambda = lam / impact_factor
            decayed = initial_activation * math.exp(-effective_lambda * delta_t)
        else:
            beta = (
                self._params["beta_fact"]
                if mtype == "fact"
                else self._params["beta_episode"]
            )
            effective_beta = beta / impact_factor
            decayed = initial_activation * ((delta_t + 1) ** (-effective_beta))

        # Clamp to [0, 1]
        return min(max(decayed, 0.0), 1.0)

    def tick(self) -> None:
        """Advance time by 1 step.

        For each memory node:
        1. Calculate delta_t = current_tick - last_activated_tick
        2. Apply decay formula using current activation as base
        3. Update activation_score and last_activated_tick
        """
        self.current_tick += 1

        for nid, attrs in self._graph._graph.nodes(data=True):
            if attrs.get("type") in ("unknown", None):
                continue

            delta_t = self.current_tick - attrs["last_activated_tick"]
            if delta_t <= 0:
                continue

            initial_activation = attrs["activation_score"]
            mtype = attrs["type"]
            impact = attrs["impact"]

            new_activation = self._compute_decay(
                initial_activation, delta_t, impact, mtype
            )

            self._graph._graph.nodes[nid]["activation_score"] = new_activation
            self._graph._graph.nodes[nid]["last_activated_tick"] = self.current_tick
