"""Time-based decay engine for memory activation scores."""

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
    ):
        self._graph = graph
        self.decay_type = decay_type
        self.current_tick = 0

        # Tuned for gradual decay over ~200 ticks
        # Exponential: e^(-0.02) per tick → after 100 ticks: 0.135 (fact, low impact)
        # Power law: (t)^(-0.08) per tick → slower initial decay, longer tail
        self._params = {
            "lambda_fact": 0.02,
            "lambda_episode": 0.035,
            "beta_fact": 0.08,
            "beta_episode": 0.12,
            "alpha": 0.5,
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

        Exponential: A_new = A₀ * exp(-λ_eff * Δt)
          where λ_eff = λ / (1 + α * impact)
        Power law:   A_new = A₀ * (Δt + 1)^(-β_eff)
          where β_eff = β / (1 + α * impact)
        """
        alpha = self._params["alpha"]
        impact_factor = 1.0 + alpha * impact

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

        return min(max(decayed, 0.0), 1.0)

    def tick(self) -> None:
        """Advance time by 1 step.

        Each memory decays from its own last_activated_tick.
        Memories inserted at later ticks decay less overall.
        """
        self.current_tick += 1

        for nid, attrs in self._graph._graph.nodes(data=True):
            if attrs.get("type") in ("unknown", None):
                continue

            initial_activation = attrs["activation_score"]
            mtype = attrs["type"]
            impact = attrs["impact"]
            last_tick = attrs["last_activated_tick"]

            delta_t = self.current_tick - last_tick
            if delta_t <= 0:
                continue

            new_activation = self._compute_decay(
                initial_activation, delta_t, impact, mtype
            )

            self._graph._graph.nodes[nid]["activation_score"] = new_activation
            self._graph._graph.nodes[nid]["last_activated_tick"] = self.current_tick
