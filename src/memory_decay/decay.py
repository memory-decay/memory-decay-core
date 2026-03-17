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

        # Default parameters
        self._params = {
            "lambda_fact": 0.05,
            "lambda_episode": 0.08,
            "beta_fact": 0.3,
            "beta_episode": 0.5,
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

        Exponential: A(t) = A₀ * exp(-λ * Δt) * (1 + α * impact)
        Power law:   A(t) = A₀ * (Δt + 1)^(-β) * (1 + α * impact)
        """
        alpha = self._params["alpha"]
        impact_mod = 1.0 + alpha * impact

        if self.decay_type == "exponential":
            lam = (
                self._params["lambda_fact"]
                if mtype == "fact"
                else self._params["lambda_episode"]
            )
            decayed = initial_activation * math.exp(-lam * delta_t) * impact_mod
        else:
            beta = (
                self._params["beta_fact"]
                if mtype == "fact"
                else self._params["beta_episode"]
            )
            decayed = initial_activation * ((delta_t + 1) ** (-beta)) * impact_mod

        return min(max(decayed, 0.0), 2.0)

    def tick(self) -> None:
        """Advance time by 1 step.

        For each memory node:
        1. Calculate delta_t = current_tick - last_activated_tick
        2. Store current activation as A₀
        3. Apply decay formula
        4. Update activation_score and last_activated_tick
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
