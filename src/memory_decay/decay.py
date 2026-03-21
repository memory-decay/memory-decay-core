"""Time-based decay engine for memory activation and stability scores."""

from __future__ import annotations

import math
from typing import Literal

import numpy as np

from .graph import MemoryGraph


def _sigmoid_gate(value: float, center: float, width: float) -> float:
    """Numerically stable logistic gate used for soft consolidation."""
    scaled = (value - center) / max(width, 1e-6)
    if scaled >= 0:
        z = math.exp(-scaled)
        return 1.0 / (1.0 + z)
    z = math.exp(scaled)
    return z / (1.0 + z)


def soft_floor_decay_step(
    activation: float,
    impact: float,
    stability: float,
    *,
    lam: float,
    alpha: float = 2.0,
    rho: float = 0.8,
    floor_min: float = 0.05,
    floor_max: float = 0.35,
    floor_power: float = 2.0,
    gate_center: float = 0.4,
    gate_width: float = 0.08,
    consolidation_gain: float = 0.6,
    min_rate_scale: float = 0.1,
) -> float:
    """Decay toward an impact floor without ever increasing activation.

    The update has the closed form
      a_{t+1} = f(i) + (a_t - f(i)) * exp(-r(a_t, i, s))
    with f(i) clamped so that f(i) <= a_t. This guarantees
      f(i) <= a_{t+1} <= a_t
    for every pure-decay step.
    """
    activation = min(max(float(activation), 0.0), 1.0)
    if activation <= 0.0:
        return 0.0

    impact = min(max(float(impact), 0.0), 1.0)
    stability = max(float(stability), 0.0)
    floor_min = min(max(float(floor_min), 0.0), 1.0)
    floor_max = min(max(float(floor_max), floor_min), 1.0)
    floor_power = max(float(floor_power), 1e-6)
    min_rate_scale = min(max(float(min_rate_scale), 0.0), 1.0)

    combined = max(math.exp(alpha * impact) * (1.0 + rho * stability), 1e-9)
    raw_floor = floor_min + (floor_max - floor_min) * (impact ** floor_power)
    floor = min(raw_floor, activation)

    gate = _sigmoid_gate(activation, gate_center, gate_width)
    rate_scale = 1.0 - consolidation_gain * impact * gate
    rate_scale = min(max(rate_scale, min_rate_scale), 1.0)
    effective_rate = max(float(lam) * rate_scale / combined, 0.0)

    updated = floor + (activation - floor) * math.exp(-effective_rate)
    return min(max(updated, floor), activation)


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
        # Pre-extracted node arrays for fast tick() — built lazily
        self._tick_arrays_built = False
        self._tick_nids: list[str] = []
        self._tick_retrieval: np.ndarray | None = None
        self._tick_storage: np.ndarray | None = None
        self._tick_stability: np.ndarray | None = None
        self._tick_impact: np.ndarray | None = None
        self._tick_created: np.ndarray | None = None
        self._tick_is_fact: np.ndarray | None = None

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

    def _build_tick_arrays(self) -> None:
        """Pre-extract node data into parallel arrays for fast tick()."""
        nids = []
        retrieval = []
        storage = []
        stability = []
        impact = []
        created = []
        is_fact = []

        for nid, attrs in self._graph._graph.nodes(data=True):
            if attrs.get("type") in ("unknown", None):
                continue
            nids.append(nid)
            retrieval.append(float(
                attrs.get("retrieval_score", attrs.get("activation_score", 0.0))
            ))
            storage.append(float(
                attrs.get("storage_score", attrs.get("activation_score", 0.0))
            ))
            stability.append(float(attrs.get("stability_score", 0.0)))
            impact.append(float(attrs.get("impact", 0.0)))
            created.append(int(attrs.get("created_tick", 0)))
            is_fact.append(attrs.get("type") == "fact")

        self._tick_nids = nids
        self._tick_retrieval = np.array(retrieval, dtype=np.float64)
        self._tick_storage = np.array(storage, dtype=np.float64)
        self._tick_stability = np.array(stability, dtype=np.float64)
        self._tick_impact = np.array(impact, dtype=np.float64)
        self._tick_created = np.array(created, dtype=np.int64)
        self._tick_is_fact = np.array(is_fact, dtype=np.bool_)
        self._tick_arrays_built = True

    def _sync_tick_arrays_from_graph(self) -> None:
        """Re-read mutable scores from graph into arrays (after re_activate)."""
        nodes = self._graph._graph.nodes
        for i, nid in enumerate(self._tick_nids):
            attrs = nodes[nid]
            self._tick_retrieval[i] = float(
                attrs.get("retrieval_score", attrs.get("activation_score", 0.0))
            )
            self._tick_storage[i] = float(
                attrs.get("storage_score", attrs.get("activation_score", 0.0))
            )
            self._tick_stability[i] = float(attrs.get("stability_score", 0.0))

    def tick(self) -> None:
        """Advance time by 1 step.

        Storage and retrieval scores decay every tick once a memory has been created.
        Stability also decays slowly so reinforcement has a long but finite effect.
        """
        self.current_tick += 1

        if not self._tick_arrays_built:
            self._build_tick_arrays()
        else:
            self._sync_tick_arrays_from_graph()

        n = len(self._tick_nids)
        if n == 0:
            return

        # Mask: only process nodes created at or before current tick
        active = self._tick_created <= self.current_tick

        # Stability decay: vectorized (doesn't depend on custom_decay_fn)
        stability_decay = self._params["stability_decay"]
        stability_cap = self._params["stability_cap"]
        new_stability = np.where(
            active,
            np.minimum(self._tick_stability * (1.0 - stability_decay), stability_cap),
            self._tick_stability,
        )

        # Decay scores: must use per-element custom_decay_fn
        new_retrieval = self._tick_retrieval.copy()
        new_storage = self._tick_storage.copy()

        compute = self._compute_decay
        retrieval_arr = self._tick_retrieval
        storage_arr = self._tick_storage
        impact_arr = self._tick_impact
        stability_arr = self._tick_stability
        is_fact_arr = self._tick_is_fact

        for i in range(n):
            if not active[i]:
                continue
            mtype = "fact" if is_fact_arr[i] else "episode"
            new_retrieval[i] = compute(
                retrieval_arr[i], impact_arr[i], stability_arr[i], mtype,
            )
            new_storage[i] = compute(
                storage_arr[i], impact_arr[i], stability_arr[i], mtype,
            )

        # Write back to graph in bulk
        nodes = self._graph._graph.nodes
        emb_nid_to_idx = self._graph._emb_nid_to_idx
        emb_scores = self._graph._emb_retrieval_scores

        for i in range(n):
            if not active[i]:
                continue
            nid = self._tick_nids[i]
            nr = new_retrieval[i]
            nodes[nid]["retrieval_score"] = nr
            nodes[nid]["activation_score"] = nr
            nodes[nid]["storage_score"] = new_storage[i]
            nodes[nid]["stability_score"] = new_stability[i]
            if emb_scores is not None:
                idx = emb_nid_to_idx.get(nid)
                if idx is not None:
                    emb_scores[idx] = max(nr, 0.0)
