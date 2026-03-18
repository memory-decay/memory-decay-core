"""Human review calibration evaluation utilities."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


def sigmoid_probability(logit: float) -> float:
    """Numerically stable sigmoid."""
    clipped = max(min(float(logit), 60.0), -60.0)
    if clipped >= 0:
        z = math.exp(-clipped)
        value = 1.0 / (1.0 + z)
    else:
        z = math.exp(clipped)
        value = z / (1.0 + z)
    if value <= 0.0:
        return math.nextafter(0.0, 1.0)
    if value >= 1.0:
        return math.nextafter(1.0, 0.0)
    return value


@dataclass
class _State:
    activation: float = 1.0
    stability: float = 0.0
    memory_type: str = "fact"


class HumanCalibrationEvaluator:
    """Replay human review events and score probabilistic calibration."""

    def __init__(self, decay_params: dict, observation_params: dict):
        self.decay_params = dict(decay_params)
        self.observation_params = dict(observation_params)
        self._state: dict[tuple[str, str], _State] = {}

    def get_state(self, user_id: str, item_id: str) -> dict:
        state = self._state[(str(user_id), str(item_id))]
        return {
            "activation": state.activation,
            "stability": state.stability,
            "memory_type": state.memory_type,
        }

    def _ensure_state(self, event: dict) -> _State:
        key = (str(event["user_id"]), str(event["item_id"]))
        state = self._state.get(key)
        if state is None:
            state = _State(memory_type=str(event.get("memory_type", "fact")))
            self._state[key] = state
        return state

    def _decay_activation(
        self,
        activation: float,
        stability: float,
        memory_type: str,
        t_elapsed: float,
    ) -> float:
        lam = (
            self.decay_params["lambda_fact"]
            if memory_type == "fact"
            else self.decay_params["lambda_episode"]
        )
        rho = self.decay_params["stability_weight"]
        alpha = self.decay_params.get("alpha", 0.0)
        impact = 1.0
        combined = max((1.0 + alpha * impact) * (1.0 + rho * stability), 1e-9)
        effective_lambda = lam / combined
        decayed = activation * math.exp(-effective_lambda * max(float(t_elapsed), 0.0))
        return min(max(decayed, 0.0), 1.0)

    def _predict_probability(self, activation: float, stability: float) -> float:
        logit = (
            self.observation_params.get("activation_scale", 1.0) * activation
            + self.observation_params.get("stability_scale", 0.0) * stability
            + self.observation_params.get("bias", 0.0)
        )
        return sigmoid_probability(logit)

    def replay_event(self, event: dict) -> dict:
        state = self._ensure_state(event)
        activation_before_review = self._decay_activation(
            state.activation,
            state.stability,
            state.memory_type,
            float(event["t_elapsed"]),
        )
        predicted_probability = self._predict_probability(
            activation_before_review, state.stability
        )

        outcome = int(event["outcome"])
        if outcome == 1:
            gain = self.decay_params["reinforcement_gain_direct"]
            cap = max(float(self.decay_params["stability_cap"]), 1e-9)
            state.stability = min(
                cap,
                state.stability + gain * (1.0 - state.stability / cap),
            )
            state.activation = 1.0
        else:
            state.activation = activation_before_review

        return {
            "predicted_probability": predicted_probability,
            "activation_before_review": activation_before_review,
            "outcome": outcome,
        }

    def evaluate(self, events: list[dict]) -> dict:
        if not events:
            return {"nll": 0.0, "brier": 0.0, "ece": 0.0, "num_events": 0}

        predictions = []
        labels = []
        for event in events:
            result = self.replay_event(event)
            p = min(max(float(result["predicted_probability"]), 1e-6), 1.0 - 1e-6)
            y = int(result["outcome"])
            predictions.append(p)
            labels.append(y)

        preds = np.asarray(predictions, dtype=np.float64)
        ys = np.asarray(labels, dtype=np.float64)

        nll = float(-(ys * np.log(preds) + (1.0 - ys) * np.log(1.0 - preds)).mean())
        brier = float(np.mean((preds - ys) ** 2))
        ece = self._expected_calibration_error(preds, ys)

        return {
            "nll": nll,
            "brier": brier,
            "ece": ece,
            "num_events": len(events),
        }

    def _expected_calibration_error(
        self, predictions: np.ndarray, labels: np.ndarray, num_bins: int = 10
    ) -> float:
        bins = np.linspace(0.0, 1.0, num_bins + 1)
        total = len(predictions)
        ece = 0.0

        for i in range(num_bins):
            left = bins[i]
            right = bins[i + 1]
            if i == num_bins - 1:
                mask = (predictions >= left) & (predictions <= right)
            else:
                mask = (predictions >= left) & (predictions < right)
            if not np.any(mask):
                continue
            bin_pred = float(predictions[mask].mean())
            bin_label = float(labels[mask].mean())
            ece += abs(bin_pred - bin_label) * (mask.sum() / total)

        return float(ece)
