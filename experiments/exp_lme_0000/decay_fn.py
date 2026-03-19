"""Baseline: default exponential decay (same as original DecayEngine)."""

import math


def compute_decay(activation, impact, stability, mtype, params):
    alpha = params.get("alpha", 0.5)
    rho = params.get("stability_weight", 0.8)
    impact_factor = 1.0 + alpha * impact
    stability_factor = 1.0 + rho * stability
    combined = max(impact_factor * stability_factor, 1e-9)

    lam = params.get("lambda_fact", 0.02) if mtype == "fact" else params.get("lambda_episode", 0.035)
    effective_lambda = lam / combined
    return activation * math.exp(-effective_lambda)
