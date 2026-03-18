"""Cubic decay with boosted reinforcement parameters.

Same cubic decay as exp_0004 but with stronger reinforcement effects:
higher stability gains from reactivation, slower stability decay.
This creates a bigger gap between reactivated items (high activation,
high stability → very slow decay) and non-reactivated items (baseline
decay). The gap should improve correlation especially at threshold 0.5.
"""

import math


def compute_decay(activation, impact, stability, mtype, params):
    alpha = params.get("alpha", 2.0)
    rho = params.get("stability_weight", 1.2)
    impact_factor = math.exp(alpha * impact)
    stability_factor = 1.0 + rho * stability
    combined = max(impact_factor * stability_factor, 1e-9)

    lam = params.get("lambda_fact", 0.010) if mtype == "fact" else params.get("lambda_episode", 0.020)
    # Cubic decay: rate proportional to a^2
    decay_rate = lam * activation * activation / combined
    return activation * (1.0 - decay_rate)
