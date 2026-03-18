"""Quadratic decay with impact-based activation floor.

Builds on exp_0001's power-law tail but adds a floor proportional to
impact^2. High-impact memories asymptote to a nonzero activation,
increasing the spread between important and unimportant memories.
This should improve activation-recall correlation without hurting recall.
"""


def compute_decay(activation, impact, stability, mtype, params):
    alpha = params.get("alpha", 0.8)
    rho = params.get("stability_weight", 0.8)
    impact_factor = 1.0 + alpha * impact
    stability_factor = 1.0 + rho * stability
    combined = max(impact_factor * stability_factor, 1e-9)

    lam = params.get("lambda_fact", 0.012) if mtype == "fact" else params.get("lambda_episode", 0.022)
    decay_rate = lam * activation / combined
    decayed = activation * (1.0 - decay_rate)

    # Impact-based floor: high-impact memories never fully decay
    floor_weight = params.get("floor_weight", 0.4)
    floor = floor_weight * impact * impact
    return max(decayed, floor)
