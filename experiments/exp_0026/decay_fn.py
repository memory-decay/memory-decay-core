"""Quadratic decay, tuned lambdas (improve on exp_0003).

exp_0003 used lambda_fact=0.018, lambda_episode=0.030.
This experiment tunes to lambda_fact=0.015, lambda_episode=0.035
for slightly more recall from facts (slower decay) and more
episode differentiation (faster episode decay, ratio=2.33).
"""

import math


def compute_decay(activation, impact, stability, mtype, params):
    alpha = params.get("alpha", 2.0)
    rho = params.get("stability_weight", 0.8)
    impact_factor = math.exp(alpha * impact)
    stability_factor = 1.0 + rho * stability
    combined = max(impact_factor * stability_factor, 1e-9)

    lam = params.get("lambda_fact", 0.015) if mtype == "fact" else params.get("lambda_episode", 0.035)
    decay_rate = lam * activation / combined
    return activation * (1.0 - decay_rate)
