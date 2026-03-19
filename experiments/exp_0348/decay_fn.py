"""Tanh-of-squared excess decay: decay = lambda * tanh(excess^2 / scale^2).

Compared to power-jost (excess^jost_power):
- Power: extremely gentle near floor (excess^4 ≈ 0), extremely aggressive far out
- Tanh-sq: gentle near floor (squared excess → tanh → linear), moderate saturation far out

This is a smoother alternative to power-jost, preserving more memories in the
mid-activation range while still having a saturation effect at high excess.
"""

import math


def compute_decay(activation, impact, stability, mtype, params):
    if activation <= 0:
        return 0.0

    alpha = params.get("alpha", 1.5)
    rho = params.get("stability_weight", 1.0)
    importance = (impact * alpha + stability * rho) / (alpha + rho)

    floor_max = params.get("floor_max", 0.55)
    sigmoid_k = params.get("sigmoid_k", 20.0)
    sigmoid_mid = params.get("sigmoid_mid", 0.25)

    z = sigmoid_k * (importance - sigmoid_mid)
    if z >= 0:
        sigmoid = 1.0 / (1.0 + math.exp(-z))
    else:
        ez = math.exp(z)
        sigmoid = ez / (1.0 + ez)

    floor = floor_max * sigmoid
    floor = min(floor, activation)

    lambda_fact = params.get("lambda_fact", 0.015)
    lambda_episode = params.get("lambda_episode", 0.040)
    base_lambda = lambda_fact if mtype == "fact" else lambda_episode

    retention = max(1.0 + alpha * impact + rho * stability, 1.0)
    effective_lambda = base_lambda / retention

    excess = max(activation - floor, 0.0)
    tanh_scale = params.get("tanh_scale", 0.30)
    decay_amount = effective_lambda * math.tanh((excess ** 2) / (tanh_scale ** 2))

    new_activation = activation - decay_amount
    new_activation = max(new_activation, floor)

    return max(0.0, min(new_activation, activation))
