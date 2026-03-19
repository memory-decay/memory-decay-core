"""Activation-adaptive sigmoid steepness (gentle version).

k_factor=0.3 (much gentler than exp_0357's 1.0 which hurt recall).
Plus lower lambda_fact=0.007 to compensate for any near-floor protection effect.

k_factor=0.3: effective_k = 20 * (1 + 0.3*(1-activation))
- activation=1.0: k=20 (standard)
- activation=0.5: k=23 (15% steeper)
- activation=0.3: k=24.2 (21% steeper)
"""

import math


def compute_decay(activation, impact, stability, mtype, params):
    if activation <= 0:
        return 0.0

    alpha = params.get("alpha", 1.5)
    rho = params.get("stability_weight", 1.0)
    importance = (impact * alpha + stability * rho) / (alpha + rho)

    floor_max = params.get("floor_max", 0.55)
    base_k = params.get("sigmoid_k", 20.0)
    sigmoid_mid = params.get("sigmoid_mid", 0.25)
    adaptive_factor = params.get("adaptive_k_factor", 0.3)

    effective_k = base_k * (1.0 + adaptive_factor * (1.0 - activation))

    z = effective_k * (importance - sigmoid_mid)
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
    jost_power = params.get("jost_power", 4.0)
    decay_amount = effective_lambda * (excess ** jost_power)

    new_activation = activation - decay_amount
    new_activation = max(new_activation, floor)

    return max(0.0, min(new_activation, activation))
