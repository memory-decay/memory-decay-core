"""Jost's Law decay - impact-only floor."""

import math


def compute_decay(activation, impact, stability, mtype, params):
    if activation <= 0:
        return 0.0

    # --- Floor: ONLY impact matters, stability only affects retention ---
    alpha = params.get("alpha", 1.5)
    rho = params.get("stability_weight", 1.0)

    floor_max = params.get("floor_max", 0.43)
    sigmoid_k = params.get("sigmoid_k", 35.0)
    sigmoid_mid = params.get("sigmoid_mid", 0.30)

    z = sigmoid_k * (impact - sigmoid_mid)
    if z >= 0:
        sigmoid = 1.0 / (1.0 + math.exp(-z))
    else:
        ez = math.exp(z)
        sigmoid = ez / (1.0 + ez)

    floor = floor_max * sigmoid
    floor = min(floor, activation)

    # --- Jost decay - retention uses BOTH impact and stability ---
    lambda_fact = params.get("lambda_fact", 0.009)
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
