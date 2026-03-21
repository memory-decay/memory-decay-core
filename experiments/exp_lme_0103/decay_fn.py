"""Jost's Law decay with zero-impact bypass: pure exponential decay for impact=0.

Same as exp_lme_0068. Tests with LOWER stability_weight (0.5 vs 1.0).
"""

import math


def compute_decay(activation, impact, stability, mtype, params):
    if activation <= 0:
        return 0.0

    if impact <= 0:
        lambda_fact = params.get("lambda_fact", 0.009)
        lambda_episode = params.get("lambda_episode", 0.040)
        base_lambda = lambda_fact if mtype == "fact" else lambda_episode
        effective_lambda = base_lambda * 3.0
        new_activation = activation * math.exp(-effective_lambda)
        return max(0.0, new_activation)

    alpha = params.get("alpha", 1.5)
    rho = params.get("stability_weight", 1.0)
    importance = (impact * alpha + stability * rho) / (alpha + rho)

    floor_max = params.get("floor_max", 0.45)
    sigmoid_k = params.get("sigmoid_k", 30.0)
    sigmoid_mid = params.get("sigmoid_mid", 0.32)

    z = sigmoid_k * (importance - sigmoid_mid)
    if z >= 0:
        sigmoid = 1.0 / (1.0 + math.exp(-z))
    else:
        ez = math.exp(z)
        sigmoid = ez / (1.0 + ez)

    floor = floor_max * sigmoid
    floor = min(floor, activation)

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
