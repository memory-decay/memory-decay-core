"""Jost decay with a dual-sigmoid floor."""

import math


def _sigmoid(x):
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    ex = math.exp(x)
    return ex / (1.0 + ex)


def compute_decay(activation, impact, stability, mtype, params):
    if activation <= 0:
        return 0.0

    alpha = params.get("alpha", 1.5)
    rho = params.get("stability_weight", 1.0)
    importance = (impact * alpha + stability * rho) / (alpha + rho)

    floor_max = params.get("floor_max", 0.60)
    sigmoid_k1 = params.get("sigmoid_k1", 18.0)
    sigmoid_mid1 = params.get("sigmoid_mid1", 0.20)
    sigmoid_k2 = params.get("sigmoid_k2", 28.0)
    sigmoid_mid2 = params.get("sigmoid_mid2", 0.42)
    mix = params.get("floor_mix", 0.72)

    floor_shape = mix * _sigmoid(sigmoid_k1 * (importance - sigmoid_mid1))
    floor_shape += (1.0 - mix) * _sigmoid(sigmoid_k2 * (importance - sigmoid_mid2))
    floor = floor_max * floor_shape
    floor = min(floor, activation)

    lambda_fact = params.get("lambda_fact", 0.008)
    lambda_episode = params.get("lambda_episode", 0.035)
    base_lambda = lambda_fact if mtype == "fact" else lambda_episode

    retention = max(1.0 + alpha * impact + rho * stability, 1.0)
    effective_lambda = base_lambda / retention

    excess = max(activation - floor, 0.0)
    jost_power = params.get("jost_power", 4.0)
    decay_amount = effective_lambda * (excess ** jost_power)

    new_activation = activation - decay_amount
    new_activation = max(new_activation, floor)

    return max(0.0, min(new_activation, activation))
