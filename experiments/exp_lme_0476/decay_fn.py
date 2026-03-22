"""Hebbian-Decay with Extreme Pruning for Low-Impact Memories.

Based on exp_lme_0411's winning configuration.
Key mechanism: prune_threshold aggressively removes low-importance distractors,
creating a bimodal distribution that separates important memories (floor ~0.68)
from unimportant ones (fast decay to near 0).
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

    # Extreme Pruning: low-importance memories decay extremely fast
    prune_threshold = params.get("prune_threshold", 0.20)
    if importance < prune_threshold:
        fast_lambda = params.get("prune_fast_lambda", 0.1)
        return max(0.0, activation * math.exp(-fast_lambda))

    floor_max = params.get("floor_max", 0.70)
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

    # Hebbian-Decay: distance-from-floor modulation
    distance_scale = params.get("distance_scale", 0.5)
    distance_from_floor = max(activation - floor, 0.0)
    effective_lambda *= (1.0 + distance_scale * (distance_from_floor / max(activation, 0.01)))

    excess = max(activation - floor, 0.0)
    jost_power = params.get("jost_power", 1.0)
    decay_amount = effective_lambda * (excess ** jost_power)

    new_activation = activation - decay_amount
    new_activation = max(new_activation, floor)

    return max(0.0, min(new_activation, activation))
