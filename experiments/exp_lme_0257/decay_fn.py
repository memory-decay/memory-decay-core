"""Adaptive Threshold Decay: two-phase decay — Jost above threshold, exponential below.

Follow-up to exp_lme_0234 with adaptive threshold decay.
Hypothesis: Two-phase decay — above importance threshold use Jost with no floor,
below threshold use exponential decay to zero.
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

    # Adaptive Threshold Decay: two-phase approach
    importance_threshold = params.get("importance_threshold", 0.30)

    if importance >= importance_threshold:
        # Above threshold: Jost decay with no floor
        jost_power = params.get("jost_power", 1.0)
        decay_amount = effective_lambda * (activation ** jost_power)
        new_activation = activation - decay_amount
        # No floor enforcement above threshold
    else:
        # Below threshold: exponential decay to zero (no floor)
        new_activation = activation * math.exp(-effective_lambda * 2.0)

    return max(0.0, min(new_activation, activation))
