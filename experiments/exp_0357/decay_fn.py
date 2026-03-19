"""Activation-adaptive sigmoid steepness: sigmoid steepness (k) scales with activation level.

The standard sigmoid floor has fixed k=20, mid=0.25 regardless of current activation.
This variant makes the sigmoid steeper when activation is LOW (near floor), creating a
"clutch" effect: memories approaching their floor get stronger floor protection
(less likely to slip below), while high-activation memories decay more aggressively.

Formula: effective_k = base_k * (1 + adaptive_factor * (1 - activation))
- activation near 1.0: effective_k ≈ base_k (standard behavior)
- activation near floor: effective_k ≈ base_k * (1 + adaptive_factor) (steeper = more protection)

This is the opposite of the failed "activation-boosted floor" (exp_0346 attempt 2):
that blended activation INTO the floor value; this makes the floor's steepness
RESPOND to activation level.
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
    adaptive_factor = params.get("adaptive_k_factor", 2.0)

    # Adaptive steepness: steeper when activation is low (near floor)
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
