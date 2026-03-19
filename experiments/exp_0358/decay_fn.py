"""Activation-adaptive retention: retention boosts for memories far above their floor.

Standard retention = 1 + alpha*impact + rho*stability (independent of activation).
This variant adds a second term: retention also increases when the memory is
far above its floor (high excess_ratio = excess/activation).

Effect: Memories well above their floor get a retention BONUS on top of the
standard impact+stability retention. This creates a "buffer" effect —
strong memories stay stronger longer. Near-floor memories get no bonus,
decaying at the base rate toward their floor.

This is different from jost_power which controls HOW FAST you approach the floor.
This controls HOW MUCH retention you get as a function of WHERE you are relative to floor.
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

    # Standard retention from impact + stability
    retention = max(1.0 + alpha * impact + rho * stability, 1.0)

    # Adaptive bonus: memories far above floor get additional retention boost
    excess = max(activation - floor, 0.0)
    retention_bonus = params.get("retention_bonus", 0.5)
    excess_ratio = excess / activation if activation > 0 else 0.0
    retention = retention * (1.0 + retention_bonus * excess_ratio)

    effective_lambda = base_lambda / retention

    jost_power = params.get("jost_power", 4.0)
    decay_amount = effective_lambda * (excess ** jost_power)

    new_activation = activation - decay_amount
    new_activation = max(new_activation, floor)

    return max(0.0, min(new_activation, activation))
