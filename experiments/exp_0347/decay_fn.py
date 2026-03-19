"""Super-linear retention via square root: retention = 1 + alpha*sqrt(impact) + rho*stability.

Replaces the linear retention (1 + alpha*impact + rho*stability) with a concave form.
Concave retention: low-impact memories get less retention boost (decay faster),
clearing distractors sooner. High-impact memories still get strong protection.
This is the opposite of the failed quadratic attempts that hurt weak memories.
"""

import math


def compute_decay(activation, impact, stability, mtype, params):
    if activation <= 0:
        return 0.0

    alpha = params.get("alpha", 1.5)
    rho = params.get("stability_weight", 1.0)

    # Concave (sqrt) retention: low-impact memories decay faster,
    # high-impact memories still well-protected
    retention = max(1.0 + alpha * math.sqrt(impact) + rho * stability, 1.0)

    # --- Floor: sigmoid function of importance (linear, unchanged from best) ---
    floor_max = params.get("floor_max", 0.55)
    sigmoid_k = params.get("sigmoid_k", 20.0)
    sigmoid_mid = params.get("sigmoid_mid", 0.25)

    importance = (impact * alpha + stability * rho) / (alpha + rho)

    z = sigmoid_k * (importance - sigmoid_mid)
    if z >= 0:
        sigmoid = 1.0 / (1.0 + math.exp(-z))
    else:
        ez = math.exp(z)
        sigmoid = ez / (1.0 + ez)

    floor = floor_max * sigmoid
    floor = min(floor, activation)

    # --- Jost decay ---
    lambda_fact = params.get("lambda_fact", 0.015)
    lambda_episode = params.get("lambda_episode", 0.040)
    base_lambda = lambda_fact if mtype == "fact" else lambda_episode

    effective_lambda = base_lambda / retention

    excess = max(activation - floor, 0.0)
    jost_power = params.get("jost_power", 4.0)
    decay_amount = effective_lambda * (excess ** jost_power)

    new_activation = activation - decay_amount
    new_activation = max(new_activation, floor)

    return max(0.0, min(new_activation, activation))
