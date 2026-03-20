"""Jost's Law decay: activation-dependent rate + sigmoid floor."""

import math


def compute_decay(activation, impact, stability, mtype, params):
    if activation <= 0:
        return 0.0

    # --- Floor: sigmoid function of importance ---
    alpha = params.get("alpha", 1.5)
    rho = params.get("stability_weight", 1.0)
    importance = (impact * alpha + stability * rho) / (alpha + rho)

    floor_max = params.get("floor_max", 0.45)  # Reduced from 0.60 per reviewer feedback
    sigmoid_k = params.get("sigmoid_k", 20.0)
    sigmoid_mid = params.get("sigmoid_mid", 0.30)  # Raised from 0.25 per reviewer feedback

    z = sigmoid_k * (importance - sigmoid_mid)
    if z >= 0:
        sigmoid = 1.0 / (1.0 + math.exp(-z))
    else:
        ez = math.exp(z)
        sigmoid = ez / (1.0 + ez)

    floor = floor_max * sigmoid
    floor = min(floor, activation)

    # --- Jost decay: rate proportional to activation ---
    # da/dt ~ -lambda * a^jost_power  (jost_power > 1 => faster when high)
    lambda_fact = params.get("lambda_fact", 0.015)
    lambda_episode = params.get("lambda_episode", 0.040)
    base_lambda = lambda_fact if mtype == "fact" else lambda_episode

    # Retention from impact + stability
    retention = max(1.0 + alpha * impact + rho * stability, 1.0)
    effective_lambda = base_lambda / retention

    # Jost: decay rate scales with how far above floor we are
    # This naturally slows decay as activation approaches floor
    excess = max(activation - floor, 0.0)
    jost_power = params.get("jost_power", 1.5)
    decay_amount = effective_lambda * (excess ** jost_power)

    new_activation = activation - decay_amount
    new_activation = max(new_activation, floor)

    return max(0.0, min(new_activation, activation))
