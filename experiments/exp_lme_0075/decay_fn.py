import math


def compute_decay(activation, impact, stability, mtype, params):
    """Two-phase decay: active phase (above threshold) uses Jost with no floor;
    inactive phase (below threshold) uses exponential with a hard floor.

    The threshold itself is sigmoid-shaped in importance.
    """
    if activation <= 0:
        return 0.0

    if impact <= 0:
        lambda_fact = params.get("lambda_fact", 0.009)
        lambda_episode = params.get("lambda_episode", 0.040)
        base_lambda = lambda_fact if mtype == "fact" else lambda_episode
        effective_lambda = base_lambda * 3.0
        return max(0.0, activation * math.exp(-effective_lambda))

    alpha = params.get("alpha", 1.5)
    rho = params.get("stability_weight", 1.0)
    importance = (impact * alpha + stability * rho) / (alpha + rho)

    # Compute activation threshold (sigmoid-shaped in importance)
    floor_max = params.get("floor_max", 0.45)
    sigmoid_k = params.get("sigmoid_k", 30.0)
    sigmoid_mid = params.get("sigmoid_mid", 0.32)
    z = sigmoid_k * (importance - sigmoid_mid)
    if z >= 0:
        sigmoid = 1.0 / (1.0 + math.exp(-z))
    else:
        sigmoid = math.exp(z) / (1.0 + math.exp(z))
    threshold = floor_max * sigmoid

    lambda_fact = params.get("lambda_fact", 0.009)
    lambda_episode = params.get("lambda_episode", 0.040)
    base_lambda = lambda_fact if mtype == "fact" else lambda_episode

    retention = max(1.0 + alpha * impact + rho * stability, 1.0)
    effective_lambda = base_lambda / retention

    if activation > threshold:
        # Active phase: Jost decay, no floor constraint (floor is the threshold itself)
        excess = max(activation - threshold, 0.0)
        jost_power = params.get("jost_power", 4.0)
        decay_amount = effective_lambda * (excess ** jost_power)
        new_activation = activation - decay_amount
        new_activation = max(new_activation, threshold)
    else:
        # Inactive phase: exponential decay toward zero
        new_activation = activation * math.exp(-effective_lambda * 2.0)

    return max(0.0, min(new_activation, activation))