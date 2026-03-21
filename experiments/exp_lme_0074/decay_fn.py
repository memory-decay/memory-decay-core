import math


def compute_decay(activation, impact, stability, mtype, params):
    """Jost's Law decay with hyperbolic importance floor.

    floor = floor_max * impact / (impact + k_hyp)
    where k_hyp controls the half-life point of the floor transition.
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

    # Hyperbolic floor instead of sigmoid
    floor_max = params.get("floor_max", 0.45)
    k_hyp = params.get("k_hyp", 0.20)  # half-life of importance
    floor = floor_max * importance / (importance + k_hyp)
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