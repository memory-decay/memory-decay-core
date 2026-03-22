"""Hebbian-Decay: distance-from-floor modulation.

Same as exp_lme_0292.
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
        return max(0.0, activation * math.exp(-effective_lambda))
    alpha = params.get("alpha", 1.5)
    rho = params.get("stability_weight", 1.0)
    importance = (impact * alpha + stability * rho) / (alpha + rho)
    floor_max = params.get("floor_max", 0.45)
    sigmoid_k = params.get("sigmoid_k", 30.0)
    sigmoid_mid = params.get("sigmoid_mid", 0.32)
    z = sigmoid_k * (importance - sigmoid_mid)
    sigmoid = 1.0 / (1.0 + math.exp(-z)) if z >= 0 else math.exp(z) / (1.0 + math.exp(z))
    floor = min(floor_max * sigmoid, activation)
    lambda_fact = params.get("lambda_fact", 0.009)
    lambda_episode = params.get("lambda_episode", 0.040)
    base_lambda = lambda_fact if mtype == "fact" else lambda_episode
    retention = max(1.0 + alpha * impact + rho * stability, 1.0)
    effective_lambda = base_lambda / retention
    distance_scale = params.get("distance_scale", 0.5)
    distance_from_floor = max(activation - floor, 0.0)
    effective_lambda *= (1.0 + distance_scale * (distance_from_floor / max(activation, 0.01)))
    excess = max(activation - floor, 0.0)
    jost_power = params.get("jost_power", 3.0)
    new_activation = activation - effective_lambda * (excess ** jost_power)
    return max(0.0, min(new_activation, floor))
