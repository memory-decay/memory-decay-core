"""Power-law discriminative floor: impact^2 creates sharp selectivity."""
import math

def compute_decay(activation, impact, stability, mtype, params):
    alpha = params.get("alpha", 2.0)
    rho = params.get("stability_weight", 0.8)
    combined = max(math.exp(alpha * impact) * (1.0 + rho * stability), 1e-9)
    lam = params.get("lambda_fact", 0.015) if mtype == "fact" else params.get("lambda_episode", 0.045)

    # Power-law floor: impact^2 is much more selective than sqrt(impact)
    floor_scale = params.get("floor_scale", 0.50)
    floor = (impact ** 2) * floor_scale

    # Faster base decay to push low-impact items down quickly
    excess = max(activation - floor, 0.0)
    rate = lam / combined
    new_excess = excess * (1.0 - rate)
    return floor + max(new_excess, 0.0)
