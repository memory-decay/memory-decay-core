"""Impact-floor decay with type-dependent floor scaling."""
import math

def compute_decay(activation, impact, stability, mtype, params):
    alpha = params.get("alpha", 2.0)
    rho = params.get("stability_weight", 0.8)
    combined = max(math.exp(alpha * impact) * (1.0 + rho * stability), 1e-9)
    lam = params.get("lambda_fact", 0.012) if mtype == "fact" else params.get("lambda_episode", 0.036)
    fs = params.get("floor_fact", 0.25) if mtype == "fact" else params.get("floor_episode", 0.40)
    floor = impact * fs
    excess = max(activation - floor, 0.0)
    new_excess = excess * (1.0 - lam * excess / combined)
    return floor + max(new_excess, 0.0)
