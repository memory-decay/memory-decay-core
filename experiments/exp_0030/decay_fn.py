"""Quadratic decay, lambda_fact=0.015, lambda_episode=0.030."""
import math

def compute_decay(activation, impact, stability, mtype, params):
    alpha = params.get("alpha", 2.0)
    rho = params.get("stability_weight", 0.8)
    combined = max(math.exp(alpha * impact) * (1.0 + rho * stability), 1e-9)
    lam = params.get("lambda_fact", 0.015) if mtype == "fact" else params.get("lambda_episode", 0.030)
    return activation * (1.0 - lam * activation / combined)
