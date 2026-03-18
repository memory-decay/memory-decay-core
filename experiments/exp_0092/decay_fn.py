"""Extreme impact sensitivity: alpha=5.0, no floor. exp(5*impact) creates 148x
differential between impact=0 and impact=1 items."""
import math

def compute_decay(activation, impact, stability, mtype, params):
    alpha = params.get("alpha", 5.0)
    rho = params.get("stability_weight", 0.8)
    combined = max(math.exp(alpha * impact) * (1.0 + rho * stability), 1e-9)
    lam = params.get("lambda_fact", 0.025) if mtype == "fact" else params.get("lambda_episode", 0.075)
    return activation * (1.0 - lam * activation / combined)
