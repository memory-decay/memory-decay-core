def compute_decay(activation: float, impact: float, stability: float, mtype: str, params: dict) -> float:
    """
    Sigmoid-gated floor: strictly prune the weakest memories, strongly protect the rest.
    """
    import math

    lambda_base = params.get("lambda_fact", 0.02) if mtype == "fact" else params.get("lambda_episode", 0.035)
    alpha = params.get("alpha", 1.5)
    rho = params.get("stability_weight", 1.0)
    floor_max = params.get("floor_max", 0.5)
    
    # Combined importance score
    importance = (impact * alpha + stability * rho) / (alpha + rho)
    
    # Sigmoid parameters
    k = params.get("sigmoid_k", 20.0) # steepness
    mid = params.get("sigmoid_mid", 0.25) # threshold
    
    # Floor drops sharply to 0 for importance < 0.25, and hits floor_max for importance > 0.25
    sigmoid = 1.0 / (1.0 + math.exp(-k * (importance - mid)))
    floor = floor_max * sigmoid
    floor = min(floor, activation)
    
    # Retention factor
    retention = max(1.0 + alpha * impact + rho * stability, 1.0)
    
    # Decay rate
    effective_lambda = lambda_base / retention
    
    # If importance is very low, accelerate decay
    if importance < mid:
        effective_lambda *= 2.5
    
    new_activation = floor + (activation - floor) * math.exp(-effective_lambda)
    
    return max(0.0, min(new_activation, activation))
