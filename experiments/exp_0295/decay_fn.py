def compute_decay(activation: float, impact: float, stability: float, mtype: str, params: dict) -> float:
    """
    Decay function balancing selective pruning with retention.
    Linear floor based on impact. Low impact drops below thresholds, high impact stays alive.
    """
    import math

    lambda_base = params.get("lambda_fact", 0.02) if mtype == "fact" else params.get("lambda_episode", 0.03)
    alpha = params.get("alpha", 1.5)
    rho = params.get("stability_weight", 1.0)
    floor_min = params.get("floor_min", 0.05)
    floor_max = params.get("floor_max", 0.45)
    floor_power = params.get("floor_power", 1.0)
    
    # Combined retention factor
    retention = max(1.0 + alpha * impact + rho * stability, 1.0)
    
    # Determine the floor
    target_floor = floor_min + (floor_max - floor_min) * (impact ** floor_power)
    floor = min(target_floor, activation)
    
    # Effective decay rate
    effective_lambda = lambda_base / retention
    
    # Exponential decay towards the floor
    new_activation = floor + (activation - floor) * math.exp(-effective_lambda)
    
    # Ensure monotonicity and bounds
    return max(0.0, min(new_activation, activation))
