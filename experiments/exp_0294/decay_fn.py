def compute_decay(activation: float, impact: float, stability: float, mtype: str, params: dict) -> float:
    """
    Decay function focused on "Selective Forgetting" to maximize precision_lift.
    
    Strategy:
    - High impact/stability nodes get a strong protective shield (floor) and slow decay.
    - Low impact/stability nodes have zero floor and fast decay (rapid pruning).
    - Uses a sharp thresholding logic implicitly through mathematics: 
      if combined importance < threshold, decay is aggressive.
    """
    import math

    # Unpack parameters
    lambda_fact = params.get("lambda_fact", 0.05)
    lambda_episode = params.get("lambda_episode", 0.08)
    alpha = params.get("alpha", 2.0)
    rho = params.get("stability_weight", 1.5)
    floor_max = params.get("floor_max", 0.6)
    
    # Calculate combined importance score (0 to ~1)
    importance = (impact * alpha + stability * rho) / (alpha + rho)
    
    # Floor is only given to reasonably important memories
    # Using quadratic scaling so low importance gets almost no floor
    floor = floor_max * (importance ** 2)
    
    # Decay rate is base rate divided by importance factor
    # Low importance -> high decay. High importance -> low decay.
    base_lambda = lambda_fact if mtype == "fact" else lambda_episode
    
    # If importance is very low, accelerate decay significantly (pruning)
    prune_multiplier = 1.0
    if importance < 0.3:
        prune_multiplier = 3.0
        
    effective_lambda = (base_lambda * prune_multiplier) / max(1.0 + alpha * impact + rho * stability, 1.0)
    
    # Compute standard exponential decay towards the floor
    new_activation = floor + (activation - floor) * math.exp(-effective_lambda)
    
    # Ensure monotonicity (never increase)
    new_activation = min(activation, new_activation)
    
    # Ensure bounded 0-1
    return max(0.0, new_activation)
