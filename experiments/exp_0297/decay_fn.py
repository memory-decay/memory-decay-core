import math

def compute_decay(activation: float, impact: float, stability: float, mtype: str, params: dict) -> float:
    # Unpack parameters
    lambda_fact = params.get("lambda_fact", 0.02)
    lambda_episode = params.get("lambda_episode", 0.035)
    alpha = params.get("alpha", 1.5)
    rho = params.get("stability_weight", 1.0)
    floor_max = params.get("floor_max", 0.5)
    sigmoid_k = params.get("sigmoid_k", 15.0)
    sigmoid_mid = params.get("sigmoid_mid", 0.3)
    prune_factor = params.get("prune_factor", 2.5)
    
    # Combined importance score
    importance = (impact * alpha + stability * rho) / (alpha + rho)
    
    # Floor drops sharply to 0 for importance < sigmoid_mid
    if importance - sigmoid_mid >= 0:
        sigmoid = 1.0 / (1.0 + math.exp(-sigmoid_k * (importance - sigmoid_mid)))
    else:
        # avoid overflow
        z = math.exp(sigmoid_k * (importance - sigmoid_mid))
        sigmoid = z / (1.0 + z)
        
    floor = floor_max * sigmoid
    floor = min(floor, activation)
    
    # Retention factor
    retention = max(1.0 + alpha * impact + rho * stability, 1.0)
    
    # Base decay rate
    base_lambda = lambda_fact if mtype == "fact" else lambda_episode
    effective_lambda = base_lambda / retention
    
    # If importance is low, accelerate decay
    if importance < sigmoid_mid:
        effective_lambda *= prune_factor
    
    new_activation = floor + (activation - floor) * math.exp(-effective_lambda)
    
    return max(0.0, min(new_activation, activation))
