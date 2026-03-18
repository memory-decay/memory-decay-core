import math

def compute_decay(activation: float, impact: float, stability: float, mtype: str, params: dict) -> float:
    lambda_fact = params.get("lambda_fact", 0.015)
    lambda_episode = params.get("lambda_episode", 0.025)
    alpha = params.get("alpha", 1.5)
    rho = params.get("stability_weight", 1.0)
    floor_max = params.get("floor_max", 0.65)
    sigmoid_k = params.get("sigmoid_k", 20.0)
    sigmoid_mid = params.get("sigmoid_mid", 0.20)
    prune_factor = params.get("prune_factor", 5.0)
    
    importance = (impact * alpha + stability * rho) / (alpha + rho)
    
    if importance - sigmoid_mid >= 0:
        sigmoid = 1.0 / (1.0 + math.exp(-sigmoid_k * (importance - sigmoid_mid)))
    else:
        z = math.exp(sigmoid_k * (importance - sigmoid_mid))
        sigmoid = z / (1.0 + z)
        
    floor = floor_max * sigmoid
    floor = min(floor, activation)
    
    retention = max(1.0 + alpha * impact + rho * stability, 1.0)
    
    base_lambda = lambda_fact if mtype == "fact" else lambda_episode
    effective_lambda = base_lambda / retention
    
    if importance < sigmoid_mid:
        effective_lambda *= prune_factor
    
    new_activation = floor + (activation - floor) * math.exp(-effective_lambda)
    
    return max(0.0, min(new_activation, activation))
