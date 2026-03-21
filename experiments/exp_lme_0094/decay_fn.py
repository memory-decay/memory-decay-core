"""Pure exponential decay: no sigmoid floor, no Jost power law.

Tests if the complexity of Jost+sigmoid is unnecessary with retrieval_consolidation.
Simpler decay might perform better if retrieval_consolidation handles reinforcement well.
"""

import math


def compute_decay(activation, impact, stability, mtype, params):
    if activation <= 0:
        return 0.0

    lambda_fact = params.get("lambda_fact", 0.009)
    lambda_episode = params.get("lambda_episode", 0.040)
    base_lambda = lambda_fact if mtype == "fact" else lambda_episode

    # Simple: lambda scales inversely with stability
    stability_weight = params.get("stability_weight", 1.0)
    retention = 1.0 + stability_weight * stability
    effective_lambda = base_lambda / retention

    new_activation = activation * math.exp(-effective_lambda)
    return max(0.0, new_activation)
