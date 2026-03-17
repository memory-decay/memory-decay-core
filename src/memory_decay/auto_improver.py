"""AutoImprover: LLM-driven iterative parameter optimization.

Uses the Anthropic Claude API to analyze evaluation results and propose
modifications to DecayEngine parameters (lambda, beta, alpha, boost magnitude,
activation threshold).

Supports three program.md guidance levels:
- Minimal: "Improve recall_rate while maintaining precision > 0.8"
- Default: describes decay functions, re-activation, and impact system
- Expert: references Ebbinghaus forgetting curve, spacing effect, levels of processing theory

Runs an iterative improvement loop with a configurable budget (N iterations),
where each iteration evaluates current performance, sends results to the LLM agent,
and applies the proposed parameter changes.

Key methods (to implement):
- improve(): run the iterative improvement loop
- analyze_results(): send evaluation metrics to the LLM and get proposed changes
- apply_changes(): update DecayEngine parameters based on LLM suggestions
"""
