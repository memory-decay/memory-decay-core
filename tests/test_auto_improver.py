"""Unit tests for AutoImprover."""

import json
import pytest
from unittest.mock import MagicMock

from memory_decay.auto_improver import AutoImprover, GUIDANCE


class TestAutoImprover:
    def _make_mock_response(self, params: dict, reasoning: str = "test"):
        text = json.dumps({"reasoning": reasoning, "parameters": params}, ensure_ascii=False)
        mock_choice = MagicMock()
        mock_choice.message.content = text
        mock_resp = MagicMock()
        mock_resp.choices = [mock_choice]
        return mock_resp

    def test_init_invalid_guidance(self):
        with pytest.raises(ValueError, match="Invalid guidance_level"):
            AutoImprover(api_key="test", guidance_level="nonexistent")

    def test_guidance_levels_exist(self):
        assert "minimal" in GUIDANCE
        assert "default" in GUIDANCE
        assert "expert" in GUIDANCE

    def test_propose_parameters(self):
        improver = AutoImprover(api_key="test", guidance_level="default")
        new_params = {
            "lambda_fact": 0.03, "lambda_episode": 0.06,
            "beta_fact": 0.2, "beta_episode": 0.4, "alpha": 0.7,
            "stability_weight": 0.9, "stability_decay": 0.02,
            "reinforcement_gain_direct": 0.25, "reinforcement_gain_assoc": 0.06,
            "stability_cap": 1.2,
        }
        improver.client.chat.completions.create = MagicMock(
            return_value=self._make_mock_response(new_params)
        )

        current = {
            "lambda_fact": 0.05, "lambda_episode": 0.08,
            "beta_fact": 0.3, "beta_episode": 0.5, "alpha": 0.5,
            "stability_weight": 0.8, "stability_decay": 0.01,
            "reinforcement_gain_direct": 0.2, "reinforcement_gain_assoc": 0.05,
            "stability_cap": 1.0,
        }

        result = improver.propose_parameters(current, [], iteration=1, total_budget=10)
        assert result["lambda_fact"] == 0.03
        assert result["alpha"] == 0.7
        assert result["stability_weight"] == 0.9
        assert len(improver.history) == 1

    def test_validate_params_clamps_values(self):
        improver = AutoImprover(api_key="test")

        proposed = {
            "lambda_fact": 999.0,
            "lambda_episode": -1.0,
            "beta_fact": "not_a_number",
            "beta_episode": 1.0,
            "alpha": 0.5,
            "stability_weight": 9.0,
            "stability_decay": -1.0,
            "reinforcement_gain_direct": 2.0,
            "reinforcement_gain_assoc": "bad",
            "stability_cap": 99.0,
        }
        current = {
            "lambda_fact": 0.05, "lambda_episode": 0.08,
            "beta_fact": 0.3, "beta_episode": 0.5, "alpha": 0.5,
            "stability_weight": 0.8, "stability_decay": 0.01,
            "reinforcement_gain_direct": 0.2, "reinforcement_gain_assoc": 0.05,
            "stability_cap": 1.0,
        }

        result = improver._validate_params(proposed, current)
        assert result["lambda_fact"] <= 0.5
        assert result["lambda_episode"] >= 0.001
        assert result["beta_fact"] == 0.3
        assert result["beta_episode"] == 1.0
        assert result["stability_weight"] <= 5.0
        assert result["stability_decay"] >= 0.0
        assert result["reinforcement_gain_assoc"] == 0.05
        assert result["stability_cap"] <= 5.0

    def test_should_stop_budget_exhausted(self):
        improver = AutoImprover(api_key="test")
        assert improver.should_stop([], iteration=12, total_budget=12)

    def test_should_stop_no_improvement(self):
        improver = AutoImprover(api_key="test")
        history = [
            {"composite_score": 0.5}, {"composite_score": 0.5},
            {"composite_score": 0.5}, {"composite_score": 0.5},
            {"composite_score": 0.5},
        ]
        assert improver.should_stop(history, iteration=5, total_budget=12, patience=4)

    def test_should_stop_memorization_detected(self):
        improver = AutoImprover(api_key="test")
        history = [
            {"composite_score": 0.8, "recall_rate": 0.96},
            {"composite_score": 0.8, "recall_rate": 0.97},
            {"composite_score": 0.8, "recall_rate": 0.98},
        ]
        assert improver.should_stop(history, iteration=3, total_budget=10)

    def test_should_not_stop_early(self):
        improver = AutoImprover(api_key="test")
        history = [
            {"composite_score": 0.3}, {"composite_score": 0.5},
            {"composite_score": 0.7},
        ]
        assert not improver.should_stop(history, iteration=3, total_budget=12)

    def test_propose_handles_markdown_wrapped(self):
        improver = AutoImprover(api_key="test")
        params = {
            "lambda_fact": 0.04, "lambda_episode": 0.07,
            "beta_fact": 0.25, "beta_episode": 0.45, "alpha": 0.6,
            "stability_weight": 0.9, "stability_decay": 0.01,
            "reinforcement_gain_direct": 0.22, "reinforcement_gain_assoc": 0.055,
            "stability_cap": 1.1,
        }
        text = f"Here are my suggestions:\n```json\n{json.dumps({'reasoning': 'analysis', 'parameters': params}, ensure_ascii=False)}\n```"
        mock_choice = MagicMock()
        mock_choice.message.content = text
        mock_resp = MagicMock()
        mock_resp.choices = [mock_choice]
        improver.client.chat.completions.create = MagicMock(return_value=mock_resp)

        current = {
            "lambda_fact": 0.05, "lambda_episode": 0.08,
            "beta_fact": 0.3, "beta_episode": 0.5, "alpha": 0.5,
            "stability_weight": 0.8, "stability_decay": 0.01,
            "reinforcement_gain_direct": 0.2, "reinforcement_gain_assoc": 0.05,
            "stability_cap": 1.0,
        }
        result = improver.propose_parameters(current, [], iteration=1, total_budget=5)
        assert result["lambda_fact"] == 0.04

    def test_build_prompt_includes_score_breakdown(self):
        improver = AutoImprover(api_key="test", guidance_level="expert")
        prompt = improver._build_prompt(
            current_params={
                "lambda_fact": 0.05, "lambda_episode": 0.08,
                "beta_fact": 0.3, "beta_episode": 0.5, "alpha": 0.5,
                "stability_weight": 0.8, "stability_decay": 0.01,
                "reinforcement_gain_direct": 0.2, "reinforcement_gain_assoc": 0.05,
                "stability_cap": 1.0,
            },
            evaluation_history=[{
                "overall_score": 0.5,
                "retrieval_score": 0.4,
                "plausibility_score": 0.6,
                "threshold_metrics": {
                    "0.2": {"recall_rate": 0.5, "precision_rate": 0.2},
                },
            }],
            iteration=1,
            total_budget=12,
        )
        assert "retrieval_score" in prompt
        assert "plausibility_score" in prompt
        assert "threshold_metrics" in prompt
