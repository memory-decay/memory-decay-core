"""Tests for HypothesisGenerator."""

from __future__ import annotations

from unittest.mock import patch, MagicMock


def test_hypothesis_generator_returns_structured_response():
    """HypothesisGenerator.generate_hypothesis returns required keys."""
    from memory_decay.hypothesis_generator import HypothesisGenerator

    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = '{"hypothesis": "test hyp", "decay_fn_code": "def compute_decay(...): pass", "params": {"lambda_fact": 0.015}, "predicted_effect": "improved recall", "risk": "low"}'

    with patch("memory_decay.hypothesis_generator.OpenAI") as mock_openai:
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client

        gen = HypothesisGenerator(api_key="test-key")
        result = gen.generate_hypothesis(
            memory_chain_context="Round 0: baseline",
            current_decay_fn="def compute_decay(...): pass",
            current_params={"lambda_fact": 0.015},
            round_num=1,
        )

    assert "hypothesis" in result
    assert "decay_fn_code" in result
    assert "params" in result
    assert result["round_num"] == 1


def test_hypothesis_generator_api_failure_returns_error():
    """API failure returns an error dict."""
    from memory_decay.hypothesis_generator import HypothesisGenerator

    with patch("memory_decay.hypothesis_generator.OpenAI") as mock_openai:
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = Exception("API down")
        mock_openai.return_value = mock_client

        gen = HypothesisGenerator(api_key="test-key")
        result = gen.generate_hypothesis("ctx", "fn", {}, 1)

    assert "error" in result


def test_hypothesis_generator_json_parse_failure_returns_error():
    """Malformed JSON returns error dict."""
    from memory_decay.hypothesis_generator import HypothesisGenerator

    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "Not JSON at all"

    with patch("memory_decay.hypothesis_generator.OpenAI") as mock_openai:
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client

        gen = HypothesisGenerator(api_key="test-key")
        result = gen.generate_hypothesis("ctx", "fn", {}, 1)

    assert "error" in result
