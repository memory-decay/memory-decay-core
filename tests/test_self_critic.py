"""Tests for SelfCritic."""

from __future__ import annotations

from unittest.mock import patch, MagicMock


def test_self_critic_generates_critique():
    """SelfCritic.critique_round returns structured critique dict."""
    from memory_decay.self_critic import SelfCritic

    # Mock the OpenAI API call
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = '{"observations": ["test obs"], "diagnosis": "test diag", "next_direction": "try harder", "expected_impact": "MEDIUM", "risk": "low"}'

    with patch("memory_decay.self_critic.OpenAI") as mock_openai:
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client

        critic = SelfCritic(api_key="test-key")
        result = critic.critique_round(
            memory_chain_summary="Round 0: baseline",
            history_last_10=[{"experiment": "exp_0000", "overall_score": 0.02}],
            round_num=1,
        )

    assert "observations" in result
    assert "next_direction" in result
    assert "diagnosis" in result
    assert result["round_num"] == 1


def test_self_critic_api_failure_returns_default():
    """API failure returns a graceful fallback dict."""
    from memory_decay.self_critic import SelfCritic

    with patch("memory_decay.self_critic.OpenAI") as mock_openai:
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = Exception("API down")
        mock_openai.return_value = mock_client

        critic = SelfCritic(api_key="test-key")
        result = critic.critique_round("summary", [], 1)

    assert result["next_direction"] == "Continue current decay parameter refinement"
    assert result["risk"] == "Unknown due to API failure"


def test_self_critic_json_parse_failure_returns_default():
    """Malformed JSON from API returns graceful fallback."""
    from memory_decay.self_critic import SelfCritic

    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "This is not JSON at all"

    with patch("memory_decay.self_critic.OpenAI") as mock_openai:
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client

        critic = SelfCritic(api_key="test-key")
        result = critic.critique_round("summary", [], 1)

    assert result["observations"] == ["Could not parse critique"]
    assert result["next_direction"] == "Continue parameter refinement"


def test_self_critic_api_failure_returns_round_num():
    """API failure fallback includes round_num."""
    from memory_decay.self_critic import SelfCritic

    with patch("memory_decay.self_critic.OpenAI") as mock_openai:
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = Exception("API down")
        mock_openai.return_value = mock_client

        critic = SelfCritic(api_key="test-key")
        result = critic.critique_round("summary", [], 42)

    assert result["round_num"] == 42


def test_self_critic_json_code_block_extraction():
    """API response wrapped in ```json ... ``` is parsed correctly."""
    from memory_decay.self_critic import SelfCritic

    raw_json = '{"observations": ["code block test"], "diagnosis": "test", "next_direction": "go north", "expected_impact": "HIGH", "risk": "low"}'
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = f"```json\n{raw_json}\n```"

    with patch("memory_decay.self_critic.OpenAI") as mock_openai:
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client

        critic = SelfCritic(api_key="test-key")
        result = critic.critique_round("summary", [], 1)

    assert result["observations"] == ["code block test"]
    assert result["next_direction"] == "go north"
    assert result["round_num"] == 1


def test_self_critic_regex_fallback_parses_raw_json():
    """API response with raw JSON (no code block) uses regex fallback."""
    from memory_decay.self_critic import SelfCritic

    raw_json = '{"observations": ["regex fallback test"], "diagnosis": "test", "next_direction": "go south", "expected_impact": "LOW", "risk": "medium"}'
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = raw_json  # No ```json wrapper

    with patch("memory_decay.self_critic.OpenAI") as mock_openai:
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client

        critic = SelfCritic(api_key="test-key")
        result = critic.critique_round("summary", [], 1)

    assert result["observations"] == ["regex fallback test"]
    assert result["next_direction"] == "go south"
    assert result["round_num"] == 1


def test_critique_from_chain():
    """critique_from_chain loads chain/history and calls critique_round."""
    from memory_decay.self_critic import SelfCritic
    from pathlib import Path
    import tempfile
    import json

    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = '{"observations": ["chain test"], "diagnosis": "test", "next_direction": "go east", "expected_impact": "MEDIUM", "risk": "low"}'

    with patch("memory_decay.self_critic.OpenAI") as mock_openai:
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client

        with patch("memory_decay.memory_chain_indexer.MemoryChainIndexer") as mock_idx_cls:
            mock_idx = MagicMock()
            mock_idx.read_chain_summary.return_value = "chain summary here"
            mock_idx_cls.return_value = mock_idx

            with tempfile.TemporaryDirectory() as tmpdir:
                chain_dir = Path(tmpdir) / "chain"
                chain_dir.mkdir()
                history_path = Path(tmpdir) / "history.jsonl"
                history_path.write_text(json.dumps({"experiment": "exp_0001", "overall_score": 0.1}) + "\n")

                critic = SelfCritic(api_key="test-key")
                result = critic.critique_from_chain(chain_dir, history_path, round_num=7)

    assert result["observations"] == ["chain test"]
    assert result["round_num"] == 7
    mock_idx.read_chain_summary.assert_called_once()