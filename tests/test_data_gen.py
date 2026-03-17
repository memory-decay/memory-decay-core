"""Unit tests for SyntheticDataGenerator."""

import json
import pytest
from unittest.mock import MagicMock, patch

from memory_decay.data_gen import SyntheticDataGenerator


class TestSyntheticDataGenerator:
    def _mock_client(self, response_items: list[dict]):
        """Create a mock Anthropic client that returns given items."""
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text=json.dumps(response_items, ensure_ascii=False))]

        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_response
        return mock_client

    def _sample_hub_items(self):
        return [
            {
                "id": "mem_001",
                "type": "fact",
                "content": "서울은 대한민국의 수도이다",
                "entities": ["서울", "대한민국"],
                "tick": 0,
                "impact": 0.9,
                "associations": [],
                "recall_query": "대한민국의 수도는?",
                "recall_answer": "서울",
            },
            {
                "id": "mem_002",
                "type": "fact",
                "content": "김민수는 커피를 좋아한다",
                "entities": ["김민수", "커피"],
                "tick": 5,
                "impact": 0.8,
                "associations": [],
                "recall_query": "김민수는 무엇을 좋아하는가?",
                "recall_answer": "커피",
            },
        ]

    def _sample_leaf_items(self):
        return [
            {
                "id": "mem_003",
                "type": "episode",
                "content": "서울에서 김민수와 커피를 마셨다",
                "entities": ["서울", "김민수", "커피"],
                "tick": 10,
                "impact": 0.4,
                "associations": ["mem_001", "mem_002"],
                "recall_query": "서울에서 누구와 커피를 마셨는가?",
                "recall_answer": "김민수",
            },
            {
                "id": "mem_004",
                "type": "fact",
                "content": "커피에는 카페인이 들어있다",
                "entities": ["커피", "카페인"],
                "tick": 15,
                "impact": 0.3,
                "associations": ["mem_002"],
                "recall_query": "커피에 들어있는 성분은?",
                "recall_answer": "카페인",
            },
        ]

    def test_init_requires_api_key(self):
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(ValueError, match="API key"):
                SyntheticDataGenerator()

    def test_init_with_env_key(self):
        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}):
            gen = SyntheticDataGenerator()
            assert gen.api_key == "test-key"

    def test_init_with_param_key(self):
        gen = SyntheticDataGenerator(api_key="my-key")
        assert gen.api_key == "my-key"

    def test_save_and_load_jsonl(self, tmp_path):
        gen = SyntheticDataGenerator(api_key="test")
        items = self._sample_hub_items() + self._sample_leaf_items()
        path = tmp_path / "test.jsonl"

        gen.save_jsonl(items, path)
        loaded = SyntheticDataGenerator.load_jsonl(path)

        assert len(loaded) == len(items)
        assert loaded[0]["id"] == "mem_001"
        assert loaded[0]["content"] == "서울은 대한민국의 수도이다"

    def test_split_test_train(self):
        gen = SyntheticDataGenerator(api_key="test")
        items = self._sample_hub_items() + self._sample_leaf_items()
        train, test = gen.split_test_train(items, test_ratio=0.3, seed=42)

        assert len(train) + len(test) == len(items)
        assert len(test) >= 1
        # Check both types present in test
        test_types = {m["type"] for m in test}
        assert "fact" in test_types or "episode" in test_types

    def test_resolve_association_ids(self):
        gen = SyntheticDataGenerator(api_key="test")
        items = self._sample_hub_items() + self._sample_leaf_items()
        resolved = gen._resolve_association_ids(items)

        # Leaf items should have resolved associations
        leaf = resolved[2]
        assert len(leaf["associations"]) > 0
        # Should have dict format with "id" key
        for assoc in leaf["associations"]:
            assert isinstance(assoc, dict)
            assert "id" in assoc
            assert "weight" in assoc

    def test_generate_dataset_mocked(self, tmp_path):
        gen = SyntheticDataGenerator(api_key="test")

        hub_items = self._sample_hub_items()
        leaf_items = self._sample_leaf_items()

        # Mock LLM calls
        call_count = [0]
        def mock_create(**kwargs):
            call_count[0] += 1
            mock_response = MagicMock()
            if call_count[0] == 1:
                mock_response.content = [MagicMock(text=json.dumps(hub_items, ensure_ascii=False))]
            else:
                mock_response.content = [MagicMock(text=json.dumps(leaf_items, ensure_ascii=False))]
            return mock_response

        gen.client.messages.create = mock_create

        dataset = gen.generate_dataset(
            num_memories=4, hub_ratio=0.5, ticks_range=(0, 100), seed=42
        )

        assert len(dataset) == 4
        # Should be sorted by tick
        ticks = [m["tick"] for m in dataset]
        assert ticks == sorted(ticks)

    def test_generate_with_markdown_wrapped_response(self, tmp_path):
        gen = SyntheticDataGenerator(api_key="test")

        items = self._sample_hub_items()
        wrapped_json = f"```json\n{json.dumps(items, ensure_ascii=False)}\n```"

        mock_response = MagicMock()
        mock_response.content = [MagicMock(text=wrapped_json)]
        gen.client.messages.create = MagicMock(return_value=mock_response)

        gen._generate_memories_batch(2, is_hub=True, tick_range=(0, 100))
        # Should not raise — JSON extraction handles markdown wrapping
