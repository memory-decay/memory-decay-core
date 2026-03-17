"""Synthetic memory dataset generator using LLM API.

Generates memory items with associations, impact scores, and recall test queries.
Designs hub-and-leaf topology for meaningful re-activation experiments.
"""

from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Optional

import time

from openai import OpenAI


class SyntheticDataGenerator:
    """Generate synthetic memory datasets for simulation experiments."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4o-mini",
        base_url: Optional[str] = None,
    ):
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OpenAI API key required. Set OPENAI_API_KEY env var "
                "or pass api_key parameter."
            )
        kwargs = {"api_key": self.api_key}
        if base_url:
            kwargs["base_url"] = base_url
        self.client = OpenAI(**kwargs)
        self.model = model

    def generate_dataset(
        self,
        num_memories: int = 50,
        hub_ratio: float = 0.2,
        max_associations: int = 4,
        ticks_range: tuple[int, int] = (0, 100),
        seed: int | None = None,
    ) -> list[dict]:
        """Generate a complete synthetic memory dataset."""
        if seed is not None:
            random.seed(seed)

        tick_min, tick_max = ticks_range
        num_hubs = max(2, int(num_memories * hub_ratio))
        num_leaves = num_memories - num_hubs

        # Generate hubs (in smaller batches if needed)
        hub_memories = self._generate_batch_with_retry(
            num_hubs, is_hub=True, tick_range=ticks_range
        )

        # Generate leaves
        leaf_memories = self._generate_leaves_with_retry(
            num_leaves, hub_memories, max_associations, (tick_min, tick_max)
        )

        # Add cross-associations between hubs
        self._add_hub_associations(hub_memories, max_associations=2)

        all_memories = hub_memories + leaf_memories
        all_memories.sort(key=lambda m: m["tick"])
        all_memories = self._resolve_association_ids(all_memories)

        return all_memories

    def _generate_batch_with_retry(
        self, target_count: int, is_hub: bool, tick_range: tuple[int, int], max_retries: int = 3
    ) -> list[dict]:
        """Generate memories, retrying if count is off."""
        for attempt in range(max_retries):
            prompt = self._build_generation_prompt(target_count, is_hub, tick_range)
            items = self._call_llm(prompt)
            if len(items) >= target_count * 0.8:
                return items[:target_count]
            print(f"  Retry {attempt+1}: got {len(items)}, wanted {target_count}")

        return items[:target_count]

    def _generate_leaves_with_retry(
        self, count: int, hub_memories: list[dict],
        max_associations: int, tick_range: tuple[int, int], max_retries: int = 3
    ) -> list[dict]:
        """Generate leaf memories, retrying if count is off."""
        hub_summaries = [
            {"id": h["id"], "content": h["content"], "entities": h["entities"]}
            for h in hub_memories
        ]

        for attempt in range(max_retries):
            prompt = self._build_leaf_generation_prompt(
                count, hub_summaries, max_associations, tick_range
            )
            items = self._call_llm(prompt)
            if len(items) >= count * 0.8:
                return items[:count]
            print(f"  Leaf retry {attempt+1}: got {len(items)}, wanted {count}")

        return items[:count]

    def _add_hub_associations(self, hub_memories: list[dict], max_associations: int) -> None:
        for i, mem in enumerate(hub_memories):
            possible = [h["id"] for j, h in enumerate(hub_memories) if j != i]
            n_assoc = min(max_associations, len(possible))
            chosen = random.sample(possible, n_assoc)
            mem["associations"] = list(set(mem.get("associations", []) + chosen))

    def _resolve_association_ids(self, memories: list[dict]) -> list[dict]:
        entity_index: dict[str, list[str]] = {}
        for m in memories:
            for ent in m.get("entities", []):
                entity_index.setdefault(ent, []).append(m["id"])

        for m in memories:
            resolved = []
            for assoc in m.get("associations", []):
                if isinstance(assoc, dict):
                    resolved.append(assoc)
                elif isinstance(assoc, str) and assoc.startswith("mem_"):
                    resolved.append({"id": assoc, "weight": random.uniform(0.3, 1.0)})
                elif isinstance(assoc, str):
                    matches = entity_index.get(assoc, [])
                    for mid in matches:
                        if mid != m["id"]:
                            resolved.append({"id": mid, "weight": random.uniform(0.3, 1.0)})
            m["associations"] = resolved
        return memories

    def _build_generation_prompt(self, count: int, is_hub: bool, tick_range: tuple[int, int]) -> str:
        tick_min, tick_max = tick_range
        hub_note = ""
        if is_hub:
            hub_note = """이 기억들은 "허브(hub)" 기억입니다. 여러 다른 기억들이 이 기억들을 참조하게 됩니다.
중요하고 자주 언급될 만한 핵심 엔티티(인물, 장소, 관심사)를 중심으로 만들어주세요.
impact는 0.7~1.0 사이의 높은 값을 가집니다."""
        else:
            hub_note = """이 기억들은 보통 기억들입니다. impact는 0.1~0.5 사이의 낮은 값을 가집니다."""

        return f"""반드시 정확히 {count}개의 기억 항목을 생성해주세요. fact와 episode를 대략 반반 섞어주세요.

{hub_note}

각 기억은 다음 JSON 형식:
{{"id": "mem_XXX", "type": "fact"|"episode", "content": "기억 내용", "entities": ["엔티티1", "엔티티2"], "tick": {tick_min}~{tick_max}, "impact": 0.1~1.0, "associations": [], "recall_query": "테스트 질문", "recall_answer": "예상 답변"}}

- ID는 mem_001부터 순차 부여, 반드시 {count}개
- content는 한국어 1-2문장
- tick은 {tick_min}~{tick_max} 범위에서 랜덤
- associations는 빈 리스트

JSON 배열만 출력. 정확히 {count}개여야 합니다."""

    def _build_leaf_generation_prompt(
        self, count: int, hub_summaries: list[dict],
        max_associations: int, tick_range: tuple[int, int]
    ) -> str:
        tick_min, tick_max = tick_range
        hub_list = "\n".join(
            f"- {h['id']}: {h['content']} (엔티티: {', '.join(h['entities'])})"
            for h in hub_summaries
        )

        return f"""반드시 정확히 {count}개의 리프 기억을 생성해주세요.

기존 허브 기억들:
{hub_list}

각 새 기억은 기존 허브 기억 중 1~{max_associations}개와 연관되어야 합니다 (엔티티 중복).

JSON 형식:
{{"id": "mem_XXX", "type": "fact"|"episode", "content": "허브 기억의 엔티티를 포함한 내용", "entities": ["엔티티1"], "tick": {tick_min}~{tick_max}, "impact": 0.1~0.6, "associations": ["참조하는_허브_ID"], "recall_query": "질문", "recall_answer": "답변"}}

- ID는 이어서 번호 매기기, 반드시 {count}개
- associations에는 참조하는 허브 기억의 ID(예: "mem_001")를 넣을 것

JSON 배열만 출력. 정확히 {count}개여야 합니다."""

    def generate_leaves_only(
        self,
        num_leaves: int,
        hub_memories: list[dict],
        max_associations: int = 4,
        ticks_range: tuple[int, int] = (0, 100),
        seed: int | None = None,
    ) -> list[dict]:
        """Generate only leaf memories referencing existing hub memories."""
        if seed is not None:
            random.seed(seed)

        leaf_memories = self._generate_leaves_with_retry(
            num_leaves, hub_memories, max_associations, ticks_range
        )

        # Re-assign leaf IDs to avoid collisions with hub IDs
        hub_ids = {h["id"] for h in hub_memories}
        max_hub_num = 0
        for hid in hub_ids:
            # Extract numeric suffix from IDs like "mem_001"
            parts = hid.replace("mem_", "")
            try:
                max_hub_num = max(max_hub_num, int(parts))
            except ValueError:
                pass
        for j, leaf in enumerate(leaf_memories):
            if leaf["id"] in hub_ids:
                leaf["id"] = f"mem_{max_hub_num + j + 1:03d}"

        all_memories = hub_memories + leaf_memories
        all_memories = self._resolve_association_ids(all_memories)

        # Return only the new leaves (now with resolved associations)
        return [m for m in all_memories if m["id"] not in hub_ids]

    def _call_llm(self, prompt: str) -> list[dict]:
        import openai

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                max_tokens=4096,
                messages=[{"role": "user", "content": prompt}],
            )
        except (openai.BadRequestError, openai.RateLimitError) as e:
            print(f"  API error: {e}. Retrying in 2s...")
            time.sleep(2)
            response = self.client.chat.completions.create(
                model=self.model,
                max_tokens=4096,
                messages=[{"role": "user", "content": prompt}],
            )

        text = response.choices[0].message.content.strip()

        if "```json" in text:
            text = text.split("```json")[1].split("```")[0].strip()
        elif "```" in text:
            text = text.split("```")[1].split("```")[0].strip()

        try:
            items = json.loads(text)
        except json.JSONDecodeError:
            # Attempt to fix common LLM JSON issues
            cleaned = text.replace("\n", "").strip()
            # Remove trailing commas before ] or }
            import re
            cleaned = re.sub(r',\s*([}\]])', r'\1', cleaned)
            try:
                items = json.loads(cleaned)
            except json.JSONDecodeError:
                # Last resort: extract individual JSON objects
                objects = re.findall(r'\{[^{}]*\}', text)
                items = []
                for obj_str in objects:
                    try:
                        items.append(json.loads(obj_str))
                    except json.JSONDecodeError:
                        pass

        if not isinstance(items, list):
            items = [items]

        return items

    def save_jsonl(self, memories: list[dict], path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            for mem in memories:
                f.write(json.dumps(mem, ensure_ascii=False) + "\n")
        print(f"Saved {len(memories)} memories to {path}")

    @staticmethod
    def load_jsonl(path: str | Path) -> list[dict]:
        memories = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    memories.append(json.loads(line))
        return memories

    def split_test_train(
        self, memories: list[dict], test_ratio: float = 0.2, seed: int = 42
    ) -> tuple[list[dict], list[dict]]:
        if seed is not None:
            random.seed(seed)
        facts = [m for m in memories if m["type"] == "fact"]
        episodes = [m for m in memories if m["type"] == "episode"]
        random.shuffle(facts)
        random.shuffle(episodes)
        n_test_facts = max(1, int(len(facts) * test_ratio))
        n_test_episodes = max(1, int(len(episodes) * test_ratio))
        test = facts[:n_test_facts] + episodes[:n_test_episodes]
        train = facts[n_test_facts:] + episodes[n_test_episodes:]
        return train, test
