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

from openai import OpenAI


class SyntheticDataGenerator:
    """Generate synthetic memory datasets for simulation experiments.

    Uses OpenAI API to create realistic memory items with:
    - Semantic associations between memories
    - Hub-and-leaf topology (some memories referenced frequently)
    - Impact scores (emotional significance)
    - Held-out recall test queries
    """

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
        """Generate a complete synthetic memory dataset.

        Args:
            num_memories: Total number of memory items to generate.
            hub_ratio: Fraction of memories that are "hubs" (highly connected).
            max_associations: Max number of associations per memory.
            ticks_range: (min, max) tick range for memory creation times.
            seed: Random seed for reproducibility.

        Returns:
            List of memory item dicts following the JSONL schema.
        """
        if seed is not None:
            random.seed(seed)

        tick_min, tick_max = ticks_range
        num_hubs = max(1, int(num_memories * hub_ratio))
        num_leaves = num_memories - num_hubs

        # Step 1: Generate hub memories
        hub_memories = self._generate_memories_batch(
            num_hubs, is_hub=True, tick_range=ticks_range
        )

        # Step 2: Generate leaf memories that reference hubs
        leaf_memories = self._generate_leaves_batch(
            num_leaves,
            hub_memories,
            max_associations=max_associations,
            tick_range=ticks_range,
        )

        # Step 3: Add cross-associations between hubs
        self._add_hub_associations(hub_memories, max_associations=2)

        all_memories = hub_memories + leaf_memories

        # Step 4: Sort by tick
        all_memories.sort(key=lambda m: m["tick"])

        # Step 5: Fix association references
        all_memories = self._resolve_association_ids(all_memories)

        return all_memories

    def _generate_memories_batch(
        self, count: int, is_hub: bool, tick_range: tuple[int, int]
    ) -> list[dict]:
        if count == 0:
            return []
        prompt = self._build_generation_prompt(count, is_hub, tick_range)
        return self._call_llm(prompt)

    def _generate_leaves_batch(
        self,
        count: int,
        hub_memories: list[dict],
        max_associations: int,
        tick_range: tuple[int, int],
    ) -> list[dict]:
        if count == 0:
            return []
        hub_summaries = [
            {"id": h["id"], "content": h["content"], "entities": h["entities"]}
            for h in hub_memories
        ]
        prompt = self._build_leaf_generation_prompt(
            count, hub_summaries, max_associations, tick_range
        )
        return self._call_llm(prompt)

    def _add_hub_associations(
        self, hub_memories: list[dict], max_associations: int
    ) -> None:
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
                            resolved.append(
                                {"id": mid, "weight": random.uniform(0.3, 1.0)}
                            )
            m["associations"] = resolved

        return memories

    def _build_generation_prompt(
        self, count: int, is_hub: bool, tick_range: tuple[int, int]
    ) -> str:
        tick_min, tick_max = tick_range
        hub_note = ""
        if is_hub:
            hub_note = """
이 기억들은 "허브(hub)" 기억입니다. 여러 다른 기억들이 이 기억들을 참조하게 됩니다.
중요하고 자주 언급될 만한 핵심 엔티티(인물, 장소, 관심사)를 중심으로 만들어주세요.
impact는 0.7~1.0 사이의 높은 값을 가집니다."""
        else:
            hub_note = """
이 기억들은 보통 기억들입니다. impact는 0.1~0.5 사이의 낮은 값을 가집니다."""

        return f"""한국어로 {count}개의 기억 항목을 생성해주세요. 각 항목은 "fact"(사실) 또는 "episode"(에피소드) 타입 중 하나입니다.

{hub_note}

각 기억은 다음 JSON 형식을 따릅니다:
```json
{{
  "id": "mem_XXX",
  "type": "fact" 또는 "episode",
  "content": "기억 내용 (한국어, 1-2문장)",
  "entities": ["엔티티1", "엔티티2"],
  "tick": {tick_min}~{tick_max} 사이의 정수 (생성 시점),
  "impact": 0.1~1.0 사이의 float (감정적 중요도),
  "associations": [],
  "recall_query": "이 기억을 테스트하기 위한 질문 (한국어)",
  "recall_answer": "예상 답변"
}}
```

규칙:
- ID는 mem_001, mem_002, ... 형식으로 순차 부여
- fact와 episode를 대략 반반 섞어주세요
- fact: "서울은 대한민국의 수도이다" 같은 객관적 정보
- episode: "작년 여름 서울에 여행을 갔다" 같은 개인적 경험
- entities는 content에 등장하는 핵심 명사/고유명사
- tick은 랜덤하게 분포시키되, 중복 허용
- associations는 빈 리스트로 두세요 (나중에 자동으로 채워집니다)
- recall_query는 자연스러운 질문 형태

JSON 배열만 출력해주세요. 다른 설명은 불필요합니다."""

    def _build_leaf_generation_prompt(
        self,
        count: int,
        hub_summaries: list[dict],
        max_associations: int,
        tick_range: tuple[int, int],
    ) -> str:
        tick_min, tick_max = tick_range

        hub_list = "\n".join(
            f"- {h['id']}: {h['content']} (엔티티: {', '.join(h['entities'])})"
            for h in hub_summaries
        )

        return f"""다음 허브 기억들을 참조하는 {count}개의 리프(leaf) 기억을 생성해주세요.

기존 허브 기억들:
{hub_list}

각 새 기억은 기존 허브 기억 중 1~{max_associations}개와 연관되어야 합니다.
연관은 엔티티 중복(같은 인물/장소/주제 언급)으로 만들어주세요.

JSON 형식 (배열):
```json
{{
  "id": "mem_XXX",
  "type": "fact" 또는 "episode",
  "content": "기억 내용 (허브 기억의 엔티티를 포함해야 함)",
  "entities": ["엔티티1", "엔티티2"],
  "tick": {tick_min}~{tick_max} 사이의 정수,
  "impact": 0.1~0.6 사이의 float (리프 기억은 상대적으로 낮은 impact),
  "associations": ["참조하는 허브 기억의 엔티티명 또는 ID"],
  "recall_query": "테스트 질문",
  "recall_answer": "예상 답변"
}}
```

규칙:
- ID는 이어서 번호를 매겨주세요
- content에 허브 기억과 겹치는 엔티티를 반드시 포함하세요
- associations에는 참조하는 허브 기억의 ID(예: "mem_001")를 넣어주세요
- impact는 0.1~0.6 범위
- fact와 episode를 대략 반반 섞어주세요

JSON 배열만 출력해주세요."""

    def _call_llm(self, prompt: str) -> list[dict]:
        """Call OpenAI API and parse response as JSON array."""
        response = self.client.chat.completions.create(
            model=self.model,
            max_tokens=4096,
            messages=[{"role": "user", "content": prompt}],
        )

        text = response.choices[0].message.content.strip()

        # Extract JSON from response
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0].strip()
        elif "```" in text:
            text = text.split("```")[1].split("```")[0].strip()

        try:
            items = json.loads(text)
        except json.JSONDecodeError:
            text = text.replace("\n", "").strip()
            items = json.loads(text)

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
