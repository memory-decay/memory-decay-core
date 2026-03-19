"""Convert LongMemEval oracle JSON to memory-decay JSONL format.

Input:  data/longmemeval_oracle.json
Output: data/longmemeval.jsonl

Each user message becomes a memory node. Assistant messages are excluded.
Deduplicates across questions by (session_id, message_index).
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path


def parse_date(date_str: str) -> datetime:
    """Parse LongMemEval date format: '2023/04/10 (Mon) 23:07'."""
    # Strip day-of-week in parentheses
    cleaned = re.sub(r"\s*\([A-Za-z]+\)\s*", " ", date_str).strip()
    return datetime.strptime(cleaned, "%Y/%m/%d %H:%M")


TYPE_MAP = {
    "single-session-user": "fact",
    "single-session-assistant": "fact",
    "single-session-preference": "fact",
    "knowledge-update": "fact",
    "temporal-reasoning": "episode",
    "multi-session": "episode",
}


def map_type(question_type: str) -> str:
    """Map LongMemEval question type to memory type."""
    return TYPE_MAP.get(question_type, "fact")


def convert(data: list[dict]) -> list[dict]:
    """Convert LongMemEval questions to memory-decay JSONL records.

    Returns a list of memory node dicts matching the memories_500.jsonl schema.
    """
    # Pass 1: collect all dates for tick normalization
    all_dates: list[datetime] = []
    for q in data:
        for date_str in q.get("haystack_dates", []):
            all_dates.append(parse_date(date_str))

    if not all_dates:
        return []

    min_date = min(all_dates)
    max_date = max(all_dates)
    date_range = (max_date - min_date).total_seconds()
    if date_range == 0:
        date_range = 1.0  # avoid division by zero

    def date_to_tick(dt: datetime) -> int:
        elapsed = (dt - min_date).total_seconds()
        return int(elapsed / date_range * 200)

    # Pass 2: deduplicate and build memory nodes
    # Key: (session_id, user_msg_index_within_session) → memory dict
    seen: dict[tuple[str, int], dict] = {}
    # Track which memory IDs are associated per question
    question_associations: dict[str, list[str]] = {}

    for q in data:
        qid = q["question_id"]
        qtype = map_type(q["question_type"])
        session_ids = q.get("haystack_session_ids", [])
        sessions = q.get("haystack_sessions", [])
        dates = q.get("haystack_dates", [])

        answer_mem_ids: list[str] = []

        for s_idx, session in enumerate(sessions):
            sess_id = session_ids[s_idx] if s_idx < len(session_ids) else f"sess_{s_idx}"
            sess_date = parse_date(dates[s_idx]) if s_idx < len(dates) else min_date
            tick = date_to_tick(sess_date)

            user_msg_idx = 0
            for msg in session:
                if msg["role"] != "user":
                    continue

                dedup_key = (sess_id, user_msg_idx)
                mem_id = f"lme_{qid}_s{s_idx}_m{user_msg_idx}"
                has_answer = msg.get("has_answer", False)

                if dedup_key not in seen:
                    mem = {
                        "id": mem_id,
                        "type": qtype,
                        "content": msg["content"],
                        "entities": [],
                        "tick": tick,
                        "impact": 0.8 if has_answer else 0.3,
                        "associations": [],
                    }

                    if has_answer:
                        mem["recall_query"] = q["question"]
                        mem["recall_answer"] = q["answer"]
                        answer_mem_ids.append(mem_id)

                    seen[dedup_key] = mem
                else:
                    # Already seen this message from another question
                    existing = seen[dedup_key]
                    if has_answer and "recall_query" not in existing:
                        existing["recall_query"] = q["question"]
                        existing["recall_answer"] = q["answer"]
                        existing["impact"] = 0.8
                    if has_answer:
                        answer_mem_ids.append(existing["id"])

                user_msg_idx += 1

        question_associations[qid] = answer_mem_ids

    # Pass 3: build associations between answer-bearing messages of the same question
    memories = list(seen.values())
    mem_by_id = {m["id"]: m for m in memories}

    for qid, answer_ids in question_associations.items():
        if len(answer_ids) < 2:
            continue
        for i, aid in enumerate(answer_ids):
            if aid not in mem_by_id:
                continue
            for j, other_id in enumerate(answer_ids):
                if i == j or other_id not in mem_by_id:
                    continue
                existing_assoc_ids = {a["id"] for a in mem_by_id[aid]["associations"]}
                if other_id not in existing_assoc_ids:
                    mem_by_id[aid]["associations"].append(
                        {"id": other_id, "weight": 0.7}
                    )

    # Sort by tick for deterministic output
    memories.sort(key=lambda m: (m["tick"], m["id"]))

    return memories


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Convert LongMemEval oracle JSON to memory-decay JSONL"
    )
    parser.add_argument(
        "--input",
        default="data/longmemeval_oracle.json",
        help="Path to LongMemEval oracle JSON",
    )
    parser.add_argument(
        "--output",
        default="data/longmemeval.jsonl",
        help="Output JSONL path",
    )
    args = parser.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        data = json.load(f)

    memories = convert(data)

    with open(args.output, "w", encoding="utf-8") as f:
        for mem in memories:
            f.write(json.dumps(mem, ensure_ascii=False) + "\n")

    # Stats
    with_query = sum(1 for m in memories if "recall_query" in m)
    facts = sum(1 for m in memories if m["type"] == "fact")
    episodes = sum(1 for m in memories if m["type"] == "episode")
    print(f"Converted {len(memories)} memories ({facts} facts, {episodes} episodes)")
    print(f"Recall queries: {with_query}")
    print(f"Tick range: {min(m['tick'] for m in memories)}-{max(m['tick'] for m in memories)}")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
