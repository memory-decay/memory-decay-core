"""Tests for LongMemEval → memory-decay JSONL converter."""

import json
import pytest

# Minimal LongMemEval fixture matching real schema
FIXTURE = [
    {
        "question_id": "test_q001",
        "question_type": "single-session-user",
        "question": "What degree did I graduate with?",
        "answer": "Business Administration",
        "question_date": "2023/04/10 (Mon) 23:07",
        "haystack_dates": ["2023/04/05 (Wed) 14:00", "2023/04/10 (Mon) 17:50"],
        "haystack_session_ids": ["sess_001", "sess_002"],
        "haystack_sessions": [
            [
                {"role": "user", "content": "I enjoy hiking on weekends.", "has_answer": False},
                {"role": "assistant", "content": "That sounds fun!", "has_answer": False},
            ],
            [
                {"role": "user", "content": "I graduated with a degree in Business Administration.", "has_answer": True},
                {"role": "assistant", "content": "Great field of study!", "has_answer": False},
                {"role": "user", "content": "Thanks, I really enjoyed my time at university.", "has_answer": False},
            ],
        ],
    },
    {
        "question_id": "test_q002",
        "question_type": "temporal-reasoning",
        "question": "What was the first thing I mentioned about my trip?",
        "answer": "Booking flights",
        "question_date": "2023/04/12 (Wed) 10:00",
        "haystack_dates": ["2023/04/08 (Sat) 09:00"],
        "haystack_session_ids": ["sess_003"],
        "haystack_sessions": [
            [
                {"role": "user", "content": "I just booked flights for my vacation.", "has_answer": True},
                {"role": "assistant", "content": "Where are you going?", "has_answer": False},
                {"role": "user", "content": "Planning to visit Japan next month.", "has_answer": False},
            ],
        ],
    },
]


def test_parse_date():
    from scripts.convert_longmemeval import parse_date

    dt = parse_date("2023/04/10 (Mon) 23:07")
    assert dt.year == 2023
    assert dt.month == 4
    assert dt.day == 10
    assert dt.hour == 23
    assert dt.minute == 7


def test_map_type():
    from scripts.convert_longmemeval import map_type

    assert map_type("single-session-user") == "fact"
    assert map_type("single-session-assistant") == "fact"
    assert map_type("single-session-preference") == "fact"
    assert map_type("knowledge-update") == "fact"
    assert map_type("temporal-reasoning") == "episode"
    assert map_type("multi-session") == "episode"


def test_convert_produces_valid_jsonl():
    from scripts.convert_longmemeval import convert

    memories = convert(FIXTURE)

    # Only user messages become memory nodes
    # q001: 3 user msgs (sess_001 has 1 user, sess_002 has 2 user)
    # q002: 2 user msgs (sess_003 has 2 user)
    # Total: 5 user messages
    assert len(memories) == 5

    # Check required fields on every node
    for mem in memories:
        assert "id" in mem
        assert "type" in mem and mem["type"] in ("fact", "episode")
        assert "content" in mem and len(mem["content"]) > 0
        assert "tick" in mem and 0 <= mem["tick"] <= 200
        assert "impact" in mem and 0.0 <= mem["impact"] <= 1.0
        assert "associations" in mem


def test_recall_query_assigned_to_has_answer_message():
    from scripts.convert_longmemeval import convert

    memories = convert(FIXTURE)

    # Find the memory with recall_query for q001
    recall_mems = [m for m in memories if m.get("recall_query")]
    assert len(recall_mems) == 2  # one per question

    q001_recall = [m for m in recall_mems if "Business Administration" in m["content"]]
    assert len(q001_recall) == 1
    assert q001_recall[0]["recall_query"] == "What degree did I graduate with?"
    assert q001_recall[0]["recall_answer"] == "Business Administration"


def test_type_mapping_from_question_type():
    from scripts.convert_longmemeval import convert

    memories = convert(FIXTURE)

    # q001 is single-session-user → fact
    q001_mems = [m for m in memories if m["id"].startswith("lme_test_q001")]
    for m in q001_mems:
        assert m["type"] == "fact"

    # q002 is temporal-reasoning → episode
    q002_mems = [m for m in memories if m["id"].startswith("lme_test_q002")]
    for m in q002_mems:
        assert m["type"] == "episode"


def test_impact_based_on_has_answer():
    from scripts.convert_longmemeval import convert

    memories = convert(FIXTURE)

    for mem in memories:
        if mem.get("recall_query"):
            # has_answer=true messages get high impact
            assert mem["impact"] == 0.8
        else:
            # non-answer messages get low impact
            assert mem["impact"] == 0.3


def test_tick_range():
    from scripts.convert_longmemeval import convert

    memories = convert(FIXTURE)
    ticks = [m["tick"] for m in memories]
    assert min(ticks) == 0
    assert max(ticks) <= 200


def test_assistant_messages_excluded():
    from scripts.convert_longmemeval import convert

    memories = convert(FIXTURE)

    for mem in memories:
        assert "That sounds fun" not in mem["content"]
        assert "Great field of study" not in mem["content"]
        assert "Where are you going" not in mem["content"]


def test_deduplication_across_questions():
    """Same session appearing in two questions should not produce duplicate nodes."""
    from scripts.convert_longmemeval import convert

    # Create two questions sharing the same session
    shared_session = [
        {"role": "user", "content": "I live in Seoul.", "has_answer": True},
        {"role": "assistant", "content": "Nice city!", "has_answer": False},
    ]
    data = [
        {
            "question_id": "dup_q1",
            "question_type": "single-session-user",
            "question": "Where do I live?",
            "answer": "Seoul",
            "question_date": "2023/05/01 (Mon) 10:00",
            "haystack_dates": ["2023/04/20 (Thu) 10:00"],
            "haystack_session_ids": ["shared_sess"],
            "haystack_sessions": [shared_session],
        },
        {
            "question_id": "dup_q2",
            "question_type": "single-session-user",
            "question": "What city am I in?",
            "answer": "Seoul",
            "question_date": "2023/05/02 (Tue) 10:00",
            "haystack_dates": ["2023/04/20 (Thu) 10:00"],
            "haystack_session_ids": ["shared_sess"],
            "haystack_sessions": [shared_session],
        },
    ]

    memories = convert(data)

    # "I live in Seoul." should appear only once
    seoul_mems = [m for m in memories if "Seoul" in m["content"]]
    assert len(seoul_mems) == 1

    # But it should have a recall_query (from first question)
    assert seoul_mems[0]["recall_query"] == "Where do I live?"
