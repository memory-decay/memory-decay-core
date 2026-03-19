"""Shared utilities for memory_decay modules."""

from __future__ import annotations

import json
import re
from typing import Any


def extract_json_from_markdown(text: str) -> dict[str, Any]:
    """Extract JSON object from LLM markdown-formatted response.

    Handles responses wrapped in ```json fences or bare ``` fences.
    Falls back to regex search for first JSON object on parse failure.
    """
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0].strip()
    elif "```" in text:
        text = text.split("```")[1].split("```")[0].strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{[\s\S]*\}", text)
        if match:
            return json.loads(match.group())
        raise
