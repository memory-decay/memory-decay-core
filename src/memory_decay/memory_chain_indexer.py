"""Memory chain reader/writer for M27-style persistent learnings."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional


MEMORY_CHAIN_DIR = Path(__file__).resolve().parent.parent.parent / "memory_chain"
ROUND_RE = re.compile(r"^round_(\d+)\.md$")
CHAIN_SUMMARY_PREVIEW_LEN = 500


@dataclass
class RoundData:
    """Bundled data for a single memory-chain round."""

    round_num: int
    experiment: str
    scores: dict[str, float]
    hypothesis: str
    observations: list[str]
    decisions: list[str]
    open_questions: list[str]
    next_direction: str
    parent_round: Optional[int] = None


class MemoryChainIndexer:
    """Read/write the memory chain directory."""

    def __init__(self, chain_dir: Optional[Path] = None):
        self.chain_dir = chain_dir or MEMORY_CHAIN_DIR
        try:
            self.chain_dir.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            raise RuntimeError(f"Failed to create memory chain directory {self.chain_dir}: {e}") from e

    def _ensure_latest_symlink(self) -> None:
        try:
            self.chain_dir.mkdir(parents=True, exist_ok=True)
        except OSError:
            pass  # Already exists
        latest = self.chain_dir / "round_latest.md"
        if not latest.is_symlink() and not latest.exists():
            latest.touch()

    def latest_round_number(self) -> Optional[int]:
        rounds = []
        for f in self.chain_dir.glob("round_*.md"):
            m = ROUND_RE.match(f.name)
            if m:
                rounds.append(int(m.group(1)))
        return max(rounds) if rounds else None

    def write_round(self, data: RoundData) -> Path:
        """Write a round markdown file."""
        self._ensure_latest_symlink()

        prev_ref = f"[round_{data.parent_round:04d}.md](round_{data.parent_round:04d}.md)" if data.parent_round is not None else "none"
        date = datetime.now().strftime("%Y-%m-%d")

        # Build scores table
        scores_lines = []
        for k, v in data.scores.items():
            if isinstance(v, float):
                scores_lines.append(f"| {k} | {v:.4f} |")
            else:
                scores_lines.append(f"| {k} | {v} |")
        scores_table = "\n".join(scores_lines)

        observations_list = "\n".join(f"- {o}" for o in data.observations)
        decisions_list = "\n".join(f"- {d}" for d in data.decisions)
        open_questions_list = "\n".join(f"- {q}" for q in data.open_questions)

        content = f"""# Memory Chain — Round {data.round_num:04d}

## Experiment: {data.experiment}
**Date**: {date}
**Parent**: {prev_ref}

## Scores
| Metric | Value |
|--------|-------|
{scores_table}

## Hypothesis
{data.hypothesis}

## Key Observations
{observations_list}

## Decisions Made
{decisions_list}

## Open Questions
{open_questions_list}

## Next Step Direction
{data.next_direction}
"""
        path = self.chain_dir / f"round_{data.round_num:04d}.md"
        path.write_text(content)

        # Atomically update symlink via temp file in same directory
        latest = self.chain_dir / "round_latest.md"
        tmp = self.chain_dir / f".round_latest.tmp"
        try:
            tmp.write_text(path.name)
            os.replace(tmp, latest)
        finally:
            # Clean up temp file if something went wrong before os.replace
            if tmp.exists():
                tmp.unlink()

        # Append to index
        self._append_index(data.round_num, data.experiment, data.next_direction)
        return path

    def _append_index(self, round_num: int, experiment: str, next_direction: str) -> None:
        index_path = self.chain_dir / "memory_index.jsonl"
        with open(index_path, "a") as f:
            f.write(json.dumps({
                "round": round_num,
                "experiment": experiment,
                "next_direction": next_direction,
                "timestamp": datetime.now().isoformat(),
            }) + "\n")

    def read_chain_summary(self, from_round: Optional[int] = None) -> str:
        """Read recent chain entries for context."""
        lines = []
        for f in sorted(self.chain_dir.glob("round_*.md")):
            m = ROUND_RE.match(f.name)
            if m and (from_round is None or int(m.group(1)) > from_round):
                lines.append(f.read_text()[:CHAIN_SUMMARY_PREVIEW_LEN])
        return "\n---\n".join(lines)