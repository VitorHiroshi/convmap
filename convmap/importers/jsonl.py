"""JSONL importer — one conversation per line.

Expected format per line:
{
    "id": "conv-123",
    "turns": [
        {"speaker": "agent", "text": "Hello, how can I help?"},
        {"speaker": "customer", "text": "I have a billing issue"}
    ],
    "metadata": {"client": "acme", "duration": 120}  // optional
}

Also supports a flat transcript format:
{
    "id": "conv-123",
    "transcript": "agent: Hello\ncustomer: I have a billing issue",
    "metadata": {}
}
"""

from __future__ import annotations

import json
from pathlib import Path

from ..types import Conversation, Turn


def load(source: str | Path) -> list[Conversation]:
    """Load conversations from a JSONL file."""
    path = Path(source)
    conversations = []

    with path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                conv = _parse_record(data, line_num)
                if conv:
                    conversations.append(conv)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON on line {line_num}: {e}") from e

    return conversations


def load_records(records: list[dict]) -> list[Conversation]:
    """Load conversations from a list of dictionaries."""
    conversations = []
    for i, record in enumerate(records):
        conv = _parse_record(record, i + 1)
        if conv:
            conversations.append(conv)
    return conversations


def _parse_record(data: dict, index: int) -> Conversation | None:
    """Parse a single record into a Conversation."""
    conv_id = data.get("id", f"conv-{index}")
    metadata = data.get("metadata", {})

    # Format 1: structured turns
    if "turns" in data:
        turns = [
            Turn(speaker=t.get("speaker", "unknown"), text=t.get("text", ""))
            for t in data["turns"]
            if t.get("text", "").strip()
        ]
        if not turns:
            return None
        return Conversation(id=conv_id, turns=turns, metadata=metadata)

    # Format 2: flat transcript with speaker labels
    if "transcript" in data:
        turns = _parse_transcript(data["transcript"])
        if not turns:
            return None
        return Conversation(id=conv_id, turns=turns, metadata=metadata)

    # Format 3: messages array (common in chat APIs)
    if "messages" in data:
        turns = [
            Turn(
                speaker=m.get("role", m.get("speaker", "unknown")),
                text=m.get("content", m.get("text", "")),
            )
            for m in data["messages"]
            if m.get("content", m.get("text", "")).strip()
        ]
        if not turns:
            return None
        return Conversation(id=conv_id, turns=turns, metadata=metadata)

    return None


def _parse_transcript(text: str) -> list[Turn]:
    """Parse a flat transcript string into turns.

    Supports formats like:
        agent: Hello
        customer: I need help

    Or:
        [agent] Hello
        [customer] I need help
    """
    turns = []
    current_speaker = "unknown"
    current_text_parts: list[str] = []

    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue

        # Try "speaker: text" format
        if ": " in line:
            parts = line.split(": ", 1)
            speaker_candidate = parts[0].strip().lower()
            # Heuristic: speaker labels are short
            if len(speaker_candidate) < 30 and not speaker_candidate[0].isdigit():
                if current_text_parts:
                    turns.append(Turn(speaker=current_speaker, text=" ".join(current_text_parts)))
                current_speaker = speaker_candidate
                current_text_parts = [parts[1]]
                continue

        # Try "[speaker] text" format
        if line.startswith("[") and "]" in line:
            bracket_end = line.index("]")
            speaker_candidate = line[1:bracket_end].strip().lower()
            if len(speaker_candidate) < 30:
                if current_text_parts:
                    turns.append(Turn(speaker=current_speaker, text=" ".join(current_text_parts)))
                current_speaker = speaker_candidate
                current_text_parts = [line[bracket_end + 1:].strip()]
                continue

        # Continuation of current turn
        current_text_parts.append(line)

    if current_text_parts:
        turns.append(Turn(speaker=current_speaker, text=" ".join(current_text_parts)))

    return turns
