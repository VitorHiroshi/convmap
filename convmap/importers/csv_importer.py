"""CSV importer — conversations from tabular data.

Supports two layouts:

1. One-row-per-conversation:
   id, transcript, client, duration
   "conv-1", "agent: Hello\ncustomer: Help", "acme", 120

2. One-row-per-turn:
   conversation_id, speaker, text, timestamp
   "conv-1", "agent", "Hello", "2024-01-01T00:00:00"
   "conv-1", "customer", "Help", "2024-01-01T00:00:05"

The importer auto-detects the layout based on column names.
"""

from __future__ import annotations

import csv
from pathlib import Path

from ..types import Conversation, Turn
from .jsonl import _parse_transcript


def load(
    source: str | Path,
    id_column: str | None = None,
    transcript_column: str | None = None,
    speaker_column: str | None = None,
    text_column: str | None = None,
    metadata_columns: list[str] | None = None,
    delimiter: str = ",",
) -> list[Conversation]:
    """Load conversations from a CSV file.

    Auto-detects layout if column names aren't specified.
    """
    path = Path(source)

    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        rows = list(reader)

    if not rows:
        return []

    headers = set(rows[0].keys())
    layout = _detect_layout(headers, transcript_column, speaker_column, text_column)

    if layout == "per_conversation":
        return _load_per_conversation(
            rows,
            id_col=id_column or _find_column(headers, ["id", "conversation_id", "call_id"]),
            transcript_col=transcript_column or _find_column(headers, ["transcript", "text", "content", "body"]),
            metadata_columns=metadata_columns,
        )
    else:
        return _load_per_turn(
            rows,
            id_col=id_column or _find_column(headers, ["conversation_id", "call_id", "id", "session_id"]),
            speaker_col=speaker_column or _find_column(headers, ["speaker", "role", "actor", "participant"]),
            text_col=text_column or _find_column(headers, ["text", "content", "message", "utterance"]),
            metadata_columns=metadata_columns,
        )


def _detect_layout(
    headers: set[str],
    transcript_col: str | None,
    speaker_col: str | None,
    text_col: str | None,
) -> str:
    """Auto-detect CSV layout from column names."""
    if transcript_col and transcript_col in headers:
        return "per_conversation"
    if speaker_col and text_col:
        return "per_turn"

    speaker_candidates = {"speaker", "role", "actor", "participant"}
    has_speaker = bool(headers & speaker_candidates)

    # If there's a speaker/role column, it's per-turn
    if has_speaker:
        return "per_turn"

    # Only unambiguous transcript column names trigger per-conversation
    transcript_only = {"transcript", "body"}
    if headers & transcript_only:
        return "per_conversation"

    return "per_conversation"


def _find_column(headers: set[str], candidates: list[str]) -> str:
    """Find the first matching column name."""
    # Exact match
    for c in candidates:
        if c in headers:
            return c
    # Case-insensitive
    lower_map = {h.lower(): h for h in headers}
    for c in candidates:
        if c.lower() in lower_map:
            return lower_map[c.lower()]
    return candidates[0]


def _load_per_conversation(
    rows: list[dict],
    id_col: str,
    transcript_col: str,
    metadata_columns: list[str] | None,
) -> list[Conversation]:
    """Load from one-row-per-conversation layout."""
    conversations = []
    excluded = {id_col, transcript_col}

    for i, row in enumerate(rows):
        conv_id = row.get(id_col, f"conv-{i + 1}")
        transcript = row.get(transcript_col, "")
        if not transcript.strip():
            continue

        turns = _parse_transcript(transcript)
        if not turns:
            continue

        metadata = {}
        meta_cols = metadata_columns or [k for k in row if k not in excluded]
        for col in meta_cols:
            if col in row:
                metadata[col] = row[col]

        conversations.append(Conversation(id=conv_id, turns=turns, metadata=metadata))

    return conversations


def _load_per_turn(
    rows: list[dict],
    id_col: str,
    speaker_col: str,
    text_col: str,
    metadata_columns: list[str] | None,
) -> list[Conversation]:
    """Load from one-row-per-turn layout, grouping by conversation ID."""
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        conv_id = row.get(id_col, "unknown")
        grouped.setdefault(conv_id, []).append(row)

    conversations = []
    excluded = {id_col, speaker_col, text_col}

    for conv_id, turn_rows in grouped.items():
        turns = []
        for row in turn_rows:
            text = row.get(text_col, "").strip()
            if not text:
                continue
            turns.append(Turn(
                speaker=row.get(speaker_col, "unknown"),
                text=text,
            ))

        if not turns:
            continue

        # Metadata from first row
        metadata = {}
        first_row = turn_rows[0]
        meta_cols = metadata_columns or [k for k in first_row if k not in excluded]
        for col in meta_cols:
            if col in first_row:
                metadata[col] = first_row[col]

        conversations.append(Conversation(id=conv_id, turns=turns, metadata=metadata))

    return conversations
