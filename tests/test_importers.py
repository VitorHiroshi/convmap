"""Tests for transcript importers."""

import json
import tempfile
from pathlib import Path

import pytest

from convmap.importers import jsonl, csv_importer


class TestJSONLImporter:
    def test_load_structured_turns(self, tmp_path):
        data = {
            "id": "conv-1",
            "turns": [
                {"speaker": "agent", "text": "Hello, how can I help?"},
                {"speaker": "customer", "text": "I have a billing issue"},
            ],
            "metadata": {"client": "acme"},
        }
        path = tmp_path / "test.jsonl"
        path.write_text(json.dumps(data) + "\n")

        convs = jsonl.load(path)
        assert len(convs) == 1
        assert convs[0].id == "conv-1"
        assert len(convs[0].turns) == 2
        assert convs[0].turns[0].speaker == "agent"
        assert convs[0].metadata["client"] == "acme"

    def test_load_flat_transcript(self, tmp_path):
        data = {
            "id": "conv-2",
            "transcript": "agent: Hello\ncustomer: I need help with my account",
        }
        path = tmp_path / "test.jsonl"
        path.write_text(json.dumps(data) + "\n")

        convs = jsonl.load(path)
        assert len(convs) == 1
        assert len(convs[0].turns) == 2
        assert convs[0].turns[0].speaker == "agent"
        assert convs[0].turns[1].speaker == "customer"

    def test_load_messages_format(self, tmp_path):
        data = {
            "id": "conv-3",
            "messages": [
                {"role": "assistant", "content": "Welcome!"},
                {"role": "user", "content": "Thanks"},
            ],
        }
        path = tmp_path / "test.jsonl"
        path.write_text(json.dumps(data) + "\n")

        convs = jsonl.load(path)
        assert len(convs) == 1
        assert convs[0].turns[0].speaker == "assistant"
        assert convs[0].turns[1].text == "Thanks"

    def test_load_multiple_lines(self, tmp_path):
        lines = [
            json.dumps({"id": f"conv-{i}", "turns": [{"speaker": "a", "text": f"msg-{i}"}]})
            for i in range(5)
        ]
        path = tmp_path / "test.jsonl"
        path.write_text("\n".join(lines) + "\n")

        convs = jsonl.load(path)
        assert len(convs) == 5

    def test_skip_empty_lines(self, tmp_path):
        content = json.dumps({"id": "c1", "turns": [{"speaker": "a", "text": "hi"}]})
        path = tmp_path / "test.jsonl"
        path.write_text(f"\n{content}\n\n{content}\n\n")

        convs = jsonl.load(path)
        assert len(convs) == 2

    def test_skip_records_without_turns(self, tmp_path):
        data = {"id": "empty", "something": "irrelevant"}
        path = tmp_path / "test.jsonl"
        path.write_text(json.dumps(data) + "\n")

        convs = jsonl.load(path)
        assert len(convs) == 0

    def test_load_records_from_dicts(self):
        records = [
            {"id": "c1", "turns": [{"speaker": "a", "text": "hello"}]},
            {"id": "c2", "transcript": "agent: hi\ncustomer: bye"},
        ]
        convs = jsonl.load_records(records)
        assert len(convs) == 2

    def test_invalid_json_raises(self, tmp_path):
        path = tmp_path / "bad.jsonl"
        path.write_text("not json\n")

        with pytest.raises(ValueError, match="Invalid JSON"):
            jsonl.load(path)

    def test_parse_bracket_transcript(self):
        turns = jsonl._parse_transcript("[agent] Hello there\n[customer] I need help")
        assert len(turns) == 2
        assert turns[0].speaker == "agent"
        assert turns[1].text == "I need help"

    def test_auto_id_when_missing(self, tmp_path):
        data = {"turns": [{"speaker": "a", "text": "hi"}]}
        path = tmp_path / "test.jsonl"
        path.write_text(json.dumps(data) + "\n")

        convs = jsonl.load(path)
        assert convs[0].id == "conv-1"


class TestCSVImporter:
    def test_load_per_conversation(self, tmp_path):
        path = tmp_path / "test.csv"
        path.write_text(
            "id,transcript,client\n"
            '"conv-1","agent: Hello\ncustomer: Help me","acme"\n'
            '"conv-2","agent: Hi\ncustomer: Billing issue","corp"\n'
        )

        convs = csv_importer.load(path)
        assert len(convs) == 2
        assert convs[0].id == "conv-1"
        assert convs[0].turns[0].speaker == "agent"
        assert convs[0].metadata["client"] == "acme"

    def test_load_per_turn(self, tmp_path):
        path = tmp_path / "test.csv"
        path.write_text(
            "conversation_id,speaker,text\n"
            "conv-1,agent,Hello\n"
            "conv-1,customer,I need help\n"
            "conv-2,agent,Welcome\n"
            "conv-2,customer,Thanks\n"
        )

        convs = csv_importer.load(path)
        assert len(convs) == 2

        conv_ids = {c.id for c in convs}
        assert conv_ids == {"conv-1", "conv-2"}

        c1 = next(c for c in convs if c.id == "conv-1")
        assert len(c1.turns) == 2
        assert c1.turns[0].speaker == "agent"

    def test_auto_detect_per_conversation(self, tmp_path):
        path = tmp_path / "test.csv"
        path.write_text(
            "id,transcript,duration\n"
            '"c1","agent: Hi\ncustomer: Hello",120\n'
        )

        convs = csv_importer.load(path)
        assert len(convs) == 1
        assert convs[0].metadata["duration"] == "120"

    def test_auto_detect_per_turn(self, tmp_path):
        path = tmp_path / "test.csv"
        path.write_text(
            "id,role,content\n"
            "c1,agent,Hello\n"
            "c1,customer,Help\n"
        )

        convs = csv_importer.load(path)
        assert len(convs) == 1
        assert len(convs[0].turns) == 2

    def test_empty_csv_returns_empty(self, tmp_path):
        path = tmp_path / "test.csv"
        path.write_text("id,transcript\n")

        convs = csv_importer.load(path)
        assert len(convs) == 0

    def test_explicit_column_names(self, tmp_path):
        path = tmp_path / "test.csv"
        path.write_text(
            "call_ref,body,account\n"
            '"x1","agent: Hi\ncustomer: Bye","abc"\n'
        )

        convs = csv_importer.load(
            path,
            id_column="call_ref",
            transcript_column="body",
            metadata_columns=["account"],
        )
        assert len(convs) == 1
        assert convs[0].id == "x1"
        assert convs[0].metadata["account"] == "abc"

    def test_tsv_support(self, tmp_path):
        path = tmp_path / "test.tsv"
        path.write_text(
            "id\ttranscript\n"
            "c1\tagent: Hello\\ncustomer: Hi\n"
        )

        convs = csv_importer.load(path, delimiter="\t")
        assert len(convs) == 1
