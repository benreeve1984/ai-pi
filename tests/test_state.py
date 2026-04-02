import json
import tempfile
from pathlib import Path

import pytest

from app.state import AppState


class TestDefaultState:
    def test_default_prompts_exist(self, state):
        assert "drift" in state.prompts
        assert "echo" in state.prompts
        assert len(state.prompts["drift"]) > 0
        assert len(state.prompts["echo"]) > 0

    def test_history_starts_empty(self, state):
        assert state.history == []

    def test_next_speaker_starts_with_drift(self, state):
        assert state.next_speaker == "drift"


class TestPromptUpdate:
    def test_update_prompt(self, state):
        state.update_prompt("drift", "New drift prompt")
        assert state.prompts["drift"] == "New drift prompt"

    def test_update_prompt_persists_to_disk(self, state, tmp_data_dir):
        state.update_prompt("echo", "Saved echo prompt")
        data = json.loads((tmp_data_dir / "prompts.json").read_text())
        assert data["echo"] == "Saved echo prompt"

    def test_update_prompt_rejects_unknown_character(self, state):
        with pytest.raises(ValueError, match="Unknown character"):
            state.update_prompt("nobody", "test")

    def test_update_prompt_enforces_max_length(self, state):
        with pytest.raises(ValueError, match="exceeds maximum"):
            state.update_prompt("drift", "x" * 501)

    def test_update_prompt_allows_max_length(self, state):
        state.update_prompt("drift", "x" * 500)
        assert len(state.prompts["drift"]) == 500


class TestAddTurn:
    def test_add_turn_appends_to_history(self, state):
        state.add_turn("drift", "Hello there")
        assert len(state.history) == 1
        assert state.history[0]["speaker"] == "drift"
        assert state.history[0]["text"] == "Hello there"

    def test_add_turn_increments_id(self, state):
        state.add_turn("drift", "First")
        state.add_turn("echo", "Second")
        assert state.history[0]["id"] == 1
        assert state.history[1]["id"] == 2

    def test_add_turn_records_timestamp(self, state):
        state.add_turn("drift", "Hello")
        assert "timestamp" in state.history[0]

    def test_add_turn_records_prompt_hash(self, state):
        state.add_turn("drift", "Hello")
        assert "prompt_hash" in state.history[0]
        assert len(state.history[0]["prompt_hash"]) == 8

    def test_add_turn_appends_to_jsonl(self, state, tmp_data_dir):
        state.add_turn("drift", "Line one")
        state.add_turn("echo", "Line two")
        lines = (tmp_data_dir / "history.jsonl").read_text().strip().split("\n")
        assert len(lines) == 2
        assert json.loads(lines[0])["text"] == "Line one"
        assert json.loads(lines[1])["text"] == "Line two"

    def test_add_turn_alternates_next_speaker(self, state):
        assert state.next_speaker == "drift"
        state.add_turn("drift", "Hello")
        assert state.next_speaker == "echo"
        state.add_turn("echo", "Hi")
        assert state.next_speaker == "drift"


class TestContextWindow:
    def test_get_context_returns_last_10_messages(self, state):
        for i in range(15):
            speaker = "drift" if i % 2 == 0 else "echo"
            state.add_turn(speaker, f"Message {i}")
        context = state.get_context()
        assert len(context) == 10
        assert context[0]["text"] == "Message 5"
        assert context[-1]["text"] == "Message 14"

    def test_get_context_returns_all_if_fewer_than_10(self, state):
        state.add_turn("drift", "Only one")
        context = state.get_context()
        assert len(context) == 1


class TestRecentTurns:
    def test_get_recent_returns_last_50(self, state):
        for i in range(60):
            speaker = "drift" if i % 2 == 0 else "echo"
            state.add_turn(speaker, f"Message {i}")
        recent = state.get_recent(50)
        assert len(recent) == 50
        assert recent[0]["text"] == "Message 10"


class TestRestoreFromDisk:
    def test_restore_loads_history_and_prompts(self, tmp_data_dir):
        history_file = tmp_data_dir / "history.jsonl"
        prompts_file = tmp_data_dir / "prompts.json"
        history_file.write_text(
            json.dumps({"id": 1, "speaker": "drift", "text": "Restored", "timestamp": "2026-04-02T00:00:00Z", "prompt_hash": "abc12345"}) + "\n"
        )
        prompts_file.write_text(json.dumps({"drift": "Custom drift", "echo": "Custom echo"}))

        state = AppState(data_dir=tmp_data_dir)
        assert len(state.history) == 1
        assert state.history[0]["text"] == "Restored"
        assert state.prompts["drift"] == "Custom drift"
        assert state.next_speaker == "echo"
