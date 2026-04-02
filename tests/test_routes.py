import tempfile
import os

import pytest

_tmp = tempfile.mkdtemp()
os.environ["TINYCHAT_DATA_DIR"] = _tmp

from fastapi.testclient import TestClient
from app.main import app, state
from app.state import DEFAULT_PROMPTS, DEFAULT_TOPIC


@pytest.fixture(autouse=True)
def reset_state():
    state.history.clear()
    state.prompts.update(DEFAULT_PROMPTS)
    state.next_speaker = "drift"
    state.topic = DEFAULT_TOPIC
    state.context_start = 0
    state.last_prompt_change = 0.0
    state._total_turns = 0


client = TestClient(app)


class TestMainPage:
    def test_get_index_returns_html(self):
        response = client.get("/")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
        assert "tinychat.sh" in response.text

    def test_index_contains_prompt_sections(self):
        response = client.get("/")
        assert "DRIFT" in response.text
        assert "ECHO" in response.text


class TestHistoryPage:
    def test_get_history_returns_html(self):
        response = client.get("/history")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]


class TestStateEndpoint:
    def test_get_state_returns_json(self):
        response = client.get("/state")
        assert response.status_code == 200
        data = response.json()
        assert "prompts" in data
        assert "history" in data
        assert data["prompts"]["drift"] != ""

    def test_state_includes_recent_history(self):
        state.add_turn("drift", "Test message")
        response = client.get("/state")
        data = response.json()
        assert len(data["history"]) == 1
        assert data["history"][0]["text"] == "Test message"


class TestUpdate:
    def test_update_succeeds(self):
        response = client.post("/update", json={
            "drift": "New drift", "echo": "New echo", "topic": "new topic"
        })
        assert response.status_code == 200
        assert response.json()["status"] == "ok"

    def test_update_persists_both_prompts(self):
        client.post("/update", json={
            "drift": "Updated drift", "echo": "Updated echo", "topic": "test"
        })
        response = client.get("/state")
        data = response.json()
        assert data["prompts"]["drift"] == "Updated drift"
        assert data["prompts"]["echo"] == "Updated echo"
        assert data["topic"] == "test"

    def test_update_resets_context(self):
        state.add_turn("drift", "Old turn")
        state.add_turn("echo", "Old turn 2")
        client.post("/update", json={
            "drift": "New drift", "echo": "New echo", "topic": "fresh"
        })
        assert state.context_start == 2
        assert state.topic == "fresh"
        assert state.next_speaker == "drift"

    def test_update_rejects_empty_topic(self):
        response = client.post("/update", json={
            "drift": "test", "echo": "test", "topic": ""
        })
        assert response.status_code == 422

    def test_update_rejects_empty_prompt(self):
        response = client.post("/update", json={
            "drift": "", "echo": "test", "topic": "test"
        })
        assert response.status_code == 422

    def test_update_rejects_too_long_prompt(self):
        response = client.post("/update", json={
            "drift": "x" * 501, "echo": "test", "topic": "test"
        })
        assert response.status_code == 422

    def test_cooldown_prevents_rapid_changes(self):
        client.post("/update", json={
            "drift": "First", "echo": "First", "topic": "topic 1"
        })
        response = client.post("/update", json={
            "drift": "Second", "echo": "Second", "topic": "topic 2"
        })
        assert response.status_code == 429
