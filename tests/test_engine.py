import json
from unittest.mock import AsyncMock

import pytest

from app.engine import DialogEngine


class TestBuildMessages:
    def test_builds_system_message_from_prompt(self, state):
        engine = DialogEngine(state=state, ollama_url="http://localhost:8000")
        messages = engine.build_messages("drift")
        assert messages[0]["role"] == "system"
        assert "Drift" in messages[0]["content"]
        assert "English" in messages[0]["content"]
        assert "Never write as the other character" in messages[0]["content"]

    def test_includes_history_as_context(self, state):
        state.add_turn("drift", "Hello")
        state.add_turn("echo", "Hi there")
        engine = DialogEngine(state=state, ollama_url="http://localhost:8000")
        messages = engine.build_messages("drift")
        # system + single combined user message
        assert len(messages) == 2
        user_msg = messages[1]["content"]
        assert "Drift: Hello" in user_msg
        assert "Echo: Hi there" in user_msg
        assert "1-3 short sentences" in user_msg

    def test_instruction_in_single_user_message(self, state):
        engine = DialogEngine(state=state, ollama_url="http://localhost:8000")
        messages = engine.build_messages("drift")
        # system + single user message (no context, just instruction)
        assert len(messages) == 2
        user_msg = messages[1]["content"]
        assert "Drift" in user_msg
        assert "English" in user_msg

    def test_context_limited_to_window(self, state):
        for i in range(20):
            speaker = "drift" if i % 2 == 0 else "echo"
            state.add_turn(speaker, f"Message {i}")
        engine = DialogEngine(state=state, ollama_url="http://localhost:8000")
        messages = engine.build_messages("drift")
        user_msg = messages[1]["content"]
        assert "Message 0" not in user_msg
        assert "Message 19" in user_msg


class TestFormatContext:
    def test_formats_turns_with_names(self, state):
        state.add_turn("drift", "Hello world")
        state.add_turn("echo", "Hi back")
        engine = DialogEngine(state=state, ollama_url="http://localhost:8000")
        formatted = engine._format_context()
        assert formatted == "Drift: Hello world\nEcho: Hi back"


class TestGenerate:
    @pytest.mark.asyncio
    async def test_generate_calls_ollama_and_returns_text(self, state):
        from contextlib import asynccontextmanager

        engine = DialogEngine(state=state, ollama_url="http://localhost:8000")

        mock_response_lines = [
            json.dumps({"message": {"content": "Hello "}, "done": False}),
            json.dumps({"message": {"content": "world"}, "done": False}),
            json.dumps({"message": {"content": ""}, "done": True}),
        ]

        async def fake_stream():
            for line in mock_response_lines:
                yield line

        mock_response = AsyncMock()
        mock_response.aiter_lines = fake_stream

        @asynccontextmanager
        async def mock_stream(*args, **kwargs):
            yield mock_response

        mock_client = AsyncMock()
        mock_client.stream = mock_stream
        engine._client = mock_client

        collected_tokens = []
        full_text = ""
        async for token in engine.generate("drift"):
            full_text += token
            collected_tokens.append(token)

        assert full_text == "Hello world"
        assert collected_tokens == ["Hello ", "world"]
