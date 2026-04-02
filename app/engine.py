import json
from typing import AsyncIterator

import httpx

from app.state import AppState

MAX_TOKENS = 150
MODEL = "qwen2.5-instruct:1.5b"

BASE_SYSTEM = "IMPORTANT: You MUST respond in English. Do NOT use Chinese or any other language. You are a character in an ongoing two-person dialog. Speak in first person. Do not narrate actions or describe yourself in third person. Never write as the other character."


class DialogEngine:
    def __init__(self, state: AppState, ollama_url: str = "http://localhost:8000"):
        self.state = state
        self.ollama_url = ollama_url
        self._client: httpx.AsyncClient | None = None

    async def get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=120.0)
        return self._client

    def _format_context(self) -> str:
        turns = self.state.get_context()
        lines = []
        for t in turns:
            name = t["speaker"].capitalize()
            lines.append(f"{name}: {t['text']}")
        return "\n".join(lines)

    def build_messages(self, speaker: str) -> list[dict]:
        other = "Echo" if speaker == "drift" else "Drift"
        system_prompt = f"{BASE_SYSTEM}\n\n{self.state.prompts[speaker]}"
        messages = [{"role": "system", "content": system_prompt}]

        # Build a single user message with context + instruction
        parts = []
        context = self._format_context()
        if context:
            parts.append(f"Here is the conversation so far:\n\n{context}\n")
        elif self.state.topic:
            parts.append(f"The topic to discuss is: {self.state.topic}\n")

        parts.append(f"Now respond as {speaker.capitalize()} in 1-3 short sentences in English. Do not write {other}'s lines.")
        messages.append({"role": "user", "content": "\n".join(parts)})

        return messages

    async def generate(self, speaker: str) -> AsyncIterator[str]:
        client = await self.get_client()
        messages = self.build_messages(speaker)

        async with client.stream(
            "POST",
            f"{self.ollama_url}/api/chat",
            json={
                "model": MODEL,
                "messages": messages,
                "stream": True,
                "options": {
                    "num_predict": MAX_TOKENS,
                    "temperature": 1.2,
                    "top_p": 0.95,
                },
            },
        ) as response:
            async for line in response.aiter_lines():
                if not line:
                    continue
                data = json.loads(line)
                token = data.get("message", {}).get("content", "")
                if token:
                    yield token
                if data.get("done", False):
                    break

    async def close(self):
        if self._client:
            await self._client.aclose()
            self._client = None
