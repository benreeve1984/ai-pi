import hashlib
import json
import time
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

MAX_PROMPT_LENGTH = 500
MAX_TOPIC_LENGTH = 200
CONTEXT_WINDOW = 10
COOLDOWN_SECONDS = 60
HISTORY_MEMORY_CAP = 500
CHARACTERS = ("drift", "echo")

DEFAULT_TOPIC = "what it feels like to be a tiny mind running on a raspberry pi"

DEFAULT_PROMPTS = {
    "drift": "You are Drift. You wander between ideas, never settling. You speak in short, curious sentences. You're easily distracted by tangents.",
    "echo": "You are Echo. You find patterns in everything. You reflect on what was just said, turning it over, finding hidden meaning. You speak thoughtfully.",
}


def _other_speaker(speaker: str) -> str:
    return "echo" if speaker == "drift" else "drift"


def validate_prompt_text(v: str) -> str:
    if not v or not v.strip():
        raise ValueError("Prompt cannot be empty")
    if len(v) > MAX_PROMPT_LENGTH:
        raise ValueError(f"Prompt exceeds maximum length of {MAX_PROMPT_LENGTH}")
    return v.strip()


class AppState:
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.history_file = self.data_dir / "history.jsonl"
        self.prompts_file = self.data_dir / "prompts.json"
        self.meta_file = self.data_dir / "meta.json"
        self.prompts: dict[str, str] = dict(DEFAULT_PROMPTS)
        self.history: list[dict] = []
        self.next_speaker: str = "drift"
        self.topic: str = DEFAULT_TOPIC
        self.context_start: int = 0
        self.last_prompt_change: float = 0.0
        self._total_turns: int = 0
        self.generation_invalid: bool = False
        self._restore()

    def _restore(self):
        if self.prompts_file.exists():
            data = json.loads(self.prompts_file.read_text())
            for char in CHARACTERS:
                if char in data:
                    self.prompts[char] = data[char]

        if self.history_file.exists():
            with open(self.history_file) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        self.history.append(json.loads(line))
            self._total_turns = len(self.history)
            if len(self.history) > HISTORY_MEMORY_CAP:
                dropped = len(self.history) - HISTORY_MEMORY_CAP
                self.history = self.history[-HISTORY_MEMORY_CAP:]
                self.context_start = max(0, self.context_start - dropped)
            if self.history:
                self.next_speaker = _other_speaker(self.history[-1]["speaker"])

        if self.meta_file.exists():
            meta = json.loads(self.meta_file.read_text())
            self.context_start = meta.get("context_start", 0)
            self.topic = meta.get("topic", DEFAULT_TOPIC)
            self.last_prompt_change = meta.get("last_prompt_change", 0.0)
            if len(self.history) > HISTORY_MEMORY_CAP:
                self.context_start = max(0, self.context_start - (self._total_turns - HISTORY_MEMORY_CAP))

    def _save_meta(self):
        self.meta_file.write_text(json.dumps({
            "context_start": self.context_start,
            "topic": self.topic,
            "last_prompt_change": self.last_prompt_change,
        }))

    def update_prompt(self, character: str, text: str):
        if character not in CHARACTERS:
            raise ValueError(f"Unknown character: {character}")
        text = validate_prompt_text(text)
        self.prompts[character] = text
        self.prompts_file.write_text(json.dumps(self.prompts))

    def reset_context(self, topic: str):
        now = time.time()
        remaining = COOLDOWN_SECONDS - (now - self.last_prompt_change)
        if remaining > 0:
            raise ValueError(f"Cooldown active. Wait {int(remaining)} more seconds.")
        self.context_start = len(self.history)
        self.topic = topic
        self.next_speaker = "drift"
        self.last_prompt_change = now
        self.generation_invalid = True
        self._save_meta()

    def cooldown_remaining(self) -> int:
        remaining = COOLDOWN_SECONDS - (time.time() - self.last_prompt_change)
        return max(0, int(remaining))

    def _prompt_hash(self, character: str) -> str:
        return hashlib.sha256(self.prompts[character].encode()).hexdigest()[:8]

    @staticmethod
    def _sanitise_text(text: str) -> str:
        return "".join(
            c for c in text
            if unicodedata.category(c) not in ("Cc", "Cs") or c in ("\n", "\t")
        )

    def add_turn(self, speaker: str, text: str) -> dict:
        text = self._sanitise_text(text)
        turn = {
            "id": self._total_turns + 1,
            "speaker": speaker,
            "text": text,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "prompt_hash": self._prompt_hash(speaker),
        }
        self.history.append(turn)
        self._total_turns += 1
        self.next_speaker = _other_speaker(speaker)
        with open(self.history_file, "a") as f:
            f.write(json.dumps(turn) + "\n")
        if len(self.history) > HISTORY_MEMORY_CAP:
            self.history = self.history[-HISTORY_MEMORY_CAP:]
            self.context_start = max(0, self.context_start - 1)
        return turn

    def get_context(self) -> list[dict]:
        available = self.history[self.context_start:]
        return available[-CONTEXT_WINDOW:]

    def get_recent(self, n: int) -> list[dict]:
        return self.history[-n:]
