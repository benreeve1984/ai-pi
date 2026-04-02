import asyncio
import json
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, field_validator
from sse_starlette.sse import EventSourceResponse
from starlette.middleware.base import BaseHTTPMiddleware

from app.engine import DialogEngine
from app.state import (
    AppState, COOLDOWN_SECONDS, MAX_PROMPT_LENGTH, MAX_TOPIC_LENGTH,
    validate_prompt_text,
)

logger = logging.getLogger(__name__)

DATA_DIR = Path(os.environ.get("TINYCHAT_DATA_DIR", os.path.expanduser("~/tinychat")))
PAUSE_SECONDS = 8
MAX_SUBSCRIBERS = 50

state = AppState(data_dir=DATA_DIR)
engine = DialogEngine(state=state)
_subscribers: set[asyncio.Queue] = set()


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        response = await call_next(request)
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; "
            "script-src 'self' 'unsafe-inline'; "
            "style-src 'self'; "
            "connect-src 'self'; "
            "img-src 'none'; "
            "object-src 'none'; "
            "frame-ancestors 'none';"
        )
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Cache-Control"] = "no-store"
        return response


def broadcast(event: str, data: dict):
    msg = {"event": event, "data": json.dumps(data)}
    for q in list(_subscribers):
        try:
            q.put_nowait(msg)
        except asyncio.QueueFull:
            pass


async def dialog_loop():
    await asyncio.sleep(5)
    while True:
        try:
            state.generation_invalid = False
            speaker = state.next_speaker
            full_text = ""
            broadcast("generation_start", {"speaker": speaker})
            async for token in engine.generate(speaker):
                full_text += token
                broadcast("token", {"speaker": speaker, "token": token})
            if state.generation_invalid:
                # Config changed mid-generation — discard this turn
                state.generation_invalid = False
                continue
            cleaned = full_text.strip()
            for prefix in ["Drift:", "Echo:", "Drift: ", "Echo: "]:
                if cleaned.startswith(prefix):
                    cleaned = cleaned[len(prefix):].strip()
                    break
            turn = state.add_turn(speaker, cleaned)
            broadcast("turn_complete", turn)
        except (httpx.ConnectError, httpx.RemoteProtocolError) as e:
            await engine.close()
            logger.error("Connection error in dialog_loop: %s", e)
            broadcast("error", {"message": "generation error — retrying"})
            await asyncio.sleep(30)
        except Exception as e:
            logger.error("Error in dialog_loop: %s", e, exc_info=True)
            broadcast("error", {"message": "generation error — retrying"})
            await asyncio.sleep(30)
        else:
            await asyncio.sleep(PAUSE_SECONDS)


@asynccontextmanager
async def lifespan(app: FastAPI):
    task = asyncio.create_task(dialog_loop())
    yield
    task.cancel()
    await engine.close()


app = FastAPI(lifespan=lifespan)
app.add_middleware(SecurityHeadersMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://tinychat.sh"],
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)

BASE_DIR = Path(__file__).parent
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
templates = Jinja2Templates(directory=BASE_DIR / "templates")


def _state_context() -> dict:
    return {
        "prompts": state.prompts,
        "history": state.get_recent(50),
        "topic": state.topic,
        "cooldown": state.cooldown_remaining(),
    }


class UpdateRequest(BaseModel):
    drift: str
    echo: str
    topic: str

    @field_validator("drift", "echo")
    @classmethod
    def validate_prompts(cls, v: str) -> str:
        return validate_prompt_text(v)

    @field_validator("topic")
    @classmethod
    def validate_topic(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Topic cannot be empty")
        if len(v) > MAX_TOPIC_LENGTH:
            raise ValueError(f"Topic exceeds maximum length of {MAX_TOPIC_LENGTH}")
        return v.strip()


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(request, "index.html", _state_context())


@app.get("/history", response_class=HTMLResponse)
async def history_page(request: Request):
    return templates.TemplateResponse(request, "history.html", {
        "history": state.history,
    })


@app.get("/state")
async def get_state():
    return _state_context()


@app.post("/update")
async def update_all(body: UpdateRequest):
    try:
        state.reset_context(body.topic)
    except ValueError as e:
        return JSONResponse(status_code=429, content={"detail": str(e)})
    state.update_prompt("drift", body.drift)
    state.update_prompt("echo", body.echo)
    broadcast("config_updated", {
        "drift": body.drift,
        "echo": body.echo,
        "topic": body.topic,
        "cooldown": COOLDOWN_SECONDS,
    })
    return {"status": "ok"}


@app.get("/events")
async def events(request: Request):
    if len(_subscribers) >= MAX_SUBSCRIBERS:
        return JSONResponse(status_code=503, content={"detail": "Too many listeners"})

    queue: asyncio.Queue = asyncio.Queue(maxsize=100)
    _subscribers.add(queue)

    async def event_generator():
        try:
            while True:
                if await request.is_disconnected():
                    break
                try:
                    msg = await asyncio.wait_for(queue.get(), timeout=30.0)
                    yield msg
                except asyncio.TimeoutError:
                    yield {"event": "ping", "data": ""}
        finally:
            _subscribers.discard(queue)

    return EventSourceResponse(event_generator())
