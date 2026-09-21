# -*- coding: utf-8 -*-
"""FastAPI app wiring + process-wide singletons + shutdown timers.

Two phase modes:

- Phase 3: exposes ``POST /chat`` (frozen Unity contract) for free dialog.
- Phase 1: exposes only ``/phase1/*`` — Unity drives navigation, server
  owns per-index answer storage. ``/chat`` is **not** registered.

The app-level ``AppState`` is populated from CLI args before
``uvicorn.run`` is called (see ``server.py`` at the repo root).

``/chat`` wire format (Phase 3):
    request : multipart form, field ``audio`` = WAV bytes
    response: JSON {user_text, agent_text, audio_b64, latency{...}, usage{...}}
"""
from __future__ import annotations

import os
import threading
from dataclasses import dataclass, field
from typing import Any

from fastapi import Body, FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from agents import (
    Phase3RetrievalAgent,
    Phase3InContextAgent,
    Phase3ParametricAgent,
)
from agents.base import Agent
from app.config import Settings
from app.phase1_manager import Phase1Manager
from app.pipeline import process_voice
from app.routes_phase1 import router as phase1_router
from app.routes_phase3 import router as phase3_router
from app.store import SessionStore
from voice import ElevenLabsSTT, ElevenLabsTTS


# ────────────────────────────────────────────────────────────────────
# AppState + singleton loading
# ────────────────────────────────────────────────────────────────────

@dataclass
class AppState:
    """All singletons the request handler needs."""

    settings: Settings
    participant: str
    phase: str                 # "1" or "3"
    condition: str | None      # required iff phase == "3"

    stt: Any = None
    tts: Any = None
    store: SessionStore | None = None
    agent: Agent | None = None
    phase1_manager: Phase1Manager | None = None
    agent_lock: threading.Lock = field(default_factory=threading.Lock)
    phase1_shutdown_scheduled: bool = False
    phase3_shutdown_scheduled: bool = False


def build_agent(state: AppState) -> Agent:
    """Instantiate the Phase 3 agent for the fixed condition.

    Phase 1 does not use an :class:`Agent`; see :class:`Phase1Manager` and
    the dedicated ``/phase1/*`` routes.
    """
    s = state.settings
    if state.phase == "3":
        print(
            f"[server] loading Phase3 agent for participant={state.participant} "
            f"condition={state.condition}..."
        )
        model_path = str(s.model_path)
        if state.condition == "in_context":
            return Phase3InContextAgent(state.participant, model_path)
        if state.condition == "retrieval":
            return Phase3RetrievalAgent(state.participant, model_path)
        if state.condition == "parametric":
            return Phase3ParametricAgent(state.participant, model_path)
        raise RuntimeError(f"Invalid condition={state.condition!r}")
    raise RuntimeError(f"Invalid phase={state.phase!r}")


def load_singletons(state: AppState) -> None:
    """Wire up STT/TTS/store + (phase3) agent or (phase1) manager on startup."""
    print("[server] initializing STT (ElevenLabs Scribe)...")
    state.stt = ElevenLabsSTT()
    print("[server] initializing TTS (ElevenLabs)...")
    state.tts = ElevenLabsTTS()
    state.store = SessionStore(state.settings.base_dir)

    with state.agent_lock:
        if state.agent is not None:
            try:
                state.agent.cleanup()
            except Exception:
                pass
            state.agent = None

        if state.phase == "1":
            print(
                f"[server] loading Phase1Manager for participant={state.participant}..."
            )
            mgr = Phase1Manager(
                script_path=state.settings.phase1_script_path,
                store=state.store,
                tts=state.tts,
            )
            state.phase1_manager = mgr
            lines = mgr.preloadable_lines()
            if lines:
                state.tts.prewarm(lines)
            print(
                f"[server] phase1 resume: "
                f"{mgr.total - len(mgr.unanswered_indices(state.participant))}"
                f"/{mgr.total} already answered"
            )
        else:
            state.agent = build_agent(state)
            lines = state.agent.preloadable_lines()
            if lines:
                state.tts.prewarm(lines)
            if hasattr(state.agent, "warmup"):
                state.agent.warmup()

    print("[server] agent ready.")


# ────────────────────────────────────────────────────────────────────
# Graceful shutdown timers (Phase 1 finish / Phase 3 abort)
# ────────────────────────────────────────────────────────────────────

def schedule_phase1_shutdown(state: AppState, reason: str = "all questions answered") -> None:
    """Start the single-shot auto-shutdown timer."""
    if state.phase1_shutdown_scheduled:
        return
    state.phase1_shutdown_scheduled = True
    delay = float(state.settings.phase1_shutdown_delay_sec)
    print(f"[server] Phase 1 {reason}. Server will auto-shut down in {delay:.0f}s.")

    def _shutdown() -> None:
        print("[server] Phase 1 ended — shutting down server.")
        os._exit(0)

    threading.Timer(delay, _shutdown).start()


def schedule_phase3_shutdown(state: AppState, reason: str = "Phase 3 shutdown requested") -> None:
    """Phase 3 graceful-shutdown timer. Invoked by /phase3/abort when Unity Play ends."""
    if state.phase3_shutdown_scheduled:
        return
    state.phase3_shutdown_scheduled = True
    delay = float(state.settings.phase1_shutdown_delay_sec)
    print(f"[server] Phase 3 {reason}. Server will auto-shut down in {delay:.0f}s.")

    def _shutdown() -> None:
        print("[server] Phase 3 ended — shutting down server.")
        os._exit(0)

    threading.Timer(delay, _shutdown).start()


# ────────────────────────────────────────────────────────────────────
# FastAPI app factory
# ────────────────────────────────────────────────────────────────────

def create_app(
    *,
    participant: str,
    phase: str,
    condition: str | None,
    settings: Settings | None = None,
) -> FastAPI:
    """Build a FastAPI app bound to the given fixed run configuration."""
    app = FastAPI(title="HCI-Memory Experiment Server")
    state = AppState(
        settings=settings or Settings.load(),
        participant=participant.strip(),
        phase=phase,
        condition=condition if phase == "3" else None,
    )
    app.state.session = state

    @app.on_event("startup")
    async def _on_startup() -> None:
        print(
            f"[server] fixed participant={state.participant} "
            f"phase={state.phase} condition={state.condition}"
        )
        load_singletons(state)

    if phase == "1":
        app.include_router(phase1_router)
    else:
        app.include_router(phase3_router)

        @app.post("/chat")
        async def chat(audio: UploadFile = File(...)) -> JSONResponse:
            with state.agent_lock:
                agent = state.agent
            if agent is None:
                raise HTTPException(status_code=503, detail="Agent not loaded.")

            audio_bytes = await audio.read()

            try:
                result = process_voice(
                    audio_bytes,
                    stt=state.stt,
                    tts=state.tts,
                    agent=agent,
                    store=state.store,
                    pid=state.participant,
                    phase="phase3",
                    condition=state.condition,
                    min_response_ms=0.0,
                )
            except ValueError as e:
                raise HTTPException(status_code=400, detail=str(e))

            if result.usage:
                try:
                    state.store.update_phase3_usage_total(
                        state.participant, state.condition, result.usage,
                    )
                except Exception:
                    pass

            return JSONResponse(result.to_json())

        @app.post("/chat/client_latency")
        async def chat_client_latency(payload: dict = Body(...)) -> JSONResponse:
            """Called by Unity at the moment it actually plays the first audio sample.

            Request: {"turn": int, "record_to_play_ms": float}
            Storage: merged into that turn's `latency` in
                     outputs/<pid>/phase3/<cond>_log.json under the
                     `client_e2e_ms` key (separate from the server's total_ms).
            """
            try:
                turn = int(payload.get("turn", 0))
                e2e_ms = float(payload.get("record_to_play_ms", 0.0))
            except (TypeError, ValueError):
                raise HTTPException(status_code=400, detail="invalid turn/record_to_play_ms format")
            if turn <= 0:
                raise HTTPException(status_code=400, detail="turn must be >= 1")

            ok = state.store.update_turn_latency(
                pid=state.participant,
                phase="phase3",
                turn=turn,
                extra={"client_e2e_ms": round(e2e_ms, 1)},
                condition=state.condition,
            )
            if not ok:
                raise HTTPException(status_code=404, detail=f"turn {turn} not found")
            return JSONResponse({"ok": True, "turn": turn, "client_e2e_ms": round(e2e_ms, 1)})

    return app
