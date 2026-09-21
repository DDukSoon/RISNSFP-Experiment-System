# -*- coding: utf-8 -*-
"""FastAPI routes for the Phase 3 Unity-driven UX (auxiliary endpoints).

The frozen ``POST /chat`` contract lives in :mod:`app.server` directly.
This router only exposes auxiliary endpoints that Unity needs for lifecycle
management — symmetric to ``/phase1/*``:

- ``GET  /phase3/healthz`` — liveness + agent-readiness probe.
- ``POST /phase3/abort``   — schedule graceful shutdown when Unity stops Play.
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse


router = APIRouter(prefix="/phase3", tags=["phase3"])


def _state(request: Request):
    return request.app.state.session


def _require_phase3(request: Request):
    state = _state(request)
    if getattr(state, "phase", None) != "3":
        raise HTTPException(status_code=404, detail="Phase 3 not active")
    return state


# ── GET /phase3/healthz ──────────────────────────────────────────

@router.get("/healthz")
def get_healthz(request: Request) -> JSONResponse:
    """Liveness + agent-readiness check.

    If ``agent_ready`` is false the LLM is still loading, so Unity should hold
    off on sending /chat and wait briefly.
    """
    state = _state(request)
    agent_ready = bool(getattr(state, "agent", None) is not None)
    return JSONResponse({
        "ok": True,
        "phase": getattr(state, "phase", None),
        "participant": getattr(state, "participant", None),
        "condition": getattr(state, "condition", None),
        "agent_ready": agent_ready,
        "shutdown_scheduled": bool(getattr(state, "phase3_shutdown_scheduled", False)),
    })


# ── POST /phase3/abort ───────────────────────────────────────────

@router.post("/abort")
def post_abort(request: Request) -> JSONResponse:
    """Graceful shutdown trigger called when Unity exits Play mode."""
    state = _require_phase3(request)
    from app.server import schedule_phase3_shutdown

    schedule_phase3_shutdown(state, reason="Unity abort received")
    return JSONResponse({
        "status": "aborted",
        "shutdown_in_sec": float(state.settings.phase1_shutdown_delay_sec),
    })
