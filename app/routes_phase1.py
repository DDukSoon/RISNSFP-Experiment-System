# -*- coding: utf-8 -*-
"""FastAPI routes for the Phase 1 Unity-driven UX.

Endpoints
---------
- ``GET  /phase1/manifest``              — participant + script state snapshot
- ``GET  /phase1/audio/{index}``         — cached TTS WAV for a question (or 'done')
- ``POST /phase1/answer/{index}``        — upload answer WAV, STT only (no TTS)
- ``POST /phase1/finish``                — finalize. 200 if all answered, 409 otherwise.
- ``POST /phase1/abort``                 — schedule shutdown without completion check.
- ``GET  /phase1/healthz``               — lightweight liveness probe for Unity.

All routes live on a dedicated :class:`APIRouter` that :mod:`app.server`
only registers when the server boots with ``--phase 1``.
"""
from __future__ import annotations

import hashlib
from typing import Any

from fastapi import APIRouter, File, HTTPException, Request, Response, UploadFile
from fastapi.responses import JSONResponse

from app.phase1_manager import Phase1NotComplete
from app.store import UsageDelta, stt_audio_seconds


router = APIRouter(prefix="/phase1", tags=["phase1"])


# ── internal helpers ─────────────────────────────────────────────

def _state(request: Request):
    """Return the AppState attached to app.state.session."""
    return request.app.state.session


def _require_phase1_manager(request: Request):
    state = _state(request)
    if getattr(state, "phase", None) != "1":
        raise HTTPException(status_code=404, detail="Phase 1 not active")
    mgr = getattr(state, "phase1_manager", None)
    if mgr is None:
        raise HTTPException(status_code=503, detail="Phase 1 manager not loaded")
    return state, mgr


def _parse_index(index_str: str, total: int) -> int | str:
    """Return either an int index or the sentinel ``"done"``."""
    if index_str == "done":
        return "done"
    try:
        i = int(index_str)
    except ValueError:
        raise HTTPException(status_code=404, detail=f"bad index: {index_str!r}")
    if i < 0 or i >= total:
        raise HTTPException(status_code=404, detail=f"index out of range: {i}")
    return i


# ── GET /phase1/manifest ────────────────────────────────────────

@router.get("/manifest")
def get_manifest(request: Request) -> dict[str, Any]:
    state, mgr = _require_phase1_manager(request)
    return mgr.build_manifest(state.participant)


# ── GET /phase1/audio/{index} ────────────────────────────────────

@router.get("/audio/{index}")
def get_audio(index: str, request: Request) -> Response:
    state, mgr = _require_phase1_manager(request)
    parsed = _parse_index(index, mgr.total)

    text = mgr.done_message if parsed == "done" else mgr.question_text(parsed)
    tts = state.tts
    if tts is None:
        raise HTTPException(status_code=503, detail="TTS not loaded")

    cache_path = tts.get_cache_path(text)
    wav_bytes: bytes
    try:
        with open(cache_path, "rb") as f:
            wav_bytes = f.read()
    except FileNotFoundError:
        # Cache miss — synthesize on demand (should be rare after prewarm).
        try:
            wav_bytes, _ = tts.synthesize(text)
        except Exception as e:  # pragma: no cover — external API
            raise HTTPException(status_code=500, detail=f"TTS failed: {e}")

    etag = hashlib.sha256(cache_path.encode("utf-8")).hexdigest()[:16]
    return Response(
        content=wav_bytes,
        media_type="audio/wav",
        headers={"ETag": etag},
    )


# ── POST /phase1/answer/{index} ──────────────────────────────────

@router.post("/answer/{index}")
async def post_answer(index: int, request: Request, audio: UploadFile = File(...)):
    state, mgr = _require_phase1_manager(request)
    if index < 0 or index >= mgr.total:
        raise HTTPException(status_code=404, detail=f"index out of range: {index}")

    audio_bytes = await audio.read()
    stt = state.stt
    if stt is None:
        raise HTTPException(status_code=503, detail="STT not loaded")

    user_text, stt_ms = stt.transcribe(audio_bytes)
    if not user_text:
        raise HTTPException(status_code=400, detail="No speech recognized.")

    usage = UsageDelta(
        stt_audio_sec=round(stt_audio_seconds(audio_bytes), 3),
        stt_chars_out=len(user_text),
        tts_chars_in=0,
        tts_cache_hit=None,
        llm_prompt_tokens=0,
        llm_completion_tokens=0,
        llm_total_tokens=0,
        llm_tokens_estimated=False,
    )

    record = mgr.record_answer(
        pid=state.participant,
        index=index,
        user_text=user_text,
        stt_ms=float(stt_ms),
        usage=usage,
    )

    progress_unanswered = mgr.unanswered_indices(state.participant)
    answered_count = mgr.total - len(progress_unanswered)

    return JSONResponse({
        "index": record.index,
        "user_text": record.user_text,
        "revision": record.revision,
        "total_revisions": record.total_revisions,
        "answered": record.answered,
        "latency": {
            "stt_ms": round(float(stt_ms), 1),
            "total_ms": round(float(stt_ms), 1),
        },
        "usage": usage.to_dict(),
        "progress": {
            "answered_count": answered_count,
            "total": mgr.total,
            "unanswered_indices": progress_unanswered,
        },
    })


# ── POST /phase1/finish ──────────────────────────────────────────

@router.post("/finish")
def post_finish(request: Request):
    state, mgr = _require_phase1_manager(request)
    try:
        payload = mgr.finish(state.participant)
    except Phase1NotComplete as exc:
        return JSONResponse(
            status_code=409,
            content={
                "status": "incomplete",
                "unanswered_indices": exc.unanswered_indices,
                "answered_count": exc.answered_count,
                "total": exc.total,
            },
        )

    # Schedule auto-shutdown like the legacy DONE path.
    from app.server import schedule_phase1_shutdown

    schedule_phase1_shutdown(state)
    payload["shutdown_in_sec"] = float(state.settings.phase1_shutdown_delay_sec)
    return JSONResponse(payload)


# ── POST /phase1/abort ───────────────────────────────────────────

@router.post("/abort")
def post_abort(request: Request):
    """Schedule graceful shutdown without requiring all answers.

    Called when Unity stops Play mode. Since every answer is already persisted
    to disk, this endpoint only schedules the shutdown without any extra save.
    It does not set the phase1_done flag, so a later restart can resume from
    the remaining unanswered questions.
    """
    state, mgr = _require_phase1_manager(request)
    progress_unanswered = mgr.unanswered_indices(state.participant)
    answered_count = mgr.total - len(progress_unanswered)

    from app.server import schedule_phase1_shutdown

    schedule_phase1_shutdown(
        state,
        reason=f"Unity abort received ({answered_count}/{mgr.total} answered, "
               f"{len(progress_unanswered)} unanswered)",
    )
    return JSONResponse({
        "status": "aborted",
        "answered_count": answered_count,
        "total": mgr.total,
        "unanswered_indices": progress_unanswered,
        "shutdown_in_sec": float(state.settings.phase1_shutdown_delay_sec),
    })


# ── GET /phase1/healthz ──────────────────────────────────────────

@router.get("/healthz")
def get_healthz(request: Request):
    """Lightweight liveness probe. Only servers with the Phase1 router mounted respond."""
    state = _state(request)
    return JSONResponse({
        "ok": True,
        "phase": getattr(state, "phase", None),
        "participant": getattr(state, "participant", None),
        "shutdown_scheduled": bool(getattr(state, "phase1_shutdown_scheduled", False)),
    })
