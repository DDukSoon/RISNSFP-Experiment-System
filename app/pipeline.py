# -*- coding: utf-8 -*-
"""Voice turn pipeline: STT → agent → TTS + logging.

Kept free of FastAPI types so it is easy to unit-test with stubs.
"""
from __future__ import annotations

import base64
import io
import re
import time
import wave
from dataclasses import dataclass, field
from typing import Any, Callable

from app.sanitize import clean_for_tts

# Remove non-speech event tags that STT occasionally emits, e.g. "(mouse click sound)", "(2-second pause)".
# First line of defense: voice/stt.py calls with tag_audio_events=False. Second: strip here too.
# Two layers are needed to also cover old session history / other STT backends / such markers mixed into the user's speech.
_STT_EVENT_TAG_RE = re.compile(r"\s*\([^)]{2,30}\)\s*")


def _sanitize_user_text(text: str) -> str:
    """Clean up event tags and excess whitespace coming out of STT."""
    if not text:
        return text
    cleaned = _STT_EVENT_TAG_RE.sub(" ", text)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def _wav_duration_seconds(wav_bytes: bytes) -> float:
    """Return duration of a WAV payload in seconds. 0.0 on failure."""
    if not wav_bytes:
        return 0.0
    try:
        with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
            frames = wf.getnframes()
            rate = wf.getframerate()
            if rate <= 0:
                return 0.0
            return frames / float(rate)
    except (wave.Error, EOFError, OSError):
        return 0.0


@dataclass
class ChatResult:
    """Result of a single voice turn. Matches the frozen Unity wire format."""

    user_text: str
    agent_text: str
    audio_b64: str
    latency: dict[str, float]
    usage: dict[str, Any] = field(default_factory=dict)
    turn: int = 0

    def to_json(self) -> dict[str, Any]:
        return {
            "user_text": self.user_text,
            "agent_text": self.agent_text,
            "audio_b64": self.audio_b64,
            "latency": self.latency,
            "usage": self.usage,
            "turn": self.turn,
        }


def process_voice(
    audio_bytes: bytes,
    *,
    stt: Any,
    tts: Any,
    agent: Any,
    store: Any,
    pid: str,
    phase: str,
    condition: str | None,
    min_response_ms: float = 0.0,
    log: bool = True,
    now: Callable[[], float] = time.time,
    sleep: Callable[[float], None] = time.sleep,
) -> ChatResult:
    """Run one full voice turn.

    Raises:
        ValueError: if STT produces no text (caller should map to HTTP 400).
    """
    total_start = now()

    user_text, stt_ms = stt.transcribe(audio_bytes)
    user_text = _sanitize_user_text(user_text)
    if not user_text:
        raise ValueError("No speech recognized.")

    agent_result = agent.chat(user_text)
    # Agent.chat may return 2-tuple (legacy tests) or 3-tuple (current).
    if len(agent_result) == 3:
        agent_text, llm_ms, agent_usage = agent_result
    else:
        agent_text, llm_ms = agent_result
        agent_usage = {}
    agent_text = clean_for_tts(agent_text)

    # TTS cache check before synthesize (for usage tracking).
    tts_cache_hit = None
    if hasattr(tts, "is_cached"):
        try:
            tts_cache_hit = bool(tts.is_cached(agent_text))
        except Exception:
            tts_cache_hit = None

    # process_voice is exclusive to the Phase 3 /chat path. Since agent_text is LLM-generated,
    # the chance of the same agent_text recurring is near 0 → skip the cache write to avoid voice_cache/ buildup.
    audio_out, tts_ms = tts.synthesize(agent_text, cache_write=False)

    if min_response_ms > 0:
        elapsed_so_far_ms = (now() - total_start) * 1000
        if elapsed_so_far_ms < min_response_ms:
            sleep((min_response_ms - elapsed_so_far_ms) / 1000.0)

    total_ms = (now() - total_start) * 1000
    latency = {
        "stt_ms":   round(stt_ms, 1),
        "llm_ms":   round(llm_ms, 1),
        "tts_ms":   round(tts_ms, 1),
        "total_ms": round(total_ms, 1),
    }

    usage: dict[str, Any] = {
        "stt_audio_sec": round(_wav_duration_seconds(audio_bytes), 3),
        "stt_chars_out": len(user_text),
        "tts_chars_in": len(agent_text),
        "tts_cache_hit": tts_cache_hit,
        "llm_prompt_tokens": int(agent_usage.get("llm_prompt_tokens", 0) or 0),
        "llm_completion_tokens": int(agent_usage.get("llm_completion_tokens", 0) or 0),
        "llm_total_tokens": int(agent_usage.get("llm_total_tokens", 0) or 0),
        "llm_tokens_estimated": bool(agent_usage.get("llm_tokens_estimated", False)),
    }

    turn_no = 0
    if log:
        turn_no = store.log_turn(
            pid=pid,
            phase=phase,
            user_text=user_text,
            agent_text=agent_text,
            latency=latency,
            condition=condition,
            usage=usage,
        )

    return ChatResult(
        user_text=user_text,
        agent_text=agent_text,
        audio_b64=base64.b64encode(audio_out).decode(),
        latency=latency,
        usage=usage,
        turn=turn_no,
    )
