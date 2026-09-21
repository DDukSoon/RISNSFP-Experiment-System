# -*- coding: utf-8 -*-
"""Filesystem IO for participant state + usage accounting.

Handles:
- ``status.json`` read/write (condition_order, phase progress, usage totals)
- ``phase1/answers.json`` (revision history)
- ``phase1/conversation_log.json`` + ``phase3/<cond>_log.json`` (per-turn)
- Deterministic Latin-square condition order for participant IDs.
- ``UsageDelta`` + accumulators (Phase 1 / Phase 3).

Phase 2 build state is handled by :mod:`build_memory`.
"""
from __future__ import annotations

import io
import json
import wave
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


# ────────────────────────────────────────────────────────────────────
# Latin-square condition order
# ────────────────────────────────────────────────────────────────────

LATIN_SQUARE: list[list[str]] = [
    ["in_context",  "retrieval",   "parametric"],
    ["retrieval",    "parametric",   "in_context"],
    ["parametric",    "in_context", "retrieval"],
    ["in_context",  "parametric",   "retrieval"],
    ["retrieval",    "in_context", "parametric"],
    ["parametric",    "retrieval",   "in_context"],
]


def get_condition_order(participant_id: str) -> list[str]:
    """Latin-square condition order for a participant id like ``P007``."""
    digits = "".join(filter(str.isdigit, participant_id))
    num = int(digits) if digits else 1
    idx = (num - 1) % len(LATIN_SQUARE)
    return LATIN_SQUARE[idx]


# ────────────────────────────────────────────────────────────────────
# Usage delta + accumulators
# ────────────────────────────────────────────────────────────────────

def stt_audio_seconds(wav_bytes: bytes) -> float:
    """Return duration of a WAV payload in seconds. 0.0 on any failure."""
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
class UsageDelta:
    """One turn's usage (Phase 1 or Phase 3)."""

    stt_audio_sec: float = 0.0
    stt_chars_out: int = 0
    tts_chars_in: int = 0
    tts_cache_hit: bool | None = None
    llm_prompt_tokens: int = 0
    llm_completion_tokens: int = 0
    llm_total_tokens: int = 0
    llm_tokens_estimated: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _empty_phase1_total() -> dict[str, Any]:
    return {
        "stt_audio_sec": 0.0,
        "stt_chars_out": 0,
        "tts_chars_in": 0,
        "tts_cache_hit_count": 0,
        "tts_cache_miss_count": 0,
        "llm_prompt_tokens": 0,
        "llm_completion_tokens": 0,
        "llm_total_tokens": 0,
        "revision_count": 0,
        "turn_count": 0,
    }


def _empty_phase3_total() -> dict[str, Any]:
    return {
        "stt_audio_sec": 0.0,
        "stt_chars_out": 0,
        "tts_chars_in": 0,
        "tts_cache_hit_count": 0,
        "tts_cache_miss_count": 0,
        "llm_prompt_tokens": 0,
        "llm_completion_tokens": 0,
        "llm_total_tokens": 0,
        "llm_tokens_estimated": False,
        "turn_count": 0,
    }


def add_delta_to_phase1_total(
    total: dict[str, Any] | None, delta: UsageDelta | dict[str, Any],
    *, count_revision: bool = True, count_new_turn: bool = False,
) -> dict[str, Any]:
    """Fold a per-turn delta into a phase1 running total (returned new dict)."""
    out = dict(total) if total else _empty_phase1_total()
    d = delta.to_dict() if isinstance(delta, UsageDelta) else dict(delta)

    out["stt_audio_sec"] = round(
        float(out.get("stt_audio_sec", 0.0)) + float(d.get("stt_audio_sec", 0.0)), 3
    )
    out["stt_chars_out"] = int(out.get("stt_chars_out", 0)) + int(d.get("stt_chars_out", 0))
    out["tts_chars_in"] = int(out.get("tts_chars_in", 0)) + int(d.get("tts_chars_in", 0))

    hit = d.get("tts_cache_hit")
    if hit is True:
        out["tts_cache_hit_count"] = int(out.get("tts_cache_hit_count", 0)) + 1
    elif hit is False:
        out["tts_cache_miss_count"] = int(out.get("tts_cache_miss_count", 0)) + 1

    out["llm_prompt_tokens"] = int(out.get("llm_prompt_tokens", 0)) + int(d.get("llm_prompt_tokens", 0))
    out["llm_completion_tokens"] = int(out.get("llm_completion_tokens", 0)) + int(d.get("llm_completion_tokens", 0))
    out["llm_total_tokens"] = int(out.get("llm_total_tokens", 0)) + int(d.get("llm_total_tokens", 0))

    if count_revision:
        out["revision_count"] = int(out.get("revision_count", 0)) + 1
    if count_new_turn:
        out["turn_count"] = int(out.get("turn_count", 0)) + 1

    return out


def add_delta_to_phase3_total(
    total: dict[str, Any] | None, delta: UsageDelta | dict[str, Any],
) -> dict[str, Any]:
    """Fold a per-turn delta into a phase3 (per-condition) running total."""
    out = dict(total) if total else _empty_phase3_total()
    d = delta.to_dict() if isinstance(delta, UsageDelta) else dict(delta)

    out["stt_audio_sec"] = round(
        float(out.get("stt_audio_sec", 0.0)) + float(d.get("stt_audio_sec", 0.0)), 3
    )
    out["stt_chars_out"] = int(out.get("stt_chars_out", 0)) + int(d.get("stt_chars_out", 0))
    out["tts_chars_in"] = int(out.get("tts_chars_in", 0)) + int(d.get("tts_chars_in", 0))

    hit = d.get("tts_cache_hit")
    if hit is True:
        out["tts_cache_hit_count"] = int(out.get("tts_cache_hit_count", 0)) + 1
    elif hit is False:
        out["tts_cache_miss_count"] = int(out.get("tts_cache_miss_count", 0)) + 1

    out["llm_prompt_tokens"] = int(out.get("llm_prompt_tokens", 0)) + int(d.get("llm_prompt_tokens", 0))
    out["llm_completion_tokens"] = int(out.get("llm_completion_tokens", 0)) + int(d.get("llm_completion_tokens", 0))
    out["llm_total_tokens"] = int(out.get("llm_total_tokens", 0)) + int(d.get("llm_total_tokens", 0))
    out["llm_tokens_estimated"] = bool(out.get("llm_tokens_estimated", False)) or bool(
        d.get("llm_tokens_estimated", False)
    )
    out["turn_count"] = int(out.get("turn_count", 0)) + 1

    return out


# ────────────────────────────────────────────────────────────────────
# SessionStore — participant filesystem IO
# ────────────────────────────────────────────────────────────────────

class SessionStore:
    """Per-participant status + conversation log IO."""

    def __init__(self, base_dir: str | Path):
        self.base_dir = Path(base_dir)
        self.participants_dir = self.base_dir / "participants"
        self.participants_dir.mkdir(parents=True, exist_ok=True)

    # ── Paths ────────────────────────────────────────────────────
    def participant_dir(self, pid: str) -> Path:
        d = self.participants_dir / pid
        d.mkdir(parents=True, exist_ok=True)
        return d

    def status_path(self, pid: str) -> Path:
        return self.participant_dir(pid) / "status.json"

    def log_path(self, pid: str, phase: str, condition: str | None = None) -> Path:
        d = self.participant_dir(pid) / phase
        d.mkdir(parents=True, exist_ok=True)
        fname = f"{condition}_log.json" if condition else "conversation_log.json"
        return d / fname

    def phase1_answers_path(self, pid: str) -> Path:
        """Phase 1 answers.json — indexed, revision-tracking source of truth."""
        d = self.participant_dir(pid) / "phase1"
        d.mkdir(parents=True, exist_ok=True)
        return d / "answers.json"

    # ── status.json ──────────────────────────────────────────────
    def get_status(self, pid: str) -> dict[str, Any]:
        path = self.status_path(pid)
        if path.exists():
            with path.open(encoding="utf-8") as f:
                status = json.load(f)
        else:
            status = {
                "participant_id": pid,
                "condition_order": get_condition_order(pid),
                "phase1_done": False,
                "phase2_status": "not_started",
                "conditions_completed": [],
                "phase1_usage_total": {},
                "phase3_usage_total": {},
                "created_at": datetime.now().isoformat(),
            }
            self.save_status(pid, status)
            return status

        mutated = False
        if "phase1_usage_total" not in status:
            status["phase1_usage_total"] = {}
            mutated = True
        if "phase3_usage_total" not in status:
            status["phase3_usage_total"] = {}
            mutated = True
        if mutated:
            self.save_status(pid, status)
        return status

    def save_status(self, pid: str, status: dict[str, Any]) -> None:
        with self.status_path(pid).open("w", encoding="utf-8") as f:
            json.dump(status, f, ensure_ascii=False, indent=2)

    # ── Phase 1 answers.json ─────────────────────────────────────
    def load_phase1_answers(self, pid: str) -> dict[str, Any]:
        path = self.phase1_answers_path(pid)
        if path.exists():
            with path.open(encoding="utf-8") as f:
                return json.load(f)
        return {
            "participant_id": pid,
            "phase": "1",
            "total": 0,
            "started_at": None,
            "ended_at": None,
            "answers": {},
        }

    def save_phase1_answers(self, pid: str, data: dict[str, Any]) -> None:
        path = self.phase1_answers_path(pid)
        with path.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    # ── Usage total mutators ─────────────────────────────────────
    def update_phase1_usage_total(
        self, pid: str, delta: dict[str, Any],
        *, count_revision: bool = True, count_new_turn: bool = False,
    ) -> dict[str, Any]:
        status = self.get_status(pid)
        status["phase1_usage_total"] = add_delta_to_phase1_total(
            status.get("phase1_usage_total") or {},
            delta,
            count_revision=count_revision,
            count_new_turn=count_new_turn,
        )
        self.save_status(pid, status)
        return status

    def update_phase3_usage_total(
        self, pid: str, condition: str, delta: dict[str, Any],
    ) -> dict[str, Any]:
        status = self.get_status(pid)
        totals = status.get("phase3_usage_total") or {}
        totals[condition] = add_delta_to_phase3_total(totals.get(condition) or {}, delta)
        status["phase3_usage_total"] = totals
        self.save_status(pid, status)
        return status

    # ── Conversation log ─────────────────────────────────────────
    def log_turn(
        self,
        pid: str,
        phase: str,
        user_text: str,
        agent_text: str,
        latency: dict[str, float],
        condition: str | None = None,
        usage: dict[str, Any] | None = None,
    ) -> int:
        """Append a turn entry. Returns the assigned turn number (1-based)."""
        path = self.log_path(pid, phase, condition)
        turns: list[dict[str, Any]] = []
        if path.exists():
            with path.open(encoding="utf-8") as f:
                turns = json.load(f)

        turn_no = len(turns) + 1
        entry: dict[str, Any] = {
            "turn": turn_no,
            "timestamp": datetime.now().isoformat(),
            "user_text": user_text,
            "agent_text": agent_text,
            "latency": latency,
        }
        if usage is not None:
            entry["usage"] = usage
        turns.append(entry)

        with path.open("w", encoding="utf-8") as f:
            json.dump(turns, f, ensure_ascii=False, indent=2)
        return turn_no

    def update_turn_latency(
        self,
        pid: str,
        phase: str,
        turn: int,
        extra: dict[str, float],
        condition: str | None = None,
    ) -> bool:
        """Merge ``extra`` into an existing turn's ``latency`` dict.

        Used by the Unity client to attach end-to-end (record_end → first audio
        sample) latency after the /chat round-trip + audio playback start.
        Returns True on success, False if the turn was not found.
        """
        path = self.log_path(pid, phase, condition)
        if not path.exists():
            return False
        with path.open(encoding="utf-8") as f:
            turns = json.load(f)
        for entry in turns:
            if entry.get("turn") == turn:
                lat = entry.get("latency") or {}
                lat.update(extra)
                entry["latency"] = lat
                with path.open("w", encoding="utf-8") as f:
                    json.dump(turns, f, ensure_ascii=False, indent=2)
                return True
        return False
