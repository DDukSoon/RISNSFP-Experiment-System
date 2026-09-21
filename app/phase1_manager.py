# -*- coding: utf-8 -*-
"""Phase 1 manager — stateless orchestration of the scripted Q&A flow.

Responsibilities
----------------
* Load the question YAML once (:func:`_load_script`).
* Provide TTS prewarm lines (``preloadable_lines``).
* Expose ``cache_hit`` per-question via :class:`voice.tts.ElevenLabsTTS.is_cached`.
* Read / write the participant's ``answers.json`` with revision history.
* Mirror the latest revision into the legacy ``conversation_log.json`` so
  Phase 2 (``build_memory``) keeps working unchanged.
* Aggregate usage into ``status.json['phase1_usage_total']``.
* Decide when :func:`finish` can succeed and raise otherwise.

The manager does no network / STT / TTS work itself — callers (the FastAPI
routes) supply ``user_text``, ``stt_ms`` and ``usage`` after running STT.
"""
from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from app.store import UsageDelta


def _load_script(path: Path) -> tuple[list[str], str]:
    """Load the Phase 1 question script (questions + done_message) from YAML."""
    with path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    questions = list(data.get("questions") or [])
    done = str(data.get("done_message") or "")
    if not questions or not done:
        raise ValueError(f"Invalid Phase 1 script at {path}")
    return questions, done


class Phase1NotComplete(Exception):
    """Raised by :meth:`Phase1Manager.finish` when answers are missing."""

    def __init__(self, unanswered_indices: list[int], answered_count: int, total: int):
        self.unanswered_indices = unanswered_indices
        self.answered_count = answered_count
        self.total = total
        super().__init__(
            f"Phase 1 incomplete: {answered_count}/{total} answered, "
            f"unanswered={unanswered_indices}"
        )


@dataclass
class AnswerRecord:
    """Return value of :meth:`Phase1Manager.record_answer`."""

    index: int
    user_text: str
    revision: int
    total_revisions: int
    answered: bool


class Phase1Manager:
    """Phase 1 file-backed manager. Instantiate once per server process."""

    def __init__(
        self,
        script_path: str | Path,
        store: Any,                # app.store.SessionStore
        tts: Any | None = None,    # voice.tts.ElevenLabsTTS (for cache_hit probe)
    ) -> None:
        self.script_path = Path(script_path)
        self.questions, self.done_message = _load_script(self.script_path)
        self.store = store
        self._tts = tts
        self._lock = threading.Lock()
        print(
            f"[Phase1Manager] script loaded "
            f"({len(self.questions)} questions, prewarming including done_message)"
        )

    # ── TTS prewarm (called from lifecycle.load_singletons) ──────
    def preloadable_lines(self) -> list[str]:
        return list(self.questions) + [self.done_message]

    # ── Read-only accessors ──────────────────────────────────────
    @property
    def total(self) -> int:
        return len(self.questions)

    def question_text(self, index: int) -> str:
        if index < 0 or index >= self.total:
            raise IndexError(f"index out of range: {index}")
        return self.questions[index]

    # ── answers.json helpers ─────────────────────────────────────
    def _load_answers(self, pid: str) -> dict[str, Any]:
        data = self.store.load_phase1_answers(pid)
        # Ensure shape matches current schema (older files might miss fields).
        data.setdefault("participant_id", pid)
        data.setdefault("phase", "1")
        data["total"] = self.total
        data.setdefault("answers", {})
        data.setdefault("started_at", None)
        data.setdefault("ended_at", None)
        return data

    def _is_answered(self, answers_map: dict[str, Any], index: int) -> bool:
        entry = answers_map.get(str(index))
        return bool(entry) and int(entry.get("current_revision", 0)) >= 1

    # ── Public operations ────────────────────────────────────────
    def unanswered_indices(self, pid: str) -> list[int]:
        data = self._load_answers(pid)
        amap = data.get("answers") or {}
        return [i for i in range(self.total) if not self._is_answered(amap, i)]

    def build_manifest(self, pid: str) -> dict[str, Any]:
        """Snapshot for ``GET /phase1/manifest``."""
        data = self._load_answers(pid)
        amap = data.get("answers") or {}
        questions_out: list[dict[str, Any]] = []
        for i, text in enumerate(self.questions):
            entry = amap.get(str(i))
            answered = bool(entry) and int(entry.get("current_revision", 0)) >= 1
            revisions = int(entry.get("current_revision", 0)) if entry else 0
            cache_hit = bool(self._tts.is_cached(text)) if self._tts else False
            last_user_text = ""
            if answered and entry and entry.get("revisions"):
                last_user_text = entry["revisions"][-1].get("user_text") or ""
            questions_out.append({
                "index": i,
                "text": text,
                "audio_url": f"/phase1/audio/{i}",
                "cache_hit": cache_hit,
                "answered": answered,
                "revisions": revisions,
                "last_user_text": last_user_text,
            })
        done_cache_hit = bool(self._tts.is_cached(self.done_message)) if self._tts else False
        unanswered = [i for i in range(self.total) if not self._is_answered(amap, i)]
        return {
            "participant_id": pid,
            "phase": "1",
            "total": self.total,
            "questions": questions_out,
            "done_message": self.done_message,
            "done_audio_url": "/phase1/audio/done",
            "done_cache_hit": done_cache_hit,
            "all_answered": not unanswered,
            "unanswered_indices": unanswered,
        }

    def record_answer(
        self,
        pid: str,
        index: int,
        user_text: str,
        stt_ms: float,
        usage: UsageDelta | dict[str, Any],
    ) -> AnswerRecord:
        """Persist one answer (new or revision). Mutates answers.json, conversation_log.json,
        and status.json atomically under an instance-level lock.
        """
        if index < 0 or index >= self.total:
            raise IndexError(f"index out of range: {index}")

        usage_dict = usage.to_dict() if isinstance(usage, UsageDelta) else dict(usage)
        ts = datetime.now().isoformat()

        with self._lock:
            data = self._load_answers(pid)
            if data.get("started_at") is None:
                data["started_at"] = ts

            amap = data["answers"]
            entry = amap.get(str(index))
            is_new_turn = entry is None
            if is_new_turn:
                entry = {
                    "question_text": self.questions[index],
                    "first_answered_at": ts,
                    "last_answered_at": ts,
                    "current_revision": 0,
                    "revisions": [],
                }

            next_rev = int(entry.get("current_revision", 0)) + 1
            entry["revisions"].append({
                "rev": next_rev,
                "timestamp": ts,
                "user_text": user_text,
                "stt_ms": round(float(stt_ms), 1),
                "usage": usage_dict,
            })
            entry["current_revision"] = next_rev
            entry["last_answered_at"] = ts
            # Keep question_text in sync (script may have been edited between sessions)
            entry["question_text"] = self.questions[index]
            amap[str(index)] = entry
            data["answers"] = amap

            self.store.save_phase1_answers(pid, data)

            # Mirror the final-revision snapshot into conversation_log.json
            # so Phase 2 (hci/memory/build.py) keeps its current contract.
            self._rewrite_conversation_log(pid, data)

            # status.json cumulative total
            self.store.update_phase1_usage_total(
                pid, usage_dict,
                count_revision=True,
                count_new_turn=is_new_turn,
            )

        return AnswerRecord(
            index=index,
            user_text=user_text,
            revision=next_rev,
            total_revisions=next_rev,
            answered=True,
        )

    def _rewrite_conversation_log(self, pid: str, answers_doc: dict[str, Any]) -> None:
        """Overwrite conversation_log.json with one entry per answered index
        (sorted by index), using the latest revision. Unanswered indices omitted.
        """
        log_path = self.store.log_path(pid, "phase1")
        amap = answers_doc.get("answers") or {}
        turns: list[dict[str, Any]] = []
        for i in range(self.total):
            entry = amap.get(str(i))
            if not entry or int(entry.get("current_revision", 0)) < 1:
                continue
            latest = entry["revisions"][-1]
            turns.append({
                "turn": len(turns) + 1,
                "timestamp": latest.get("timestamp"),
                "user_text": latest.get("user_text", ""),
                "agent_text": entry.get("question_text", self.questions[i]),
                "latency": {
                    "stt_ms": float(latest.get("stt_ms", 0.0)),
                    "llm_ms": 0.0,
                    "tts_ms": 0.0,
                    "total_ms": float(latest.get("stt_ms", 0.0)),
                },
                "usage": dict(latest.get("usage") or {}),
            })
        with log_path.open("w", encoding="utf-8") as f:
            json.dump(turns, f, ensure_ascii=False, indent=2)

    def finish(self, pid: str) -> dict[str, Any]:
        """Finalize Phase 1. Raises :class:`Phase1NotComplete` if any answer is missing.

        Returns a dict suitable for the ``/phase1/finish`` response body.
        """
        with self._lock:
            data = self._load_answers(pid)
            amap = data.get("answers") or {}
            unanswered = [i for i in range(self.total) if not self._is_answered(amap, i)]
            answered_count = self.total - len(unanswered)
            if unanswered:
                raise Phase1NotComplete(unanswered, answered_count, self.total)

            now = datetime.now().isoformat()
            data["ended_at"] = now
            self.store.save_phase1_answers(pid, data)

            status = self.store.get_status(pid)
            status["phase1_done"] = True
            status["phase1_ended_at"] = now
            self.store.save_status(pid, status)

            return {
                "status": "finished",
                "participant_id": pid,
                "answered_count": self.total,
                "total": self.total,
                "ended_at": now,
                "usage_total": status.get("phase1_usage_total") or {},
            }
