# -*- coding:utf-8 -*-
"""Participant profile extraction helper.

Shared by all three memory modules (in_context / retrieval / parametric).
Extracts the user's name from the first turn of Phase 1 conversation_log.json
(the name question) so it can be injected directly into the Phase 3 system prompt.

Workaround for LoRA memorization failure (proper-noun prior issue): the name is
handled via prompt injection rather than training.
"""
from __future__ import annotations

import json
import re
from pathlib import Path


# Strip ambient-noise tags mixed into the STT output, e.g. "(mouse click sound)".
_STT_NOISE_RE = re.compile(r"\([^)]*\)")

# Strip polite Korean sentence endings from the answer — e.g. "홍길동입니다." (it's Hong Gildong.) -> "홍길동"
_POLITE_ENDINGS = (
    "입니다.", "입니다",
    "이에요.", "이에요",
    "예요.", "예요",
    "이야.", "이야",
    "라고 합니다.", "라고 합니다",
    "이라고 합니다.", "이라고 합니다",
)


def load_user_display_name(base_dir: Path | str, pid: str) -> str | None:
    """Extract the participant's real name from the first Phase 1 turn (name question).

    Returns:
        The name string (e.g. "홍길동"). None if extraction fails.
    """
    base = Path(base_dir)
    log_path = base / "participants" / pid / "phase1" / "conversation_log.json"
    if not log_path.exists():
        return None

    try:
        with log_path.open(encoding="utf-8") as f:
            log = json.load(f)
    except Exception:
        return None

    for turn in log:
        agent_q = turn.get("agent_text", "") or ""
        if "성함" not in agent_q:
            continue
        raw = turn.get("user_text", "") or ""
        cleaned = _STT_NOISE_RE.sub("", raw).strip()
        # Strip polite endings (prefer the longest match, scanning from the end)
        for ending in sorted(_POLITE_ENDINGS, key=len, reverse=True):
            if cleaned.endswith(ending):
                cleaned = cleaned[: -len(ending)].strip()
                break
        # Clean up any punctuation left at the end
        cleaned = cleaned.rstrip(" .?!")
        return cleaned or None

    return None
