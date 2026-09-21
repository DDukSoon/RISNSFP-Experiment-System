# -*- coding: utf-8 -*-
"""TTS input sanitizer — strips emoji and basic markdown noise.

The regex ranges are IDENTICAL to the legacy ``_EMOJI_RE`` in the old
``server.py`` so behavior is byte-for-byte preserved.
"""
from __future__ import annotations

import re

# Strip emoji + pictographs + dingbats so TTS doesn't read symbol names aloud.
_EMOJI_RE = re.compile(
    "["
    "\U0001F300-\U0001FAFF"  # symbols & pictographs (incl. emoticons block range)
    "\U0001F600-\U0001F64F"  # emoticons
    "\U0001F680-\U0001F6FF"  # transport & map
    "\U0001F900-\U0001F9FF"  # supplemental symbols & pictographs
    "\U00002600-\U000027BF"  # misc symbols & dingbats
    "\U0000FE00-\U0000FE0F"  # variation selectors
    "\U0001F1E6-\U0001F1FF"  # regional indicators (flags)
    "]+",
    flags=re.UNICODE,
)


def clean_for_tts(text: str) -> str:
    """Strip emoji and basic markdown so the TTS speaks plain prose only."""
    if not text:
        return text
    t = _EMOJI_RE.sub("", text)
    # markdown noise: **bold**, *italic*, `code`, headings, list bullets
    t = re.sub(r"[*_`#>]+", "", t)
    t = re.sub(r"^\s*[-•]\s+", "", t, flags=re.MULTILINE)
    # collapse runs of whitespace introduced by stripping
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()
