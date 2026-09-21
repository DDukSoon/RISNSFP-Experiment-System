# -*- coding: utf-8 -*-
"""WAV container helpers (pure stdlib)."""
from __future__ import annotations

import io
import wave


def pcm_to_wav(pcm_bytes: bytes, sample_rate: int = 16000) -> bytes:
    """Wrap raw 16-bit mono PCM bytes in a WAV container.

    Args:
        pcm_bytes: Little-endian signed 16-bit mono PCM samples.
        sample_rate: Sampling rate in Hz (default 16000).

    Returns:
        Complete WAV file as bytes.
    """
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_bytes)
    return buf.getvalue()
