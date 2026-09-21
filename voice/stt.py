# -*- coding: utf-8 -*-
"""ElevenLabs Scribe speech-to-text wrapper."""
from __future__ import annotations

import io
import os
import time

from dotenv import load_dotenv

load_dotenv()


class ElevenLabsSTT:
    """ElevenLabs Scribe STT with the same (text, elapsed_ms) contract as the legacy class."""

    def __init__(self, language: str = "kor") -> None:
        from elevenlabs.client import ElevenLabs  # lazy import

        api_key = os.getenv("ELEVENLABS_API_KEY")
        if not api_key:
            raise ValueError("ELEVENLABS_API_KEY is not set in .env")
        self.client = ElevenLabs(api_key=api_key)
        self.model_id = os.getenv("ELEVENLABS_STT_MODEL", "scribe_v1")
        self.language = language
        print(
            f"[STT] ElevenLabs Scribe initialized "
            f"(model={self.model_id}, lang={self.language or 'auto'})"
        )

    def transcribe(self, audio_bytes: bytes) -> tuple[str, float]:
        """Transcribe audio bytes -> (text, elapsed_ms).

        We set tag_audio_events=False explicitly: ElevenLabs Scribe defaults to
        True, which inserts non-speech sounds ("mouse click", "camera shutter",
        etc.) into user_text and pollutes the LLM prompt / logs / profile
        extraction. Force speech-only recognition.
        """
        t0 = time.time()
        if not audio_bytes:
            return "", 0.0

        f = io.BytesIO(audio_bytes)
        f.name = "audio.wav"

        kwargs = {
            "file": f,
            "model_id": self.model_id,
            "tag_audio_events": False,
        }
        if self.language:
            kwargs["language_code"] = self.language

        try:
            result = self.client.speech_to_text.convert(**kwargs)
        except TypeError:
            # Legacy SDK compatibility: language_code or tag_audio_events unsupported
            kwargs.pop("language_code", None)
            kwargs.pop("tag_audio_events", None)
            f.seek(0)
            result = self.client.speech_to_text.convert(**kwargs)
        except Exception as e:
            print(f"[STT] ElevenLabs convert failed: {e}")
            return "", (time.time() - t0) * 1000

        text = getattr(result, "text", None)
        if text is None and isinstance(result, dict):
            text = result.get("text", "")
        text = (text or "").strip()
        return text, (time.time() - t0) * 1000
