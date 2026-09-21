# -*- coding: utf-8 -*-
"""ElevenLabs TTS with on-disk cache and prewarm.

Cache layout is identical to the legacy implementation — the key is
``sha256(f"{voice_id}|{model}|{output_format}|{text}")`` stored under
``voice/voice_cache/<voice_id>/<sha>.wav`` — so previously generated files still hit.
"""
from __future__ import annotations

import hashlib
import os
import time

from dotenv import load_dotenv

from voice.wav import pcm_to_wav

load_dotenv()

_VOICE_DIR = os.path.dirname(os.path.abspath(__file__))
_CACHE_ROOT = os.path.join(_VOICE_DIR, "voice_cache")


class ElevenLabsTTS:
    """ElevenLabs TTS returning WAV 16-bit PCM 16kHz mono."""

    def __init__(self) -> None:
        from elevenlabs.client import ElevenLabs  # lazy import

        api_key = os.getenv("ELEVENLABS_API_KEY")
        if not api_key:
            raise ValueError("ELEVENLABS_API_KEY is not set in .env")
        self.client = ElevenLabs(api_key=api_key)
        self.voice_id = os.getenv("ELEVENLABS_VOICE_ID", "21m00Tcm4TlvDq8ikWAM")
        self.model = os.getenv("ELEVENLABS_MODEL", "eleven_multilingual_v2")
        self.output_format = "pcm_16000"
        self.sample_rate = 16000
        print(
            f"[TTS] ElevenLabs initialized "
            f"(voice_id: {self.voice_id}, format: WAV PCM16 16k)"
        )

    # ── Cache ─────────────────────────────────────────────────────
    def _cache_path(self, text: str) -> str:
        key = f"{self.voice_id}|{self.model}|{self.output_format}|{text}"
        h = hashlib.sha256(key.encode("utf-8")).hexdigest()
        folder = os.path.join(_CACHE_ROOT, self.voice_id)
        os.makedirs(folder, exist_ok=True)
        return os.path.join(folder, f"{h}.wav")

    def get_cache_path(self, text: str) -> str:
        """Public alias for ``_cache_path``. Used by Phase 1 audio route."""
        return self._cache_path(text)

    def is_cached(self, text: str) -> bool:
        """Return True if ``text`` has a pre-synthesized WAV on disk."""
        return os.path.isfile(self._cache_path(text))

    def _load_cached(self, text: str) -> bytes | None:
        path = self._cache_path(text)
        if os.path.isfile(path):
            with open(path, "rb") as f:
                return f.read()
        return None

    def _save_cached(self, text: str, wav_bytes: bytes) -> None:
        with open(self._cache_path(text), "wb") as f:
            f.write(wav_bytes)

    def synthesize(self, text: str, *, cache_write: bool = True) -> tuple[bytes, float]:
        """Text -> (wav_bytes, elapsed_ms). Disk-cached per voice_id.

        Calling with ``cache_write=False`` skips writing the synthesized result
        to disk (reads are still always attempted). This switch avoids wasting
        disk on paths where re-hits are essentially impossible because the LLM
        response differs every time, as in Phase 3 free-form conversation.
        """
        t0 = time.time()
        cached = self._load_cached(text)
        if cached is not None:
            return cached, (time.time() - t0) * 1000

        try:
            audio_iter = self.client.text_to_speech.convert(
                voice_id=self.voice_id,
                model_id=self.model,
                text=text,
                output_format=self.output_format,
            )
        except AttributeError:
            audio_iter = self.client.generate(
                text=text,
                voice=self.voice_id,
                model=self.model,
                output_format=self.output_format,
            )
        pcm_bytes = b"".join(audio_iter)
        wav_bytes = pcm_to_wav(pcm_bytes, sample_rate=self.sample_rate)
        if cache_write:
            self._save_cached(text, wav_bytes)
        return wav_bytes, (time.time() - t0) * 1000

    def prewarm(self, texts: list[str]) -> None:
        """Ensure every text in ``texts`` has a cached WAV on disk."""
        missing = [t for t in texts if self._load_cached(t) is None]
        if not missing:
            return
        for i, t in enumerate(missing, 1):
            self.synthesize(t)
        print("[TTS] prewarm done")
