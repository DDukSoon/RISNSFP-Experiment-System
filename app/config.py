# -*- coding: utf-8 -*-
"""Runtime settings: YAML defaults merged with environment overrides."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent

# Load .env immediately at this module's import time, so that ELEVENLABS_API_KEY
# is visible even when require_elevenlabs_key() / Settings.load() run before the
# voice submodule.
# Already-defined environment variables are not overwritten (override=False is the default).
load_dotenv(BASE_DIR / ".env")


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


@dataclass
class Settings:
    """All tunables for the voice-experiment server.

    Env vars take precedence over ``config/default.yaml``.
    """

    base_dir: Path = BASE_DIR
    model_path: Path = field(
        default_factory=lambda: BASE_DIR / "model_weights" / "gemma-3-4b-it-q4_0.gguf"
    )
    phase1_shutdown_delay_sec: float = 8.0
    phase1_script_path: Path = field(
        default_factory=lambda: BASE_DIR / "config" / "phase1_script.yaml"
    )
    host: str = "0.0.0.0"
    port: int = 8000

    @classmethod
    def load(cls) -> "Settings":
        """Build a ``Settings`` by merging YAML defaults with environment overrides."""
        cfg = _load_yaml(BASE_DIR / "config" / "default.yaml")

        phase1 = cfg.get("phase1", {}) or {}
        model = cfg.get("model", {}) or {}
        server = cfg.get("server", {}) or {}

        model_rel = model.get("path", "model_weights/gemma-3-4b-it-q4_0.gguf")
        script_rel = phase1.get("script_file", "config/phase1_script.yaml")

        return cls(
            base_dir=BASE_DIR,
            model_path=BASE_DIR / model_rel,
            phase1_shutdown_delay_sec=float(
                os.getenv(
                    "PHASE1_SHUTDOWN_DELAY_SEC",
                    phase1.get("shutdown_delay_sec", 8),
                )
            ),
            phase1_script_path=BASE_DIR / script_rel,
            host=os.getenv("HCI_HOST", server.get("host", "0.0.0.0")),
            port=int(os.getenv("HCI_PORT", server.get("port", 8000))),
        )


def require_elevenlabs_key() -> None:
    """Fail fast (exit code 2) if the ElevenLabs API key is missing."""
    if not os.getenv("ELEVENLABS_API_KEY"):
        print(
            "[server] ERROR: ELEVENLABS_API_KEY environment variable is not set. "
            "Add it to your .env file or export it in the shell, then run again."
        )
        raise SystemExit(2)
