# -*- coding: utf-8 -*-
"""Agent abstract base class."""
from __future__ import annotations

from abc import ABC, abstractmethod


class Agent(ABC):
    """Common interface for every phase agent."""

    @abstractmethod
    def chat(self, user_text: str) -> tuple[str, float, dict]:
        """Produce a reply to ``user_text``.

        Returns:
            (response_text, llm_elapsed_ms, usage_dict)

            ``usage_dict`` keys (all optional, default 0):
              - ``llm_prompt_tokens`` (int)
              - ``llm_completion_tokens`` (int)
              - ``llm_total_tokens`` (int)
              - ``llm_tokens_estimated`` (bool): False if measured directly
                from the LLM usage payload; True for legacy/streaming paths.

            Phase 1 agents return zeros + ``llm_tokens_estimated=False``
            since no LLM inference occurs.
        """

    def preloadable_lines(self) -> list[str]:
        """Lines worth pre-synthesizing on startup (TTS prewarm)."""
        return []

    def cleanup(self) -> None:
        """Release any heavy resources held by the agent."""
        return None
