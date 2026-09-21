# -*- coding: utf-8 -*-
"""Phase 3 agents — thin wrappers over the three memory backends.

Each class delegates to the corresponding ``agents.memory_*`` backend
(in_context / retrieval / parametric) and adapts the backend's
``chat()`` return into the :class:`agents.base.Agent` 3-tuple contract
(response_text, llm_elapsed_ms, usage_dict).
"""
from __future__ import annotations

import gc
import time
from typing import Any

from agents.base import Agent


class _Phase3AgentBase(Agent):
    """Shared wrapping over ``agents.memory_*`` backends."""

    label: str = ""  # subclass-provided log tag

    def __init__(self, participant_id: str, backend: Any) -> None:
        self.participant_id = participant_id
        self.agent = backend
        init_msg = self.agent.initialize_user(participant_id)
        print(f"[Phase3/{self.label}] {init_msg}")

    def chat(self, user_text: str) -> tuple[str, float, dict]:
        t0 = time.time()
        response = self.agent.chat(user_text)
        usage = {
            "llm_prompt_tokens": int(getattr(self.agent, "last_prompt_tokens", 0) or 0),
            "llm_completion_tokens": int(getattr(self.agent, "last_completion_tokens", 0) or 0),
            "llm_total_tokens": int(getattr(self.agent, "last_total_tokens", 0) or 0),
            "llm_tokens_estimated": False,
        }
        return response, (time.time() - t0) * 1000, usage

    def warmup(self) -> None:
        """Delegate to the backend's `warmup()` if it provides one — removes first-utterance latency.

        Currently only the retrieval backend implements it (avoids Ollama embedding lazy load).
        in_context / parametric are no-ops.
        """
        if self.agent is not None and hasattr(self.agent, "warmup"):
            try:
                self.agent.warmup()
            except Exception as e:
                print(f"[Phase3/{self.label}] Warmup failed (continuing): {e}")

    def cleanup(self) -> None:
        if self.agent is not None and hasattr(self.agent, "finalize_session"):
            try:
                self.agent.finalize_session()
            except Exception:
                pass
        self.agent = None
        gc.collect()


class Phase3InContextAgent(_Phase3AgentBase):
    label = "in_context"

    def __init__(self, participant_id: str, model_path: str) -> None:
        from agents.memory_incontext import in_context

        super().__init__(
            participant_id,
            in_context(model_path=model_path, temperature=0.7),
        )


class Phase3RetrievalAgent(_Phase3AgentBase):
    label = "retrieval"

    def __init__(self, participant_id: str, model_path: str) -> None:
        from agents.memory_retrieval import retrieval

        super().__init__(
            participant_id,
            retrieval(
                model_path=model_path,
                embed_model_name="embeddinggemma",
                temperature=0.7,
            ),
        )


class Phase3ParametricAgent(_Phase3AgentBase):
    label = "parametric"

    def __init__(self, participant_id: str, model_path: str) -> None:
        from agents.memory_parametric import parametric

        super().__init__(
            participant_id,
            parametric(model_path=model_path, temperature=0.7),
        )
