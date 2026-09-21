# -*- coding:utf-8 -*-
"""
in_context - sliding-window-based Full Context chatbot module
- In initialize_user, injects the Phase 2 build artifact (memory_data/in_context/<pid>_memory.json)
  into self.history within the token budget — the core of this condition.
- Phase 3 conversations are NOT saved to this file (avoids contamination). Session logs are handled by SessionStore.
- Token counts are computed in advance with the llama tokenizer before building messages;
  oldest turns are dropped when the budget is exceeded
"""
import os
import json
from llama_cpp import Llama
from agents.memory_base import MemoryBase
from agents._profile import load_user_display_name

# Path setup
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)
MEMORY_DIR = os.path.join(BASE_DIR, 'memory_data', 'in_context')

# Context budget: system + history may use up to this ratio; the rest is reserved for the response
CONTEXT_BUDGET_RATIO = 0.80
MIN_RESPONSE_TOKENS = 512


class in_context(MemoryBase):
    def __init__(self, model_path, temperature=0.7):
        self.model_path = model_path
        self.temperature = temperature
        self.max_tokens = 16384  # 12.5% of Gemma-3's training n_ctx 131072 (targeting 16GB VRAM)

        print(f"[System] Loading model (Full Context / cache OFF): {model_path}")
        self.llm = Llama(
            model_path=self.model_path,
            n_ctx=self.max_tokens,
            n_gpu_layers=-1,
            n_batch=512,
            verbose=False,
            cache_prompt=False
        )

        os.makedirs(MEMORY_DIR, exist_ok=True)
        self.user_name = None
        self.user_display_name = None  # Extracted from the Phase 1 name answer (e.g. "홍길동")
        self.user_memory = {}
        self.history = []        # Each item: [query, response]
        self.last_prompt_tokens = 0
        self.last_completion_tokens = 0
        self.last_total_tokens = 0

    # ── Token counting ──────────────────────────────────────────

    def _count_tokens(self, text: str) -> int:
        """Compute the exact token count with the llama tokenizer."""
        return len(self.llm.tokenize(text.encode("utf-8")))

    def _count_message_tokens(self, messages: list) -> int:
        """Compute the total token count of a message list.
        Approximates the chat-template overhead (role tags, etc.) as +4 tokens per message."""
        total = 0
        for msg in messages:
            total += self._count_tokens(msg["content"]) + 4
        return total

    # ── User management ────────────────────────────────────────

    def initialize_user(self, user_name: str) -> str:
        self.user_name = user_name
        self.user_display_name = load_user_display_name(BASE_DIR, user_name)
        self.user_memory = self._load_user_memory(user_name)

        # Collect all past session conversations in date/time order
        all_turns = []
        for date in sorted(self.user_memory.get("history", {}).keys()):
            for turn in self.user_memory["history"][date]:
                all_turns.append((turn["query"], turn["response"]))

        # Load within the token budget, going backward from the most recent turn
        # Budget: _build_messages also needs room for system + current input,
        # so only 60% of the total budget is allocated to past history
        history_budget = int(self.max_tokens * 0.60)
        loaded_turns = []
        accumulated = 0
        for q, r in reversed(all_turns):
            pair = [
                {"role": "user", "content": q},
                {"role": "assistant", "content": r},
            ]
            cost = self._count_message_tokens(pair)
            if accumulated + cost > history_budget:
                break
            loaded_turns.insert(0, [q, r])
            accumulated += cost

        self.history = loaded_turns

        total = len(all_turns)
        loaded = len(loaded_turns)
        return (
            f"{user_name} connected "
            f"({loaded}/{total} recent turns loaded into context / "
            f"tokens: {accumulated}/{history_budget})"
        )

    def chat(self, text: str) -> str:
        if not self.user_name:
            raise ValueError("User initialization required")

        messages = self._build_messages(text)

        response = self.llm.create_chat_completion(
            messages=messages,
            temperature=self.temperature,
            top_k=40,
            top_p=0.9,
            repeat_penalty=1.1,  # llama.cpp default — minimizes disruption to natural conversation (shared across all three conditions)
            max_tokens=MIN_RESPONSE_TOKENS,
            stream=False,
        )

        response_text = response['choices'][0]['message']['content']

        # Usage accounting (parity with retrieval / parametric)
        self.last_prompt_tokens = response['usage']['prompt_tokens']
        self.last_completion_tokens = response['usage']['completion_tokens']
        self.last_total_tokens = response['usage']['total_tokens']

        self.history.append([text, response_text])
        return response_text

    def _build_messages(self, text: str) -> list:
        """Build the message list within the token budget.
        First reserve the system prompt and the current input,
        then include as many turns as the budget allows, going backward from the most recent."""
        # Use self.user_display_name, extracted from the Phase 1 name answer, as the participant's real name.
        # self.user_name (pid, e.g. P001) must NOT be exposed.
        name_intro = ""
        if self.user_display_name:
            name_intro = (
                f"사용자의 이름은 '{self.user_display_name}' 입니다. "
                f"친근하고 자연스럽게 대화해주세요.\n"
            )
        # Korean by design — experiment stimulus
        system_prompt = (
            name_intro
            + "당신은 사용자와 음성으로 대화하는 친근한 친구입니다. "
            + "이 대화는 음성으로 진행되며 당신의 답변은 그대로 TTS 로 읽힙니다.\n"
            + "- 이모지/이모티콘/특수기호 절대 금지 (TTS 가 이름을 읽음).\n"
            + "- 마크다운(**, ##, 목록, 코드블록) 사용 금지.\n"
            + "- 1~3문장으로 짧고 자연스럽게 답하세요.\n"
            + "- 해요체(예: '그랬어요', '재밌겠네요')로 말하세요. 격식체 '합니다' 도, 완전한 반말('~했어') 도 쓰지 마세요.\n"
            + "- 대화가 자연스럽게 이어지도록 답해주세요. 매번 질문으로 끝낼 필요는 없어요."
        )

        system_msg = {"role": "system", "content": system_prompt}
        current_msg = {"role": "user", "content": text}

        budget = int(self.max_tokens * CONTEXT_BUDGET_RATIO) - MIN_RESPONSE_TOKENS
        fixed_cost = self._count_message_tokens([system_msg, current_msg])
        remaining = budget - fixed_cost

        if remaining < 0:
            print(f"[in_context] Current input exceeds token budget (fixed={fixed_cost}, budget={budget})")
            return [system_msg, current_msg]

        # Include turns within budget, going backward from the most recent
        included_turns = []
        for q, r, *_ in reversed(self.history):
            pair = [
                {"role": "user", "content": q},
                {"role": "assistant", "content": r},
            ]
            pair_cost = self._count_message_tokens(pair)
            if pair_cost > remaining:
                break
            included_turns.insert(0, pair)
            remaining -= pair_cost

        messages = [system_msg]
        for pair in included_turns:
            messages.extend(pair)
        messages.append(current_msg)
        return messages

    # ── Helper functions ──────────────────────────────────────────

    def _load_user_memory(self, user_name: str) -> dict:
        path = os.path.join(MEMORY_DIR, f'{user_name}_memory.json')
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {"name": user_name, "history": {}}
