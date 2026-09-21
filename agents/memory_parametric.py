# -*- coding:utf-8 -*-
"""
parametric - parametric long-term memory module (Full Context + personalized LoRA)
- Hot-swaps a per-user LoRA (Personalized Persona) adapter into the llama-cpp instance
- After a session ends, automatically runs LoRA fine-tuning in the background based on the conversation log
- Inference parameters are identical to in_context / retrieval (temperature=0.7, default repeat_penalty)
- Phase 1 data is NOT included in build_messages (starts with no history)
- Token counts are computed in advance with the llama tokenizer before building messages;
  oldest turns are dropped when the budget is exceeded
"""
import os
from llama_cpp import Llama
from agents.memory_base import MemoryBase
from agents._profile import load_user_display_name

# Path setup
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)
WEIGHTS_DIR = os.path.join(BASE_DIR, 'model_weights')

# Context budget: system + history may use up to this ratio; the rest is reserved for the response
CONTEXT_BUDGET_RATIO = 0.80
# Minimum number of tokens to reserve for response generation
MIN_RESPONSE_TOKENS = 512


class parametric(MemoryBase):
    def __init__(self, model_path, temperature=0.7):
        self.model_path = model_path
        self.temperature = temperature
        self.max_tokens = 16384  # 12.5% of Gemma-3's training n_ctx 131072 (targeting 16GB VRAM)

        self.llm = Llama(
            model_path=self.model_path,
            n_ctx=self.max_tokens,
            n_gpu_layers=-1,
            verbose=False
        )

        self.user_name = None
        self.user_display_name = None  # Extracted from the Phase 1 name answer (e.g. "홍길동")
        # self.history is the list of (query, response) within the current session. Discarded when the server process exits.
        # Phase 3 conversations are NOT saved to the participant's memory.json — avoids contamination.
        self.history = []
        # usage tracking (populated per chat call)
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
        # Phase 3 conversations are NOT saved to memory_data/parametric/<pid>_memory.json.
        # Session conversation logs are recorded separately by SessionStore at participants/<pid>/phase3/parametric_log.json.
        # <pid>_memory.json is a read-only artifact of Phase 2 (LoRA training).
        self.user_name = user_name
        self.user_display_name = load_user_display_name(BASE_DIR, user_name)
        self.history = []  # Volatile session history

        lora_path = os.path.join(WEIGHTS_DIR, f"{user_name}_lora_persona.gguf")
        lora_status_msg = ""

        if os.path.exists(lora_path) and os.path.isfile(lora_path):
            try:
                del self.llm  # Release the existing instance before reloading (saves VRAM)
                self.llm = Llama(
                    model_path=self.model_path,
                    lora_base=self.model_path,
                    lora_path=lora_path,
                    n_ctx=self.max_tokens,
                    n_gpu_layers=-1,
                    verbose=False
                )
                lora_status_msg = f"\n({user_name}'s latest adapter applied)"
            except Exception as e:
                print(f"LoRA load failed: {e}")

        return f"{user_name} connected" + lora_status_msg

    # ── Chat ───────────────────────────────────────────────

    def chat(self, text: str) -> str:
        if not self.user_name:
            raise ValueError("User not initialized. Call initialize_user() first.")

        if not text.strip():
            return ""

        messages = self._build_messages(text)
        response_text = self._call_llm(messages)

        self.history.append([text, response_text])
        return response_text

    def _build_messages(self, text: str) -> list:
        """Build the message list within the token budget.
        First reserve the system prompt and the current input,
        then include as many turns as the budget allows, going backward from the most recent."""
        # Use self.user_display_name, extracted from the Phase 1 name answer, as the participant's real name.
        # self.user_name (pid, e.g. P001) must NOT be exposed — the LLM must not learn it as a form of address.
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

        # Fixed cost: system + current input
        budget = int(self.max_tokens * CONTEXT_BUDGET_RATIO) - MIN_RESPONSE_TOKENS
        fixed_cost = self._count_message_tokens([system_msg, current_msg])
        remaining = budget - fixed_cost

        if remaining < 0:
            # system + current input alone already exceeds the budget → send without history
            print(f"[parametric] Current input exceeds token budget (fixed={fixed_cost}, budget={budget})")
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

        # Assemble messages
        messages = [system_msg]
        for pair in included_turns:
            messages.extend(pair)
        messages.append(current_msg)
        return messages

    def _call_llm(self, messages: list) -> str:
        try:
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
            self.last_prompt_tokens = response['usage']['prompt_tokens']
            self.last_completion_tokens = response['usage']['completion_tokens']
            self.last_total_tokens = response['usage']['total_tokens']
            return response_text
        except Exception as e:
            print(f"\nPersonalized LLM call error: {e}")
            return ""
