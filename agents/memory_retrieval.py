# -*- coding:utf-8 -*-
"""
retrieval - short-term session + RAG long-term memory chatbot module
- Retrieves memories similar to the current question from a FAISS vector DB
  (Phase 2 build artifact) and injects them into the prompt
- Ollama: embedding extraction (uses the embeddinggemma model)
- Llama-cpp: chat generation (uses Gemma-3 GGUF)
- Phase 3 conversations are NOT saved to memory.json / FAISS (avoids contamination).
  Session logs are recorded separately by SessionStore at participants/<pid>/phase3/retrieval_log.json.
- The build_memory subprocess pushes directly into the session_turns buffer and then calls
  finalize_session() to commit the Phase 1 + noise turns into the FAISS index (this path is kept).
- Token counts are computed in advance with the llama tokenizer before building messages;
  oldest turns are dropped when the budget is exceeded
"""
import os
import json
import time
import numpy as np
import ollama
from llama_cpp import Llama
from dotenv import load_dotenv
from agents.memory_base import MemoryBase
from agents._profile import load_user_display_name

# FAISS library check
try:
    import faiss
except ImportError:
    print("FAISS library not found. Please run 'pip install faiss-cpu'.")
    faiss = None

load_dotenv()

# Path setup
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)
MEMORY_DIR = os.path.join(BASE_DIR, 'memory_data', 'retrieval')

# Context budget
CONTEXT_BUDGET_RATIO = 0.80
MIN_RESPONSE_TOKENS = 512


class retrieval(MemoryBase):
    def __init__(self, model_path, embed_model_name='embeddinggemma', temperature=0.7, top_k=3):
        self.model_path = model_path
        self.embed_model = embed_model_name
        self.temperature = temperature
        self.top_k = top_k
        self.max_tokens = 16384  # 12.5% of Gemma-3's training n_ctx 131072 (targeting 16GB VRAM)

        print(f"Loading Chat LLM: {self.model_path}")
        self.chat_llm = Llama(
            model_path=self.model_path,
            n_ctx=self.max_tokens,
            n_gpu_layers=-1,
            verbose=False
        )

        os.makedirs(MEMORY_DIR, exist_ok=True)
        self.user_index_dir = os.path.join(MEMORY_DIR, 'index')
        self.user_mapping_dir = os.path.join(MEMORY_DIR, 'mapping')
        os.makedirs(self.user_index_dir, exist_ok=True)
        os.makedirs(self.user_mapping_dir, exist_ok=True)

        self.user_name = None
        self.user_display_name = None  # Extracted from the Phase 1 name answer (e.g. "홍길동")
        self.user_memory = {}
        self.history = []
        self.session_turns = []  # Buffer of un-embedded turns accumulated during the session: (query, response, date, ts)
        self.index = None
        self.doc_mapping = {}
        self.embedding_dim = None
        self.user_index_path = None
        self.user_mapping_path = None
        # usage tracking (populated per chat call)
        self.last_prompt_tokens = 0
        self.last_completion_tokens = 0
        self.last_total_tokens = 0

    # ── Token counting ──────────────────────────────────────────

    def _count_tokens(self, text: str) -> int:
        """Compute the exact token count with the llama tokenizer."""
        return len(self.chat_llm.tokenize(text.encode("utf-8")))

    def _count_message_tokens(self, messages: list) -> int:
        """Compute the total token count of a message list.
        Approximates the chat-template overhead (role tags, etc.) as +4 tokens per message."""
        total = 0
        for msg in messages:
            total += self._count_tokens(msg["content"]) + 4
        return total

    # ================= [Embedding and FAISS core logic] =================

    def _get_embedding(self, text: str) -> np.ndarray:
        try:
            response = ollama.embeddings(model=self.embed_model, prompt=text)
            return np.array(response['embedding'], dtype=np.float32)
        except Exception as e:
            print(f"Embedding extraction error: {e}")
            return None

    def _init_faiss(self):
        self.user_index_path = os.path.join(self.user_index_dir, f"{self.user_name}_faiss.index")
        self.user_mapping_path = os.path.join(self.user_mapping_dir, f"{self.user_name}_mapping.json")

        if os.path.exists(self.user_index_path) and os.path.exists(self.user_mapping_path):
            self.index = faiss.read_index(self.user_index_path)
            with open(self.user_mapping_path, 'r', encoding='utf-8') as f:
                self.doc_mapping = {int(k): v for k, v in json.load(f).items()}
            self.embedding_dim = self.index.d
        else:
            self.index = None
            self.doc_mapping = {}

    def _search_faiss(self, text: str) -> list:
        if self.index is None or self.index.ntotal == 0:
            return []

        query_embed = self._get_embedding(text)
        if query_embed is None:
            return []

        k = min(self.top_k, self.index.ntotal)
        distances, indices = self.index.search(np.array([query_embed]), k)

        return [self.doc_mapping[idx] for idx in indices[0] if idx != -1 and idx in self.doc_mapping]

    def _add_to_faiss(self, query: str, response: str, date: str, timestamp: str):
        doc_text = f"[{date} {timestamp}] User: {query} / AI: {response}"
        embed = self._get_embedding(doc_text)

        if embed is None:
            return

        if self.index is None:
            self.embedding_dim = embed.shape[0]
            self.index = faiss.IndexFlatL2(self.embedding_dim)

        next_id = self.index.ntotal
        self.index.add(np.array([embed]))
        self.doc_mapping[next_id] = doc_text

        faiss.write_index(self.index, self.user_index_path)
        with open(self.user_mapping_path, 'w', encoding='utf-8') as f:
            json.dump(self.doc_mapping, f, ensure_ascii=False, indent=2)

    # ================= [Conversation flow control] =================

    def initialize_user(self, user_name: str) -> str:
        self.user_name = user_name
        self.user_display_name = load_user_display_name(BASE_DIR, user_name)
        self.history = []  # Does not include Phase 1 data — start with an empty history
        self.session_turns = []
        self.user_memory = self._load_user_memory(user_name)
        self._init_faiss()

        total = sum(len(c) for c in self.user_memory.get("history", {}).values())
        return f"{user_name} session active ({total} past records loaded into RAG index)"

    def chat(self, text: str) -> str:
        if not self.user_name:
            raise ValueError("User initialization required.")

        retrieved_contexts = self._search_faiss(text)
        messages = self._build_messages(text, retrieved_contexts)
        response_text = self._call_llm(messages)

        self.history.append([text, response_text])
        return response_text

    def _build_messages(self, text: str, contexts: list) -> list:
        """Build the message list within the token budget.
        First reserve the system prompt (+ RAG context) and the current input,
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
        system_content = (
            name_intro
            + "당신은 사용자와 음성으로 대화하는 친근한 친구입니다. "
            + "이 대화는 음성으로 진행되며 당신의 답변은 그대로 TTS 로 읽힙니다.\n"
            + "- 이모지/이모티콘/특수기호 절대 금지 (TTS 가 이름을 읽음).\n"
            + "- 마크다운(**, ##, 목록, 코드블록) 사용 금지.\n"
            + "- 1~3문장으로 짧고 자연스럽게 답하세요.\n"
            + "- 해요체(예: '그랬어요', '재밌겠네요')로 말하세요. 격식체 '합니다' 도, 완전한 반말('~했어') 도 쓰지 마세요.\n"
            + "- 대화가 자연스럽게 이어지도록 답해주세요. 매번 질문으로 끝낼 필요는 없어요."
        )

        if contexts:
            context_str = "\n".join(contexts)
            system_content += f"\n\n[참고할 과거 기억]\n{context_str}\n\n위 내용을 바탕으로 사용자를 기억하며 대화해줘."

        system_msg = {"role": "system", "content": system_content}
        current_msg = {"role": "user", "content": text}

        budget = int(self.max_tokens * CONTEXT_BUDGET_RATIO) - MIN_RESPONSE_TOKENS
        fixed_cost = self._count_message_tokens([system_msg, current_msg])
        remaining = budget - fixed_cost

        if remaining < 0:
            print(f"[retrieval] Current input exceeds token budget (fixed={fixed_cost}, budget={budget})")
            return [system_msg, current_msg]

        # Include turns within budget, going backward from the most recent
        included_turns = []
        for q, r in reversed(self.history):
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

    def _call_llm(self, messages: list) -> str:
        response = self.chat_llm.create_chat_completion(
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

    # ================= [Helper functions] =================

    def _load_user_memory(self, user_name: str) -> dict:
        path = os.path.join(MEMORY_DIR, f'{user_name}_memory.json')
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {"name": user_name, "history": {}}

    def warmup(self) -> None:
        """Server-startup warmup — preload the Ollama `embeddinggemma` model into memory.

        If `_get_embedding()` triggers a lazy load on the first user utterance it incurs a
        delay of several to a dozen-plus seconds, so we call a dummy embedding once at startup
        to load it ahead of time.
        Failure does not block server startup — it falls back to a lazy load on the first call.
        """
        t0 = time.time()
        try:
            self._get_embedding("워밍업")
            print(f"[Retrieval] Embedding model loaded ({time.time() - t0:.1f}s)")
        except Exception as e:
            print(f"[Retrieval] Embedding warmup failed (continuing): {e}")

    def finalize_session(self):
        """Called at session end — commit the turns accumulated in the buffer (session_turns) to FAISS in bulk.

        Usually a no-op, since the Phase 3 chat() path does not push into session_turns.
        Only the path where the build_memory.py subprocess pushes Phase 1 + noise turns directly
        and then calls this actually does anything.
        """
        if not self.session_turns:
            return

        success = 0
        for query, response, date, ts in self.session_turns:
            try:
                self._add_to_faiss(query, response, date, ts)
                success += 1
            except Exception as e:
                print(f"  Embedding failed (skipped): {e}")

        print(f"[Retrieval] FAISS index update complete ({success}/{len(self.session_turns)})")
        self.session_turns = []
