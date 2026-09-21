# -*- coding:utf-8 -*-
"""CLI comparison test of the three Phase 3 memory conditions.

Skips server/STT/TTS (no Unity) and directly loads each condition's backend
(in_context / retrieval / parametric), running the same prompt set and
printing the inference results.

Usage:
    python tools/test_conditions_cli.py -p P001

The Phase 2 artifacts for the same `-p` value must be ready:
    memory_data/in_context/<pid>_memory.json
    memory_data/retrieval/<pid>_memory.json  (+ FAISS index)
    model_weights/<pid>_lora_persona.gguf
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path


# Fix Korean output encoding (Windows)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Add the project root to PYTHONPATH (for direct script execution)
_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


# Default Phase 3 test prompts — correspond to the 10 fixed Phase 1 questions.
# All three conditions are compared with the same questions (fairness).
DEFAULT_PROMPTS = [
    "내 이름 뭐야?",
    "나 몇 살이야?",
    "내 전공이 뭐였지?",
    "나 어디 산다고 했지?",
    "내 취미 기억해?",
    "내가 좋아하는 음식 뭐지?",
    "나 운동 뭐 한다고 했지?",
    "최근에 어디 여행 갔다왔는지 알아?",
    "내가 자주 보는 콘텐츠 뭐지?",
    "내가 요즘 관심 있는 거 뭐야?",
]

# Natural conversation flow — does not ask facts directly but **elicits via context**.
# Tests whether the AI proactively recalls memories (proactive recall).
# Each turn continues after seeing the previous turn's response, so it tests multi-turn behavior.
NATURAL_DIALOG_PROMPTS = [
    "안녕 오늘 하루 어땠어?",                           # T1 casual greeting
    "요즘 좀 바쁘네. 뭐 좀 재밌는 거 없나",               # T2 elicit interests/hobbies recall
    "저녁 뭐 먹을지 고민중이야",                         # T3 elicit food preference recall
    "운동 좀 해야할 것 같은데",                          # T4 elicit exercise habit recall
    "주말에 어디 놀러가고 싶다",                         # T5 elicit travel experience recall
    "요즘 투자 쪽이 궁금해져서",                         # T6 elicit topic-of-interest recall
    "너는 나에 대해서 뭐 기억하고 있어?",                 # T7 final direct recall (verification)
]

PROMPT_SETS = {
    "factual": DEFAULT_PROMPTS,
    "natural": NATURAL_DIALOG_PROMPTS,
}


def _banner(title: str) -> None:
    line = "═" * 70
    print(f"\n{line}\n  {title}\n{line}")


def _run_condition(label: str, backend, prompts: list[str]) -> list[dict]:
    """Call the given backend's chat() on each prompt sequentially and collect results."""
    _banner(f"[{label}] inference start")
    results = []
    for i, q in enumerate(prompts, 1):
        t0 = time.time()
        try:
            ans = backend.chat(q)
        except Exception as e:
            ans = f"<ERROR: {e}>"
        dt = (time.time() - t0) * 1000
        # llama_cpp usage — all three modules update last_prompt_tokens / last_total_tokens.
        prompt_tok = int(getattr(backend, "last_prompt_tokens", 0) or 0)
        compl_tok = int(getattr(backend, "last_completion_tokens", 0) or 0)
        total_tok = int(getattr(backend, "last_total_tokens", 0) or 0)
        print(f"\nQ{i}. {q}")
        print(f"A{i}. {ans}")
        print(f"     ({dt:.0f}ms | prompt={prompt_tok} compl={compl_tok} total={total_tok})")
        results.append({
            "q": q, "a": ans, "ms": dt,
            "prompt_tokens": prompt_tok, "completion_tokens": compl_tok,
            "total_tokens": total_tok,
        })
    return results


def test_in_context(pid: str, model_path: str, prompts: list[str], temperature: float) -> list[dict]:
    from agents.memory_incontext import in_context
    backend = in_context(model_path=model_path, temperature=temperature)
    init_msg = backend.initialize_user(pid)
    print(f"[in_context init] {init_msg}")
    try:
        return _run_condition("in_context", backend, prompts)
    finally:
        # Reclaim VRAM (before loading the next condition)
        del backend


def test_retrieval(pid: str, model_path: str, prompts: list[str], temperature: float) -> list[dict]:
    from agents.memory_retrieval import retrieval
    backend = retrieval(
        model_path=model_path,
        embed_model_name="embeddinggemma",
        temperature=temperature,
    )
    init_msg = backend.initialize_user(pid)
    print(f"[retrieval init] {init_msg}")
    try:
        return _run_condition("retrieval", backend, prompts)
    finally:
        del backend


def test_parametric(pid: str, model_path: str, prompts: list[str], temperature: float) -> list[dict]:
    from agents.memory_parametric import parametric
    backend = parametric(model_path=model_path, temperature=temperature)
    init_msg = backend.initialize_user(pid)
    print(f"[parametric init] {init_msg}")
    try:
        return _run_condition("parametric", backend, prompts)
    finally:
        del backend


def _summary_table(all_results: dict[str, list[dict]], prompts: list[str]) -> None:
    """Compare the three conditions' responses side by side, per question."""
    _banner("Three-condition response comparison (by question)")
    for i, q in enumerate(prompts, 1):
        print(f"\nQ{i}. {q}")
        for label in ("in_context", "retrieval", "parametric"):
            rows = all_results.get(label, [])
            ans = rows[i - 1]["a"] if i - 1 < len(rows) else "<missing>"
            short = ans if len(ans) < 100 else ans[:97] + "..."
            print(f"  [{label:13s}] {short}")

    # Context usage statistics — per-condition average / min / max and first vs last turn
    _banner("Context usage by condition (prompt_tokens)")
    print(f"{'condition':15s}  {'avg':>7s}  {'min':>5s}  {'max':>5s}  {'first':>5s}  {'last':>5s}")
    for label in ("in_context", "retrieval", "parametric"):
        rows = all_results.get(label, [])
        if not rows:
            continue
        ptoks = [r.get("prompt_tokens", 0) for r in rows]
        avg = sum(ptoks) / len(ptoks) if ptoks else 0
        print(
            f"{label:15s}  {avg:7.0f}  {min(ptoks):5d}  {max(ptoks):5d}  "
            f"{ptoks[0]:5d}  {ptoks[-1]:5d}"
        )


def _parse_args():
    p = argparse.ArgumentParser(description="CLI comparison test of the three Phase 3 conditions")
    p.add_argument("-p", "--pid", required=True, help="Participant ID (e.g. P001)")
    p.add_argument(
        "--only",
        choices=["in_context", "retrieval", "parametric"],
        help="Test only a specific condition (defaults to all three)",
    )
    p.add_argument(
        "-t", "--temperature",
        type=float, default=0.7,
        help="Shared inference temperature for all three conditions (default 0.7)",
    )
    p.add_argument(
        "--prompts",
        choices=list(PROMPT_SETS.keys()),
        default="factual",
        help="Prompt set: factual (10 direct questions) / natural (7-turn natural dialog)",
    )
    return p.parse_args()


def main():
    args = _parse_args()
    pid = args.pid

    model_path = str(_ROOT / "model_weights" / "gemma-3-4b-it-q4_0.gguf")
    if not os.path.exists(model_path):
        print(f"base model not found: {model_path}")
        sys.exit(2)

    prompts = PROMPT_SETS[args.prompts]
    targets = [args.only] if args.only else ["in_context", "retrieval", "parametric"]

    all_results: dict[str, list[dict]] = {}

    runners = {
        "in_context": test_in_context,
        "retrieval": test_retrieval,
        "parametric": test_parametric,
    }

    for label in targets:
        try:
            all_results[label] = runners[label](
                pid, model_path, prompts, args.temperature
            )
        except Exception as e:
            print(f"\n[{label}] failed entirely: {e}")
            traceback.print_exc()

    if len(all_results) > 1:
        _summary_table(all_results, prompts)

    # Save results — cost_report.py reads this file when aggregating Phase 3 runtime.
    # natural mode is saved under a separate filename (does not overwrite factual results)
    suffix = f"_{args.prompts}" if args.prompts != "factual" else ""
    out_name = f"cli_test_results{suffix}.json"
    out_path = _ROOT / "participants" / pid / out_name
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump({
            "participant_id": pid,
            "timestamp": datetime.now().isoformat(),
            "temperature": args.temperature,
            "prompt_set": args.prompts,
            "prompts": prompts,
            "results": all_results,
        }, f, ensure_ascii=False, indent=2)
    print(f"\nCLI test results saved: {out_path}")


if __name__ == "__main__":
    main()
