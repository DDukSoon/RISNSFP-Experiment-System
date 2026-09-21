# -*- coding: utf-8 -*-
"""Phase 2 memory build — single entry point.

Bundles the build CLI, the ``Phase2Builder`` pipeline, and file-only logging.
Generates the three per-condition memory artifacts (in_context, retrieval,
parametric) for one participant from Gemini-generated multi-turn scenarios
(one shared source for fairness). ``retrieval`` and ``parametric`` builds run
in subprocesses so llama_cpp model handles never cross process boundaries.

Usage:
    python build_memory.py -p P001
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib import error as urlerror
from urllib import request as urlrequest

from app.store import SessionStore

BASE_DIR = Path(__file__).resolve().parent
OLLAMA_URL = "http://127.0.0.1:11434/api/tags"


# ── File-only logging (stdout/stderr -> logs/<component>.log) ──────────
# Runtime output (ours + third-party libs like Unsloth/llama.cpp) is captured
# to a log file so the console stays quiet. A Python-level redirect does NOT
# propagate to child processes; capture a subprocess with
# ``stdout=open_append(...), stderr=STDOUT``.
_LOGS_DIR = BASE_DIR / "logs"


def open_append(component: str):
    """Open ``logs/<component>.log`` for appending (utf-8). Caller closes it."""
    _LOGS_DIR.mkdir(parents=True, exist_ok=True)
    return open(_LOGS_DIR / f"{component}.log", "a", encoding="utf-8")


def redirect_console_to_file(component: str):
    """Point ``sys.stdout``/``sys.stderr`` at ``logs/<component>.log`` (no console)."""
    f = open_append(component)
    sys.stdout = f
    sys.stderr = f
    return f


# ── Ollama daemon management ──────────────────────────────────────────
def _ollama_alive(timeout: float = 1.5) -> bool:
    try:
        with urlrequest.urlopen(OLLAMA_URL, timeout=timeout) as r:
            return r.status == 200
    except (urlerror.URLError, ConnectionError, TimeoutError, OSError):
        return False


def ensure_ollama(max_wait_sec: float = 20.0) -> bool:
    """Check whether the Ollama daemon is up; if not, spawn it in the background."""
    if _ollama_alive():
        return True

    exe = shutil.which("ollama")
    if not exe:
        print("[build_memory] WARN: 'ollama' executable not found on PATH. "
              "Please run 'ollama serve' manually.")
        return False

    try:
        creationflags = 0
        if sys.platform == "win32":
            creationflags = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
        subprocess.Popen(
            [exe, "serve"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=creationflags,
            close_fds=True,
        )
    except Exception as e:
        print(f"[build_memory] WARN: failed to spawn ollama: {e}")
        return False

    deadline = time.time() + max_wait_sec
    while time.time() < deadline:
        if _ollama_alive():
            return True
        time.sleep(0.5)
    print("[build_memory] WARN: Ollama did not respond. Manual check required.")
    return False


# ── Phase 2 builder ───────────────────────────────────────────────────
class Phase2Builder:
    """End-to-end Phase 2 memory build for one participant.

    parametric condition training: scenario_generator.py -> trainer_scenarios.py
    (Gemini multi-turn scenarios + sliding-window LoRA).
    """

    def __init__(self, base_dir: str | Path):
        self.base_dir = Path(base_dir)
        self.store = SessionStore(self.base_dir)
        self.training_dir = self.base_dir / "training"
        self.model_path = self.base_dir / "model_weights" / "gemma-3-4b-it-q4_0.gguf"
        self.status: str = "not_started"
        self.error: str | None = None

    # ── Public API ───────────────────────────────────────────────
    def run(self, pid: str) -> str:
        """Execute the full 3-condition build for ``pid``.

        Source data: flatten the Gemini scenarios and use them as a shared source
        across all three conditions (for fairness).

        Returns the final status string: ``done`` | ``error``.
        """
        try:
            print(f"\n[Phase2] {pid} starting memory build")

            log_path = self.store.log_path(pid, "phase1")
            if not log_path.exists():
                raise FileNotFoundError(f"Phase 1 log not found: {log_path}")

            # Start cost measurement
            build_cost = {
                "participant_id": pid,
                "timestamp": datetime.now().isoformat(),
                "pipeline": "scenario",
                "conditions": {},
            }

            # ── Determine the shared source turns (Gemini scenarios) ──────────────
            t_scgen = time.time()
            self._ensure_scenarios(pid)
            build_cost["scenario_generation_sec"] = round(time.time() - t_scgen, 2)
            source_turns = self._flatten_scenarios_to_turns(pid)
            # Recover Gemini usage from the scenario JSON (if present)
            scenarios_path = self.base_dir / "memory_data" / "scenarios" / f"{pid}_scenarios.json"
            with scenarios_path.open(encoding="utf-8") as f:
                sc_doc = json.load(f)
            build_cost["gemini_usage"] = sc_doc.get(
                "gemini_usage",
                {"note": "pre-existing scenarios, usage not tracked"},
            )
            build_cost["source"] = {"type": "gemini_scenarios",
                                     "num_scenarios": len(sc_doc.get("scenarios", [])),
                                     "total_turns": len(source_turns)}

            # ── Per-condition build + time each one ──────────────────────
            t0 = time.time()
            self._build_in_context(pid, source_turns)
            build_cost["conditions"]["in_context"] = {
                "wall_clock_sec": round(time.time() - t0, 2),
                "note": "file I/O only",
            }
            print(f"[Phase2] [1/3] done ({build_cost['conditions']['in_context']['wall_clock_sec']}s)")

            t0 = time.time()
            self._build_retrieval(pid, source_turns)
            build_cost["conditions"]["retrieval"] = {
                "wall_clock_sec": round(time.time() - t0, 2),
                "embed_model": "embeddinggemma",
                "embed_params": 307_580_000,
                "embed_calls": len(source_turns),
                "note": "embeddinggemma forward x n_turns + FAISS index",
            }
            print(f"[Phase2] [2/3] done ({build_cost['conditions']['retrieval']['wall_clock_sec']}s)")

            t0 = time.time()
            self._build_parametric(pid, source_turns)
            build_cost["conditions"]["parametric"] = {
                "wall_clock_sec": round(time.time() - t0, 2),
                "base_model": "gemma-3-4b-it-bnb-4bit",
                "base_params": 4_316_473_712,
                "lora_trainable_params": 16_394_240,
                "note": "Unsloth LoRA training + GGUF conversion (scenario generation excluded; see scenario_generation_sec above)",
            }
            print(f"[Phase2] [3/3] done ({build_cost['conditions']['parametric']['wall_clock_sec']}s)")

            # Save the cost file
            cost_path = self.base_dir / "participants" / pid / "build_cost.json"
            cost_path.parent.mkdir(parents=True, exist_ok=True)
            with cost_path.open("w", encoding="utf-8") as f:
                json.dump(build_cost, f, ensure_ascii=False, indent=2)

            s = self.store.get_status(pid)
            s["phase2_status"] = "done"
            self.store.save_status(pid, s)
            self.status = "done"
            print(f"[Phase2] {pid} all memory build complete")
            return "done"

        except Exception as e:
            print(f"[Phase2] error: {e}")
            import traceback
            traceback.print_exc()
            s = self.store.get_status(pid)
            s["phase2_status"] = "error"
            s["phase2_error"] = str(e)
            self.store.save_status(pid, s)
            self.status = "error"
            self.error = str(e)
            return "error"

    # ── Helpers ──────────────────────────────────────────────────
    def _ensure_scenarios(self, pid: str) -> None:
        """If the scenario JSON is missing, generate it by invoking scenario_generator.py."""
        scenarios_path = self.base_dir / "memory_data" / "scenarios" / f"{pid}_scenarios.json"
        if scenarios_path.exists():
            return

        gen_script = self.training_dir / "scenario_generator.py"
        with open_append("phase2") as _lf:
            subprocess.run(
                [sys.executable, str(gen_script), "-p", pid],
                env={**os.environ, "PYTHONIOENCODING": "utf-8"},
                cwd=str(self.base_dir),
                stdout=_lf, stderr=subprocess.STDOUT,
                check=True,
            )
        if not scenarios_path.exists():
            raise RuntimeError(
                f"Scenario generation failed: {scenarios_path} was not created. "
                f"Check GEMINI_API_KEY / network."
            )

    def _flatten_scenarios_to_turns(self, pid: str) -> list[dict[str, Any]]:
        """Convert the scenario JSON into a flat turn list.

        Each (user_i, assistant_{i+1}) pair in a scenario is grouped into one turn.
        10 scenarios x 10 pairs = roughly 100 turns.
        Since this is the shared source for all three conditions, no _noise tag is
        applied (everything is personalization-related).
        """
        scenarios_path = self.base_dir / "memory_data" / "scenarios" / f"{pid}_scenarios.json"
        with scenarios_path.open(encoding="utf-8") as f:
            doc = json.load(f)

        flat: list[dict[str, Any]] = []
        turn_num = 1
        now_iso = datetime.now().isoformat()
        for sc in doc.get("scenarios", []):
            conv = sc.get("conversation", [])
            i = 0
            while i + 1 < len(conv):
                if conv[i].get("role") == "user" and conv[i + 1].get("role") == "assistant":
                    flat.append({
                        "turn": turn_num,
                        "timestamp": now_iso,
                        "user_text": conv[i].get("content", ""),
                        "agent_text": conv[i + 1].get("content", ""),
                        "latency": {},
                        "_scenario_id": sc.get("scenario_id"),
                    })
                    turn_num += 1
                i += 2
        return flat

    def _turns_to_memory_json(self, pid: str, turns: list[dict[str, Any]]) -> dict:
        date = datetime.now().strftime("%Y-%m-%d")
        history_entries: list[dict[str, Any]] = []
        for t in turns:
            ts_raw = t.get("timestamp", "")
            entry = {
                "query": t["user_text"],
                "response": t["agent_text"],
                "timestamp": ts_raw.split("T")[-1][:8] if "T" in ts_raw else "00:00:00",
            }
            history_entries.append(entry)
        # "name" is an identifier for the JSON round-trip. It must never leak into
        # the subject of an LLM prompt / training instruction. pid is kept only in
        # participant_id.
        return {
            "name": "사용자",
            "participant_id": pid,
            "history": {date: history_entries},
        }

    def _build_in_context(self, pid: str, turns: list[dict[str, Any]]) -> None:
        mem_dir = self.base_dir / "memory_data" / "in_context"
        mem_dir.mkdir(parents=True, exist_ok=True)
        mem = self._turns_to_memory_json(pid, turns)
        with (mem_dir / f"{pid}_memory.json").open("w", encoding="utf-8") as f:
            json.dump(mem, f, ensure_ascii=False, indent=2)

    def _build_retrieval(self, pid: str, turns: list[dict[str, Any]]) -> None:
        mem_dir = self.base_dir / "memory_data" / "retrieval"
        mem_dir.mkdir(parents=True, exist_ok=True)
        mem = self._turns_to_memory_json(pid, turns)
        with (mem_dir / f"{pid}_memory.json").open("w", encoding="utf-8") as f:
            json.dump(mem, f, ensure_ascii=False, indent=2)

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        ) as tf:
            json.dump(turns, tf, ensure_ascii=False)
            turns_tmp = tf.name

        code = f"""
import sys, json, time
sys.path.insert(0, r'{self.base_dir}')
from agents.memory_retrieval import retrieval

ctx = retrieval(model_path=r'{self.model_path}', embed_model_name='embeddinggemma', temperature=0.2)
ctx.initialize_user('{pid}')

with open(r'{turns_tmp}', encoding='utf-8') as f:
    turns = json.load(f)

date = time.strftime('%Y-%m-%d')
for t in turns:
    ts = t.get('timestamp', '00:00:00')
    if 'T' in ts:
        ts = ts.split('T')[-1][:8]
    ctx.session_turns.append((t['user_text'], t['agent_text'], date, ts))

ctx.finalize_session()
"""
        try:
            with open_append("phase2") as _lf:
                subprocess.run(
                    [sys.executable, "-c", code],
                    env={**os.environ, "PYTHONIOENCODING": "utf-8",
                         "PYTHONPATH": str(self.base_dir) + os.pathsep + os.environ.get("PYTHONPATH", "")},
                    cwd=str(self.base_dir),
                    stdout=_lf, stderr=subprocess.STDOUT,
                    check=True,
                )
        finally:
            if os.path.exists(turns_tmp):
                os.unlink(turns_tmp)

    def _build_parametric(self, pid: str, turns: list[dict[str, Any]]) -> None:
        """LoRA training for the parametric condition (scenario-based)."""
        mem_dir = self.base_dir / "memory_data" / "parametric"
        mem_dir.mkdir(parents=True, exist_ok=True)
        mem_path = mem_dir / f"{pid}_memory.json"
        mem = self._turns_to_memory_json(pid, turns)
        with mem_path.open("w", encoding="utf-8") as f:
            json.dump(mem, f, ensure_ascii=False, indent=2)

        self._run_scenario_trainer(pid)

        # Validate artifacts — common
        weights_dir = self.base_dir / "model_weights"
        gguf_path = weights_dir / f"{pid}_lora_persona.gguf"
        adapter_dir = weights_dir / f"{pid}_lora_raw"
        if not gguf_path.exists():
            raise RuntimeError(
                f"Missing parametric training artifact: {gguf_path} not found "
                f"(adapter_raw exists: {adapter_dir.exists()}). "
                f"Check logs/phase2.log."
            )

    # ── parametric LoRA training (scenario-based) ──────────────────────
    def _run_scenario_trainer(self, pid: str) -> None:
        """Invoke trainer_scenarios.py. Scenarios are already secured in run()."""
        trainer = self.training_dir / "trainer_scenarios.py"
        with open_append("phase2") as _lf:
            subprocess.run(
                [sys.executable, str(trainer), "-p", pid],
                env={**os.environ, "PYTHONIOENCODING": "utf-8"},
                cwd=str(self.base_dir),
                stdout=_lf, stderr=subprocess.STDOUT,
                check=True,
            )


# ── CLI ───────────────────────────────────────────────────────────────
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Phase 2 memory build (manual).")
    parser.add_argument("-p", "--participant", required=True,
                        help="Participant ID, e.g. P001")
    parser.add_argument("--debug", action="store_true",
                        help="Print to the console instead of redirecting to logs/phase2.log")
    args = parser.parse_args(argv)
    pid = args.participant

    store = SessionStore(BASE_DIR)
    log_path = store.log_path(pid, "phase1")
    if not log_path.exists():
        print(f"[build_memory] ERROR: Phase 1 log not found: {log_path}")
        print("                Please run Phase 1 first.")
        return 2

    if not args.debug:
        # Redirect all pipeline output (ours + Unsloth/llama native) to a log file.
        print(f"[build_memory] Phase 2 build for {pid} — progress → logs/phase2.log")
        redirect_console_to_file("phase2")

    print(f"[build_memory] participant={pid} starting Phase 2 build")

    ensure_ollama()

    builder = Phase2Builder(BASE_DIR)
    status = builder.run(pid)

    print(f"\n[build_memory] final phase2_status = {status}")
    if status == "done":
        print("[build_memory] Done. You can now start the Phase 3 server:")
        print(f"               python server.py -p {pid} --phase 3 -c 0   # condition_order[0]")
        print(f"               python server.py -p {pid} --phase 3 -c 1   # condition_order[1]")
        print(f"               python server.py -p {pid} --phase 3 -c 2   # condition_order[2]")
        print(f"               (per-participant condition_order: see participants/{pid}/status.json)")
        return 0
    print("[build_memory] Failed. Check phase2_error in status.json.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
